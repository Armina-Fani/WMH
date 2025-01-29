import os
#import easybar
import shutil
import time

from catalyst import dl, metrics, utils
from catalyst.data import BatchPrefetchLoaderWrapper
from catalyst.data.sampler import DistributedSamplerWrapper

from pymongo.errors import OperationFailure

import torch
from torch.optim.lr_scheduler import (
    OneCycleLR,
)
from torch.utils.data import DataLoader

from dice import faster_dice, DiceLoss
from meshnet import enMesh_checkpoint, enMesh
from mindfultensors.gencoords import CoordsGenerator
from utils import (
    qnormalize,
    DBBatchSampler,
)

from sklearn.model_selection import KFold

from mindfultensors.mongoloader import (
    create_client,
    collate_subcubes,
    mcollate,
    MongoDataset,
    MongoClient,
    mtransform,
)

# SEED = 0
# utils.set_global_seed(SEED)
# utils.prepare_cudnn(deterministic=False, benchmark=False) # crashes everything

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:100"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
#os.environ["NCCL_SOCKET_IFNAME"] = "ib0"
#os.environ["NCCL_P2P_LEVEL"] = "NVL"

volume_shape = [256] * 3
MAXSHAPE = 300

LABELFIELD = "label"
DATAFIELD = "t1"

n_classes = 2

MONGOHOST = "arctrdcn018.rs.gsu.edu"  #"10.245.12.58"  # "arctrdcn018.rs.gsu.edu"
DBNAME = "WMH"
COLLECTION = "main"
INDEX_ID = "id"
config_file = "modelAE.json"


class WireMongoDataset(MongoDataset):
    def __init__(self, *args, keeptrying=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.keeptrying = keeptrying  # Initialize the keeptrying attribute

    def retry_on_eof_error(retry_count, verbose=False):
        def decorator(func):
            def wrapper(self, batch, *args, **kwargs):
                myException = Exception  # Default Exception if not overwritten
                for attempt in range(retry_count):
                    try:
                        return func(self, batch, *args, **kwargs)
                    except (
                        EOFError,
                        OperationFailure,
                    ) as e:  # Specifically catching EOFError
                        if self.keeptrying:
                            if verbose:
                                print(
                                    f"EOFError caught. Retrying {attempt+1}/{retry_count}"
                                )
                            time.sleep(1)
                            myException = e
                            continue
                        else:
                            raise e
                raise myException("Failed after multiple retries.")

            return wrapper

        return decorator

    @retry_on_eof_error(retry_count=5, verbose=True)
    def __getitem__(self, batch):
        # Directly use the parent class's __getitem__ method
        # The decorator will handle exceptions
        return super().__getitem__(batch)


# CustomRunner – PyTorch for-loop decomposition
# https://github.com/catalyst-team/catalyst#minimal-examples
class CustomRunner(dl.Runner):
    def __init__(
        self,
        logdir: str,
        wandb_project: str,
        wandb_experiment: str,
        model_path: str,
        n_channels: int,
        n_classes: int,
        n_epochs: int,
        optimize_inline: bool,        
        validation_percent: float,
        onecycle_lr: float,
        rmsprop_lr: float,
        num_subcubes: int,
        num_volumes: int,
        client_creator,
        off_brain_weight: float,
        prefetches=8,
        volume_shape=[256] * 3,
        subvolume_shape=[256] * 3,
        db_host=MONGOHOST,
        db_name=DBNAME,
        db_collection=COLLECTION,
        **kwargs,
        ):
        super().__init__()
        self._logdir = logdir
        self.wandb_project = wandb_project
        self.wandb_experiment = wandb_experiment
        self.model_path = model_path
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.optimize_inline = optimize_inline
        self.validation_percent = validation_percent
        self.onecycle_lr = onecycle_lr
        self.rmsprop_lr = rmsprop_lr
        self.prefetches = prefetches
        self.db_host = db_host
        self.db_name = db_name
        self.db_collection = db_collection
        self.shape = subvolume_shape[0]
        self.num_subcubes = num_subcubes
        self.num_volumes = num_volumes
        self.n_epochs = n_epochs
        self.off_brain_weight = off_brain_weight
        self.client_creator = client_creator
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.funcs = None
        self.collate = None

        self.train_indices = kwargs.get("train_indices", None)
        self.val_indices = kwargs.get("val_indices", None)

    def get_engine(self):
        if torch.cuda.device_count() > 1:
            return dl.DistributedDataParallelEngine(
                # mixed_precision='fp16',
                # ddp_kwargs={"find_unused_parameters": True, "backend": "nccl"}
                process_group_kwargs={"backend": "nccl"}
            )
        else:
            return dl.GPUEngine()

    def get_loggers(self):
        return {
            "console": dl.ConsoleLogger(),
            "csv": dl.CSVLogger(logdir=self._logdir),
            # "tensorboard": dl.TensorboardLogger(logdir=self._logdir,
            #                                     log_batch_metrics=True),
            "wandb": dl.WandbLogger(
                project=self.wandb_project,
                name=self.wandb_experiment,
                log_batch_metrics=True,
            ),
        }

    @property
    def stages(self):
        return ["train", "valid"]

    @property
    def num_epochs(self) -> int:
        return self.n_epochs

    @property
    def seed(self) -> int:
        """Experiment's seed for reproducibility."""
        random_data = os.urandom(4)
        SEED = int.from_bytes(random_data, byteorder="big")
        utils.set_global_seed(SEED)
        return SEED

    def get_stage_len(self) -> int:
        return self.n_epochs

    def get_loaders(self):
        self.funcs = {
            "createclient": self.client_creator.create_client,
            "mycollate": self.client_creator.mycollate,
            "mycollate_full": self.client_creator.mycollate_full,
            "mytransform": self.client_creator.mytransform,
        }

        self.collate = (
            self.funcs["mycollate_full"]
            if self.shape == 256
            else self.funcs["mycollate"]
        )

        client = MongoClient("mongodb://" + self.db_host + ":27017")
        db = client[self.db_name]
        posts = db[self.db_collection + ".bin"]
        num_examples = int(posts.find_one(sort=[(INDEX_ID, -1)])[INDEX_ID] + 1)

        tdataset = WireMongoDataset(
            [int(idx) for idx in self.train_indices],
            self.funcs["mytransform"],
            None,
            (DATAFIELD, LABELFIELD),
            normalize=qnormalize,
            id=INDEX_ID,
        )

        tsampler = (
            DistributedSamplerWrapper(
                DBBatchSampler(tdataset, batch_size=self.num_volumes)
            )
            if self.engine.is_ddp
            else DBBatchSampler(tdataset, batch_size=self.num_volumes)
        )

        tdataloader = BatchPrefetchLoaderWrapper(
            DataLoader(
                tdataset,
                sampler=tsampler,
                collate_fn=self.collate,
                pin_memory=True,
                worker_init_fn=self.funcs["createclient"],
                persistent_workers=True,
                prefetch_factor=3,
                num_workers=self.prefetches,
            ),
            num_prefetches=self.prefetches,
        )
        
        vdataset = WireMongoDataset(
            [int(idx) for idx in self.val_indices],
            #range(num_examples - int(self.validation_percent * num_examples), num_examples),
            self.funcs["mytransform"],
            None,
            (DATAFIELD, LABELFIELD),
            normalize=qnormalize,
            id=INDEX_ID,
        )

        vsampler = (
            DistributedSamplerWrapper(
                DBBatchSampler(vdataset, batch_size=1)
            )
            if self.engine.is_ddp
            else DBBatchSampler(vdataset, batch_size=self.num_volumes)
        )

        vdataloader = BatchPrefetchLoaderWrapper(
            DataLoader(
                vdataset,
                sampler=vsampler,
                collate_fn=self.collate,
                pin_memory=True,
                worker_init_fn=self.funcs["createclient"],
                persistent_workers=True,
                num_workers=8,
            ),
            num_prefetches=8,
        )

        return {"train": tdataloader, "valid": vdataloader}

    def get_model(self):
        if self.shape > MAXSHAPE:
            model = enMesh(
                in_channels=1,
                n_classes=self.n_classes,
                channels=self.n_channels,
                config_file=config_file,
                optimize_inline=self.optimize_inline,
            )
        else:
            model = enMesh_checkpoint(
                in_channels=1,
                n_classes=self.n_classes,
                channels=self.n_channels,
                config_file=config_file,
            )
        return model

    def get_criterion(self):
        class_weight = torch.FloatTensor(
            [self.off_brain_weight] + [1.0] * (self.n_classes - 1)
        ).to(self.engine.device)
        ce_criterion = torch.nn.CrossEntropyLoss(
            weight=class_weight, label_smoothing=0.01
        )
        dice_criterion = DiceLoss()

        def combined_loss(output, target):
            ce_loss = ce_criterion(output, target)
            dice_loss = dice_criterion(output, target)
            return dice_loss

        return combined_loss

    #def get_criterion_(self):
    #    class_weight = torch.FloatTensor(
    #        [self.off_brain_weight] + [1.0] * (self.n_classes - 1)
    #    ).to(self.engine.device)
    #    criterion = torch.nn.CrossEntropyLoss(
    #        weight=class_weight, label_smoothing=0.01
    #    )
    #    # criterion = DiceLoss()
    #    return criterion

    def get_optimizer(self, model):
        # optimizer = torch.optim.RMSprop(model.parameters(), lr=self.rmsprop_lr)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.rmsprop_lr)
        return optimizer

    def get_scheduler(self, optimizer):
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.onecycle_lr,
            div_factor=100,
            pct_start=0.2,
            epochs=self.num_epochs,
            steps_per_epoch=len(self.loaders["train"]),
        )
        return scheduler

    def get_callbacks(self):
        checkpoint_params = {"save_best": True, "metric_key": "macro_dice"}
        if self.model_path:
            checkpoint_params.update({"resume_model": self.model_path})
        return {
            "checkpoint": dl.CheckpointCallback(
                self._logdir, **checkpoint_params
            ),
            "tqdm": dl.TqdmCallback(),
        }

    def on_loader_start(self, runner):
        """
        Calls runner methods when the dataloader begins and adds
        metrics for loss and macro_dice
        """
        super().on_loader_start(runner)
        self.meters = {
            key: metrics.AdditiveValueMetric(compute_on_call=False)
            for key in ["loss", "macro_dice", "learning rate"]
        }

    def on_loader_end(self, runner):
        """
        Calls runner methods when a dataloader finishes running and updates
        metrics
        """
        for key in ["loss", "macro_dice", "learning rate"]:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)

    # model train/valid step
    def handle_batch(self, batch):
        # unpack the batch
        sample, label = batch
        #label = merge_homologs(label, self.engine.device)
        sample = sample.to(torch.float32)
        label = label.to(torch.int64)
        label[label == 2] = 0
        # np.save("labels.npy", label.cpu().numpy())
        # np.save("input.npy", sample.cpu().numpy())
        # stop
        # run model forward/backward pass
        if self.model.training:
            if self.shape > MAXSHAPE:
                if self.engine.is_ddp:
                    with self.model.no_sync():
                        loss, y_hat = self.model.forward(
                            x=sample,
                            y=label,
                            loss=self.criterion,
                            verbose=False,
                        )
                    torch.distributed.barrier()
                else:
                    loss, y_hat = self.model.forward(
                        x=sample, y=label, loss=self.criterion, verbose=False
                    )
            else:
                y_hat = self.model.forward(sample)
                loss = self.criterion(y_hat, label)
                loss.backward()
            if not self.optimize_inline:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
        else:
            with torch.no_grad():
                y_hat = self.model.forward(sample)
                loss = self.criterion(y_hat, label)
        with torch.inference_mode():
            result = torch.squeeze(torch.argmax(y_hat, 1)).long()
            labels = torch.squeeze(label)
            dice = torch.mean(
                faster_dice(result, labels, range(self.n_classes))
            )

        self.batch_metrics.update(
            {
                "loss": loss,
                "macro_dice": dice,
                "learning rate": torch.tensor(
                    self.optimizer.param_groups[0]["lr"]
                ),
            }
        )

        for key in ["loss", "macro_dice", "learning rate"]:
            self.meters[key].update(
                self.batch_metrics[key].item(), self.num_volumes
            )

        del sample
        del label
        del y_hat
        del result
        del labels
        del loss


class ClientCreator:
    def __init__(self, dbname, mongohost, volume_shape=[256] * 3):
        self.dbname = dbname
        self.mongohost = mongohost
        self.volume_shape = volume_shape
        self.subvolume_shape = None
        self.collection = None
        self.num_subcubes = None

    def set_shape(self, shape):
        self.subvolume_shape = shape
        self.coord_generator = CoordsGenerator(
            self.volume_shape, self.subvolume_shape
        )

    def set_collection(self, collection):
        self.collection = collection

    def set_num_subcubes(self, num_subcubes):
        self.num_subcubes = num_subcubes

    def create_client(self, x):
        return create_client(
            x,
            dbname=self.dbname,
            colname=self.collection,
            mongohost=self.mongohost,
        )

    def mycollate(self, x):
        return collate_subcubes(
            x,
            self.coord_generator,
            samples=self.num_subcubes,
        )

    def mycollate_full(self, x):
        return mcollate(x)

    def mytransform(self, x):
        return mtransform(x)


def assert_equal_length(*args):
    assert all(
        len(arg) == len(args[0]) for arg in args
    ), "Not all parameter lists have the same length!"


if __name__ == "__main__":
    # hparams
    validation_percent = 0
    optimize_inline = False

    model_channels = 15
    model_label = "_ss"

    #model_path = f"./logs/tmp/curriculum_enmesh_15channels_babyMasks_DiceLoss-res/model.best.pth"
    #model_path = "./logs/tmp/15_MN_wmh/model.last.pth"
    model_path = ""
    logdir = f"./logs/tmp/{model_channels}_MN_wmh_10kfold/"
    wandb_project = f"{model_channels}_MN_wmh_10kfold"

    client_creator = ClientCreator(DBNAME, MONGOHOST)

    # set up parameters of your experiment
    cubesizes = [256] * 21
    numcubes = [1] + [1, 1] * 10
    numvolumes = [1] + [1, 1] * 10
    weights = [1] + [1] * 20  # weights for the 0-class
    collections = ["main"] * 21
    epochs = [10] + [10, 10] * 10
    prefetches = [24] * 21
    attenuates = [1] * 21

    assert_equal_length(
        cubesizes,
        numcubes,
        numvolumes,
        weights,
        collections,
        epochs,
        prefetches,
        attenuates,
    )

    k_folds = 10  # Number of folds
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)


    client = MongoClient("mongodb://" + MONGOHOST + ":27017")
    db = client[DBNAME]
    posts = db[COLLECTION + ".bin"]
    num_examples = int(posts.find_one(sort=[(INDEX_ID, -1)])[INDEX_ID] + 1)
    indices = list(range(num_examples))
    
    start_experiment = 0  

    for fold, (train_indices, val_indices) in enumerate(kf.split(indices)):
        print(f"Starting Fold {fold + 1}/{k_folds}")

        for experiment in range(len(cubesizes)):
            COLLECTION = collections[experiment]
            num_subcubes = numcubes[experiment]
            num_volumes = numvolumes[experiment]

            off_brain_weight = weights[experiment]
            subvolume_shape = [cubesizes[experiment]] * 3
            attenuates = [0.95**x for x in range(21)]
            onecycle_lr = rmsprop_lr = (
                attenuates[experiment] * 0.05 * 8 * num_subcubes * num_volumes / 256
            )
            n_epochs = epochs[experiment]
            n_fetch = prefetches[experiment]
            wandb_experiment = (
                f"{start_experiment + experiment:02} cube dice only"
                + str(subvolume_shape[0])
                + " "
                + COLLECTION
                + model_label
            )

            # Set database parameters
            client_creator.set_collection(COLLECTION)
            client_creator.set_num_subcubes(num_subcubes)
            client_creator.set_shape(subvolume_shape)

            runner = CustomRunner(
                logdir=logdir,
                wandb_project=wandb_project,
                wandb_experiment=wandb_experiment,
                model_path=model_path,
                n_channels=model_channels,
                n_classes=n_classes,
                n_epochs=n_epochs,
                optimize_inline=optimize_inline,
                validation_percent=validation_percent,
                onecycle_lr=onecycle_lr,
                rmsprop_lr=rmsprop_lr,
                num_subcubes=num_subcubes,
                num_volumes=num_volumes,
                client_creator=client_creator,
                off_brain_weight=off_brain_weight,
                prefetches=n_fetch,
                db_collection=COLLECTION,
                subvolume_shape=subvolume_shape,
                train_indices=train_indices,
                val_indices=val_indices
            )      
            runner.run()

            shutil.copy(
                logdir + "/model.last.pth",
                logdir + "/model.last." + str(subvolume_shape[0]) + ".pth",
            )

            model_path = logdir + "/model.last.pth"
