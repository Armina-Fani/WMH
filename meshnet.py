from collections import OrderedDict
import gc
import ipdb
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import jvp
from torch.utils.checkpoint import checkpoint_sequential
import json
import copy


def make_functional(mod, disable_autograd_tracking=False):
    # https://gist.github.com/zou3519/7769506acc899d83ef1464e28f22e6cf
    params_dict = dict(mod.named_parameters())
    params_names = params_dict.keys()
    params_values = tuple(params_dict.values())

    stateless_mod = copy.deepcopy(mod)
    stateless_mod.to("meta")

    def fmodel(new_params_values, *args, **kwargs):
        new_params_dict = {
            name: value for name, value in zip(params_names, new_params_values)
        }
        return torch.func.functional_call(
            stateless_mod, new_params_dict, args, kwargs
        )

    if disable_autograd_tracking:
        params_values = torch.utils._pytree.tree_map(
            torch.Tensor.detach, params_values
        )
    return fmodel, params_values


def make_functional_with_buffers(mod, disable_autograd_tracking=False):
    # https://gist.github.com/zou3519/7769506acc899d83ef1464e28f22e6cf
    params_dict = dict(mod.named_parameters())
    params_names = params_dict.keys()
    params_values = tuple(params_dict.values())

    buffers_dict = dict(mod.named_buffers())
    buffers_names = buffers_dict.keys()
    buffers_values = tuple(buffers_dict.values())

    stateless_mod = copy.deepcopy(mod)
    stateless_mod.to("meta")

    def fmodel(new_params_values, new_buffers_values, *args, **kwargs):
        new_params_dict = {
            name: value for name, value in zip(params_names, new_params_values)
        }
        new_buffers_dict = {
            name: value
            for name, value in zip(buffers_names, new_buffers_values)
        }
        return torch.func.functional_call(
            stateless_mod, (new_params_dict, new_buffers_dict), args, kwargs
        )

    if disable_autograd_tracking:
        params_values = torch.utils._pytree.tree_map(
            torch.Tensor.detach, params_values
        )
    return fmodel, params_values, buffers_values


def set_channel_num(config, in_channels, n_classes, channels):
    """
    Takes a configuration json for a convolutional neural network of MeshNet architecture and changes it to have the specified number of input channels, output classes, and number of channels that each layer except the input and output layers have.

    Args:
        config (dict): The configuration json for the network.
        in_channels (int): The number of input channels.
        n_classes (int): The number of output classes.
        channels (int): The number of channels that each layer except the input and output layers will have.

    Returns:
        dict: The updated configuration json.
    """
    # input layer
    config["layers"][0]["in_channels"] = in_channels
    config["layers"][0]["out_channels"] = channels

    # output layer
    config["layers"][-1]["in_channels"] = channels
    config["layers"][-1]["out_channels"] = n_classes

    # hidden layers
    for layer in config["layers"][1:-1]:
        layer["in_channels"] = layer["out_channels"] = channels

    return config


def construct_layer(dropout_p=0, bnorm=True, gelu=False, *args, **kwargs):
    """Constructs a configurable Convolutional block with Batch Normalization and Dropout.

    Args:
    dropout_p (float): Dropout probability. Default is 0.
    bnorm (bool): Whether to include batch normalization. Default is True.
    gelu (bool): Whether to use GELU activation. Default is False.
    *args: Additional positional arguments to pass to nn.Conv3d.
    **kwargs: Additional keyword arguments to pass to nn.Conv3d.

    Returns:
    nn.Sequential: A sequential container of Convolutional block with optional Batch Normalization and Dropout.
    """
    layers = []
    layers.append(nn.Conv3d(*args, **kwargs))
    if bnorm:
        # track_running_stats=False is needed to run the forward mode AD
        layers.append(
            nn.BatchNorm3d(kwargs["out_channels"], track_running_stats=True)
        )
    layers.append(nn.ELU(inplace=True) if gelu else nn.ReLU(inplace=True))
    if dropout_p > 0:
        layers.append(nn.Dropout3d(dropout_p))
    return nn.Sequential(*layers)


def init_weights(model):
    """Set weights to be xavier normal for all Convs"""
    for m in model.modules():
        if isinstance(
            m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)
        ):
            # nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain("relu"))
            nn.init.kaiming_normal_(
                m.weight, mode="fan_out", nonlinearity="relu"
            )
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)


class SequentialConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SequentialConvLayer, self).__init__()
        self.convs = nn.ModuleList(
            [nn.Conv3d(in_channels, 1, 1) for _ in range(out_channels)]
        )

    def forward(self, x):
        # Size of the input tensor
        batch_size, _, depth, height, width = x.size()

        # Initialize the output cubes
        outB = -10000 * torch.ones(batch_size, 1, depth, height, width).to(
            x.device
        )
        outC = torch.zeros(batch_size, 1, depth, height, width).to(x.device)

        for i, conv in enumerate(self.convs):
            # Apply the current filter
            outA = conv(x)

            # Find where the new filter gives a greater response than the max so far
            greater = outA > outB
            greater = greater.float()

            # Update outB with the max values so far
            outB = (1 - greater) * outB + greater * outA

            # Update outC with the index of the filter with the max response so far
            outC = (1 - greater) * outC + greater * i

        return outC


class MeshNet(nn.Module):
    """Configurable MeshNet from https://arxiv.org/pdf/1612.00940.pdf"""

    def __init__(self, in_channels, n_classes, channels, config_file, fat=None):
        """Init"""
        with open(config_file, "r") as f:
            config = set_channel_num(
                json.load(f), in_channels, n_classes, channels
            )

        if fat is not None:
            chn = int(channels * 1.5)
            if fat in {"i", "io"}:
                config["layers"][0]["out_channels"] = chn
                config["layers"][1]["in_channels"] = chn
            if fat == "io":
                config["layers"][-1]["in_channels"] = chn
                config["layers"][-2]["out_channels"] = chn
            if fat == "b":
                config["layers"][3]["out_channels"] = chn
                config["layers"][4]["in_channels"] = chn

        super(MeshNet, self).__init__()

        layers = [
            construct_layer(
                dropout_p=config["dropout_p"],
                bnorm=config["bnorm"],
                gelu=config["gelu"],
                **block_kwargs,
            )
            for block_kwargs in config["layers"]
        ]
        # layers[-1] = SequentialConvLayer(layers[-1][0].in_channels, layers[-1][0].out_channels)
        layers[-1] = layers[-1][0]
        self.model = nn.Sequential(*layers)
        init_weights(self.model)

    def forward(self, x):
        """Forward pass"""
        x = self.model(x)
        return x


class enMesh_checkpoint(MeshNet):
    def train_forward(self, x):
        y = x
        y.requires_grad_()
        y = checkpoint_sequential(
            self.model, len(self.model), y, preserve_rng_state=False
        )
        return y

    def eval_forward(self, x):
        """Forward pass"""
        self.model.eval()
        with torch.inference_mode():
            x = self.model(x)
        return x

    def forward(self, x):
        if self.training:
            return self.train_forward(x)
        else:
            return self.eval_forward(x)

class enMesh(MeshNet):
    def __init__(
        self,
        in_channels,
        n_classes,
        channels,
        config_file,
        optimize_inline=False,
    ):
        super(enMesh, self).__init__(
            in_channels, n_classes, channels, config_file
        )
        self.n_classes = n_classes
        self.optimize_inline = optimize_inline
        if self.optimize_inline:
            self.optimizers = [
                torch.optim.Adam(net.parameters(), lr=0.02)
                for net in self.model
            ]

    def get_grads(self, grads):
        def show(self, grad_input, grad_output):
            grads["in"] = grad_input
            grads["out"] = grad_output

        return show

    def set_requires_grad_layer(self, layer, flag, trainBN=True):
        layer.train(flag)
        for x in layer.parameters():
            if not flag:
                del x.grad
                x.detach()
            x.grad = [None, x.grad][flag]
            x.requires_grad = flag
        if (
            trainBN
            and isinstance(layer, torch.nn.Sequential)
            and isinstance(layer[1], torch.nn.BatchNorm3d)
        ):
            layer[1].training = True
            layer[1].requires_grad = True

    def unset_grad(self, layer):
        self.set_requires_grad_layer(layer, False)

    def set_grad(self, layer):
        self.set_requires_grad_layer(layer, True)

    def dump_tensors(gpu_only=True):
        # torch.cuda.empty_cache()
        total_size = 0
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    if not gpu_only or obj.is_cuda:
                        del obj
                        gc.collect()
                elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                    if not gpu_only or obj.is_cuda:
                        del obj
                        gc.collect()
            except Exception as e:
                pass

    def eval_forward(self, x):
        """Forward pass"""
        with torch.inference_mode():
            for i, layer in enumerate(self.model):
                x = layer(x)
        return x

    def forward(self, x, y=None, loss=None, verbose=False):
        if self.training:
            return self.backforward(x, y, loss, verbose=verbose)
        else:
            return self.eval_forward(x)

    def backforward(self, x, y, loss, verbose=False):
        if verbose:
            h = nvmlDeviceGetHandleByIndex(0)
            info = nvmlDeviceGetMemoryInfo(h)
            print(f"total    : {info.total}")
            print(f"free     : {info.free}")
            print(f"used     : {info.used}")
            print(f"used fr  : {info.used/info.total}")

        gradients = {}
        layers = [p for p in self.model]
        for p in layers:
            self.unset_grad(p)

        grads = {}
        handle = layers[-1].register_full_backward_hook(self.get_grads(grads))

        self.set_grad(layers[-1])
        input = x
        input.requires_grad = False
        for count, layer in enumerate(layers):
            input = layer(input)
        y_hat = input
        input.requires_grad_()
        input.detach()

        if verbose:
            info = nvmlDeviceGetMemoryInfo(h)
            print(f"used fr  : {info.used/info.total}")

        if isinstance(loss, torch.nn.CrossEntropyLoss):
            output = loss(input, y)
        else:
            one_hot_targets = torch.nn.functional.one_hot(
                y, self.n_classes
            ).permute(0, 4, 1, 2, 3)
            logits_softmax = F.softmax(input, dim=1)
            output = loss(logits_softmax, one_hot_targets)
        output.backward()
        output.detach()
        lss_value = output
        del output
        del input
        self.unset_grad(layers[-1])
        handle.remove()

        dloss_dx2 = grads["out"][0]

        del grads["in"]

        if verbose:
            info = nvmlDeviceGetMemoryInfo(h)
            print(f"used fr  : {info.used/info.total}")
            print("*" * 20)

        # unembedded = True
        for i in range(len(layers) - 1, -1, -1):
            input = x.detach().clone()
            input.requires_grad = False
            grads = {}
            handle = layers[i].register_full_backward_hook(
                self.get_grads(grads)
            )
            self.set_grad(layers[i])

            # Recompute the forward pass up to the current layer
            for j in range(0, i + 1):
                if j == i:
                    input.detach()
                    input.requires_grad_()
                input = layers[j](input)

            input.detach()
            torch.autograd.backward(input, dloss_dx2)

            del dloss_dx2
            dloss_dx2 = grads["in"][0]

            if self.optimize_inline:
                self.optimizers[i].step()
                self.optimizers[i].zero_grad(set_to_none=True)
            else:
                gradients[i] = [x.grad for x in layers[i].parameters()]

            self.unset_grad(layers[i])
            handle.remove()
            del input.grad
            del x.grad
            del input
            x.requires_grad = False
        del dloss_dx2
        self.model.eval()
        if not self.optimize_inline:
            for i in range(len(layers)):
                # self.set_grad(layers[i])
                for p, g in zip(layers[i].parameters(), gradients[i]):
                    p.grad = g
        del layers
        if verbose:
            info = nvmlDeviceGetMemoryInfo(h)
            print(f"{i} used fr  : {info.used/info.total}")
        # torch.cuda.empty_cache()
        # self.dump_tensors()

        return lss_value, y_hat

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    channels = 15
    cubesize = 256
    classes = 3
    batch = 1
    config_file = "modelAE.json"

    emodel = enMesh_checkpoint(1, classes, channels, config_file).to(device)

    x = torch.rand(batch, 1, *(cubesize,) * 3, requires_grad=False).to(device)
    y = torch.randint(
        0, classes, (batch, *(cubesize,) * 3), requires_grad=False
    ).to(device)

    print("enmesh")
    from blendbatchnorm import fuse_bn_recursively

    emodel = fuse_bn_recursively(emodel)
    criterion = torch.nn.CrossEntropyLoss()
    emodel.train(False)
    t0 = time.time()
    with torch.inference_mode():
        for i in range(10):
            r = emodel.forward(x)
            del r
            torch.cuda.empty_cache()
    t1 = time.time()
    print(t1 - t0)
