from collections import abc
from inspect import getfullargspec

import numpy as np
import torch


# copy updated param from fp32_weight to fp16 net
def copy_in_params(fp16_net, fp32_weight):
    for net_param, fp32_weight_param in zip(fp16_net.parameters(),
                                            fp32_weight):
        net_param.data.copy_(fp32_weight_param.data)


# copy gradient from fp16 net to fp32 weight copy
def set_grad(fp16_net, fp32_weight):
    for param, param_w_grad in zip(fp32_weight, fp16_net.parameters()):
        if param_w_grad.grad is not None:
            if param.grad is None:
                param.grad = param.data.new(*param.data.size())
            param.grad.data.copy_(param_w_grad.grad.data)


def convert(inputs, src_type, dst_type, min_dim=0):
    if isinstance(inputs, torch.Tensor):
        # some tensor don't need to convert to fp16, e.g. gt_bboxes
        # these tensors' dim are usually smaller or equal to 2
        if inputs.dim() > min_dim and inputs.dtype == src_type:
            inputs = inputs.to(dst_type)
        return inputs
    if isinstance(inputs, str):
        return inputs
    if isinstance(inputs, np.ndarray):
        return inputs
    elif isinstance(inputs, abc.Mapping):
        return type(inputs)({
            k: convert(v, src_type, dst_type, min_dim)
            for k, v in inputs.items()
        })
    elif isinstance(inputs, abc.Iterable):
        return type(inputs)(
            convert(item, src_type, dst_type, min_dim) for item in inputs)
    else:
        return inputs


def patch_forward_module(old_forward, src_type, dst_type, convert_output):
    # conver input to fp32
    # convert output to fp16

    def new_forward(*args, **kwargs):
        output = old_forward(*convert(args, src_type, dst_type, min_dim=0),
                             **convert(kwargs, dst_type, src_type, min_dim=0))
        if convert_output:
            output = convert(output, dst_type, src_type, min_dim=0)
        return output

    return new_forward


def bn_convert_float(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
        module.forward = patch_forward_module(
            module.forward, torch.half, torch.float, convert_output=True)
    for child in module.children():
        bn_convert_float(child)
    return module


def wrap_fp16_model(model, convert_bn):
    # convert model to fp16
    model.half()
    if convert_bn:
        bn_convert_float(model)  # bn should be in fp32


def auto_fp16(apply_to=None, out_fp32=False):

    def auto_fp16_wrapper(old_func):

        def new_func(*args, **kwargs):
            args_info = getfullargspec(old_func)
            num_args = len(args)
            num_kwargs = len(kwargs)
            arg_names = args_info.args[:num_args]
            # convert args
            if num_args > 0:
                new_args = []
                for i, arg in enumerate(arg_names):
                    if arg in apply_to:
                        new_args.append(
                            convert(args[i], torch.float, torch.half))
                    else:
                        new_args.append(args[i])
            else:
                new_args = args
            # convert kwargs
            if num_kwargs > 0:
                new_kwargs = dict()
                for k, v in kwargs.items():
                    if k in apply_to:
                        new_kwargs[k] = convert(v, torch.float, torch.half)
                    else:
                        new_kwargs[k] = v
            else:
                new_kwargs = kwargs
            output = old_func(*new_args, **new_kwargs)
            if out_fp32:
                output = convert(output, torch.half, torch.float)
            return output

        return new_func

    return auto_fp16_wrapper


def force_fp32(apply_to=None, out_fp16=False):

    def force_fp32_wrapper(old_func):

        def new_func(*args, **kwargs):
            args_info = getfullargspec(old_func)
            num_args = len(args)
            num_kwargs = len(kwargs)
            arg_names = args_info.args[:num_args]
            # convert args
            if num_args > 0:
                new_args = []
                for i, arg in enumerate(arg_names):
                    if arg in apply_to:
                        new_args.append(
                            convert(args[i], torch.half, torch.float))
                    else:
                        new_args.append(args[i])
            else:
                new_args = args
            # convert kwargs
            if num_kwargs > 0:
                new_kwargs = dict()
                for k, v in kwargs.items():
                    if k in apply_to:
                        new_kwargs[k] = convert(v, torch.half, torch.float)
                    else:
                        new_kwargs[k] = v
            else:
                new_kwargs = kwargs
            output = old_func(*new_args, **new_kwargs)
            if out_fp16:
                output = convert(output, torch.float, torch.half)
            return output

        return new_func

    return force_fp32_wrapper