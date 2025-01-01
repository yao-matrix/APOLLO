import pdb
import math
import time
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def _quantize_tensor(w, q_group_size=-1, n_bit=8):

    org_w_shape = w.shape
    if q_group_size > 0:
        assert w.nelement() % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2

    max_val = w.amax(dim=1, keepdim=True)
    min_val = w.amin(dim=1, keepdim=True)
    max_int = 2**n_bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    w = torch.clamp(torch.round(w / scales) + zeros, min_int, max_int)
    w = w.reshape(org_w_shape).to(torch.uint8)

    return w, scales, zeros


class QLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                device='cpu', dtype=None, weight_data=None, bias_data=None, num_bits=8, group_size=256, stochastic_round=True) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(in_features, out_features, bias)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features)))

        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.weight.__setattr__('stochastic_round', stochastic_round)
        self.weight.__setattr__('group_size', group_size)

        if weight_data is not None:
            self.weight.data.copy_(weight_data)
        if bias_data is not None and bias is not None:
            self.bias.data.copy_(bias_data)

        self.num_bits = num_bits
        self.group_size = group_size

    def forward(self, input: Tensor) -> Tensor:
        qweight, scales, zeros = _quantize_tensor(self.weight, q_group_size=self.group_size, n_bit=self.num_bits)
        # dequantize to Bfloat16
        qweight = qweight.to(input.dtype).reshape(-1, self.group_size)
        qweight = (qweight - zeros) * scales
        qweight = qweight.reshape(self.weight.shape)
        # STE backward
        qweight = qweight.detach() + self.weight - self.weight.detach()

        output = input @ qweight.t()
        if self.bias is not None:
            output += self.bias

        return output


def prepare_model_for_int8_training_simulation(model, args, target_module):

    for name, module in reversed(model._modules.items()):

        if len(list(module.children())) > 0:
            model._modules[name] = prepare_model_for_int8_training_simulation(module, args, target_module)

        if isinstance(module, nn.Linear):
            if not name in target_module:
                print('Keep in original linear layer', name, module)
                continue

            bias_data = module.bias.data if module.bias is not None else None
            in_features = module.in_features
            out_features = module.out_features
            bias = module.bias is not None
            weight_data = module.weight.data
            new_layers = QLinear(in_features, out_features, bias=bias, device='cuda:0', 
                weight_data=weight_data, bias_data=bias_data, 
                num_bits=args.weight_bits, group_size=args.weight_group_size, stochastic_round=args.stochastic_round)

            model._modules[name] = new_layers

    return model

if __name__ == '__main__':
    GROUP_SIZE=256
    fp16_linear1 = nn.Linear(4096, 4096, bias=False).to('cuda:0').to(torch.bfloat16)
    print('after initial weight for bfloat16', '{:.2f} MB'.format(torch.cuda.memory_allocated('cuda:0')//1024/1024))
    mem_weight_float = torch.cuda.memory_allocated('cuda:0')//1024/1024
    x = torch.randn(1, 256, 4096, dtype=torch.bfloat16, device='cuda:0', requires_grad=True)
    print('after initial input for bfloat16', '{:.2f} MB'.format(torch.cuda.memory_allocated('cuda:0')//1024/1024))
    start = time.time()
    output = fp16_linear1(x)
    print('after forward for bfloat16', '{:.2f} MB'.format(torch.cuda.memory_allocated('cuda:0')//1024/1024))
    output.sum().backward()
    print('output_full', output)
    end = time.time()
    print('after backward for bfloat16', '{:.2f} MB'.format(torch.cuda.memory_allocated('cuda:0')//1024/1024))
    print('Time for FW+BW = {:.2f} s'.format(end-start))
    print('Gradient for weight:', fp16_linear1.weight.grad)
    print('------------------------------------')

    from quantization import QScaleLinear
    int8_linear1 = QScaleLinear(fp16_linear1.weight, None, device='cuda:1', num_bits=8, group_size=GROUP_SIZE)
    print('after initial weight for int8', '{:.2f} MB'.format(torch.cuda.memory_allocated('cuda:1')//1024/1024))
    mem_weight_int = torch.cuda.memory_allocated('cuda:1')//1024/1024
    x1 = x.to('cuda:1')
    print('after initial input for bfloat16', '{:.2f} MB'.format(torch.cuda.memory_allocated('cuda:1')//1024/1024))
    start = time.time()
    output_int8 = int8_linear1(x1)
    print('after forward for int8', '{:.2f} MB'.format(torch.cuda.memory_allocated('cuda:1')//1024/1024))
    output_int8.sum().backward()
    print('output_quant_real', output_int8)
    end = time.time()
    print('after backward for int8', '{:.2f} MB'.format(torch.cuda.memory_allocated('cuda:1')//1024/1024))
    print('Time for FW+BW = {:.2f} s'.format(end-start))
    print('Gradient for weight:', int8_linear1.weight.float_grad)
    print('------------------------------------')

    print('Memory saving for weight: {:.2f} MB, ratio: {:.2f}%'.format(mem_weight_float - mem_weight_int, mem_weight_int / mem_weight_float * 100))
    print('------------------------------------')

    int8_simluate_linear1 = QLinear(4096, 4096, device='cuda:2', bias=False, num_bits=8, group_size=GROUP_SIZE, weight_data=fp16_linear1.weight.data, bias_data=None).to(torch.bfloat16)
    
    print('after initial weight for int8', '{:.2f} MB'.format(torch.cuda.memory_allocated('cuda:2')//1024/1024))
    mem_weight_int = torch.cuda.memory_allocated('cuda:2')//1024/1024
    x2 = x.to('cuda:2')
    print('after initial input for bfloat16', '{:.2f} MB'.format(torch.cuda.memory_allocated('cuda:2')//1024/1024))
    start = time.time()
    output_int8_simulate = int8_simluate_linear1(x2)
    print('after forward for int8', '{:.2f} MB'.format(torch.cuda.memory_allocated('cuda:2')//1024/1024))
    output_int8_simulate.sum().backward()
    print('output_quant_simulation', output_int8_simulate)
    end = time.time()
    print('after backward for int8', '{:.2f} MB'.format(torch.cuda.memory_allocated('cuda:2')//1024/1024))
    print('Time for FW+BW = {:.2f} s'.format(end-start))
    print('Gradient for weight:', int8_simluate_linear1.weight.grad)
    print('------------------------------------')












