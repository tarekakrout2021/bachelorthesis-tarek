# following this implementation : https://github.com/Mrw33554432/Bitlinear4HF/blob/master/bitlinear.py
import torch.nn.functional as F
from torch import nn, torch

# import optimized_bitlinear as obl # TODO: After compiling the cuda kernel import the function here


def weight_quant_training(weight):
    scale = 1 / weight.abs().mean().clamp(min=1e-5)
    result = (weight * scale).round().clamp(-1, 1) / scale
    return result


def activation_quant_training(x, num_bits=8):
    Qn = -(2 ** (num_bits - 1))
    Qp = 2 ** (num_bits - 1) - 1
    scale = Qp / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    result = (x * scale).round().clamp(Qn, Qp) / scale
    return result


def weight_quant_inference(weight):
    beta = 1 / weight.abs().mean().clamp(min=1e-5)
    result = (weight * beta).round().clamp(-1, 1).to(torch.int8)

    return result, beta


def activation_quant_inference(x, num_bits=8):
    Qn = -(2 ** (num_bits - 1))
    Qp = 2 ** (num_bits - 1) - 1
    s = Qp / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    result = (x * s).round().clamp(Qn, Qp).to(torch.int8)
    return result, s


class BitLinear158(nn.Linear):
    def __init__(self, input_dim, output_dim, weight_bits=1, input_bits=8):
        super(BitLinear158, self).__init__(input_dim, output_dim)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_bits = weight_bits
        self.input_bits = input_bits
        self.beta = 1

    def forward(self, input):
        device = input.device

        w_quant = weight_quant_training(self.weight.to(device))
        quant_input = activation_quant_training(input, self.input_bits)

        # This is used for straight through estimator (STE)
        quant_input = input + (quant_input - input).detach()
        quant_weight = (
            self.weight.to(device) + (w_quant - self.weight.to(device)).detach()
        )

        out = F.linear(quant_input, quant_weight)
        if self.bias is not None:
            out += self.bias.to(device).view(1, -1).expand_as(out)

        return out


class BitLinear158Inference(nn.Linear):
    """
    Weights should be quantized once after training
    """

    def __init__(self, input_dim, output_dim, weight_bits=1, input_bits=8):
        super(BitLinear158Inference, self).__init__(input_dim, output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_bits = weight_bits
        self.input_bits = input_bits
        self.beta = 1

    def forward(self, input):
        device = input.device

        quant_input, gamma = activation_quant_inference(input, self.input_bits)

        quant_input = input + (quant_input - input).detach()

        # out = obl.mat_mul(quant_input, self.weight) / self.beta / gamma  # TODO:  should be like this after using the kernel
        out = F.linear(
            quant_input.to(device) / self.beta / gamma.to(device),
            self.weight.to(device),
        )
        if self.bias is not None:
            out += self.bias.to(device).view(1, -1).expand_as(out)

        return out
