from torch.autograd import Function
import torch
import adaptive_conv # The compiled module


torch.manual_seed(42)

class AdaptiveConv(Function):

    @staticmethod
    def forward(ctx, input, filters):
        ctx.save_for_backward(filters, input)
        b, h2, w2, f1, f2 = filters.shape
        assert f1 == f2
        result = adaptive_conv.forward(input, filters)
        return result


    @staticmethod
    def backward(ctx, grad_output):
        filters, input = ctx.saved_tensors
        grad_input = grad_filters = None
        b, h2, w2, f1, f2 = filters.shape
        assert f1 == f2

        grad_output = grad_output.contiguous()
        if ctx.needs_input_grad[0]:
            grad_input = adaptive_conv.grad_input(grad_output, filters)
        if ctx.needs_input_grad[1]:
            grad_filters = adaptive_conv.grad_filters(grad_output, input)


        return grad_input, grad_filters