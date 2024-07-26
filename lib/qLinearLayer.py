import torch
import torch.nn as nn
from .utils import DEV

def find_qlinear_layers(module, name=''):
    if type(module) == QLinearLayer:
        if module.enable_quant:
            return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_qlinear_layers(
            child, name=name + '.' + name1 if name != '' else name1
        ))
    return res

class QLinearLayer(nn.Module):
    def __init__(
        self,
        originalLayer: nn.Linear,
        args,
        enable_quant: bool = True
    ):
        super().__init__()
        self.args = args
        self.weight = nn.Parameter(originalLayer.weight.clone())
        self.enable_quant = enable_quant # whether to allow quant on weights, default True
        if originalLayer.bias is not None:
            self.bias = nn.Parameter(originalLayer.bias.clone())
        else:
            self.bias = None

    @torch.no_grad()
    def forward(self, x):
        # Ensure weight and bias are on the same device as input x
        weight = self.weight.to(x.device)
        bias = self.bias.to(x.device) if self.bias is not None else None
        y = torch.functional.F.linear(x, weight, bias)
        return y

    def to(self, *args, **kwargs):
        super(QLinearLayer, self).to(*args, **kwargs)
        self.weight = nn.Parameter(self.weight.to(*args, **kwargs))
        if self.bias is not None:
            self.bias = nn.Parameter(self.bias.to(*args, **kwargs))
        return self

    def reorder(self, in_reorder_index, out_reorder_index=None):
        if self.args.reorder == True:
            in_reorder_index = in_reorder_index.to(self.weight.device)
            self.weight = nn.Parameter(torch.index_select(self.weight, 1, in_reorder_index))
            if out_reorder_index is not None:
                out_reorder_index = out_reorder_index.to(self.weight.device)
                self.weight = nn.Parameter(torch.index_select(self.weight, 0, out_reorder_index))
        return
