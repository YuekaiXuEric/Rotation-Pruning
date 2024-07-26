import torch
import torch.nn as nn
from .utils import DEV
from .data_utils import get_loaders
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from tqdm import tqdm
from .qLlamaLayer import QLlamaDecoderLayer
import functools, math

@torch.no_grad()
def get_reorder_index(model, act_scales):
    act_orders = {}

    def reorder_tensor(tensor: torch.Tensor) -> torch.tensor:
        # assert dimension == 1
        assert tensor.dim() == 1, "Choosing outliers must be 1 dimensional"
        _, sorted_index = torch.sort(tensor, descending=False) # For putting outliers at last
        # _, sorted_index = torch.sort(tensor, descending=True) # For putting outliers at first

        return sorted_index

    def reorder_tensor_heads(tensor: torch.Tensor) -> torch.tensor:
        assert tensor.dim() == 1, "Choosing outliers must be 1 dimensional after mean."
        assert tensor.shape[0] % 128 == 0, "Hidden dimension must be divisible by 128." # Assume head_dim == 128

        num_heads = tensor.shape[0] // 128
        index_slices = []
        for i in range(num_heads):
            startIdx = i * 128
            endIdx = (i + 1) * 128
            _, unitIndices = torch.sort(tensor[startIdx:endIdx], descending=True)
            index_slices.append(unitIndices + startIdx)

        return torch.cat(index_slices).contiguous()

    for name, m in model.model.named_modules():
        if isinstance(m, nn.Linear):
            m.name = name

            # Reorder Index of each layer's input
            # Used to reorder the weight and previous layer's output
            inputName = name + ".input"
            act_orders[inputName] = reorder_tensor(act_scales[inputName])
            assert act_orders[inputName].dim() == 1, "Return Index must be 1 dimensional"

            # Reorder Index of Q,K,V's output (Self-attn's input)
            # Used to determine each head's reorder index
            # Assume head_dim == 128
            outputName = name + ".output"
            act_orders[outputName] = reorder_tensor_heads(act_scales[outputName])
            assert act_orders[outputName].dim() == 1, "Return Index must be 1 dimensional"

    return act_orders



def permute(model, args, tokenizer, device=DEV, ):
    dataloader = get_loaders("wikitext2",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen, model='meta-llama/Llama-2-7b-hf')

    print("Getting activation stats...")
    act_scales = get_act_stats_llama(model, dataloader, DEV, metric=args.act_sort_metric)

    print("Getting reording index...")
    reorder_index = get_reorder_index(model, act_scales)

    print("Reordering model...")
    model = reorder_model_llama(model, device=DEV, args=args, reorder_index=reorder_index)

def reorder_model_llama(model, device, args, reorder_index):
    model.config.use_cache = False
    layers = model.model.layers
    assert reorder_index is not None, "Reorder index is None"


    for i in tqdm(range(len(layers))):
        layers[i] = layers[i].to(device)
        if isinstance(layers[i], LlamaDecoderLayer):
            m = QLlamaDecoderLayer(
                originalLayer=layers[i],
                args=args,
            )
        elif isinstance(layers[i], QLlamaDecoderLayer):
            m = layers[i]

        nameTemplate = 'layers.{}.{}.{}.{}' # Something like layers.10.self_attn.q_proj

        m.mlp.gate_proj.reorder(
            in_reorder_index=reorder_index[nameTemplate.format(i, 'mlp', 'gate_proj', 'input')],
            out_reorder_index=reorder_index[nameTemplate.format(i, 'mlp', 'down_proj', 'input')]
        )
        m.mlp.up_proj.reorder(
            in_reorder_index=reorder_index[nameTemplate.format(i, 'mlp', 'up_proj', 'input')],
            out_reorder_index=reorder_index[nameTemplate.format(i, 'mlp', 'down_proj', 'input')]
        )
        m.mlp.down_proj.reorder(
            in_reorder_index=reorder_index[nameTemplate.format(i, 'mlp', 'down_proj', 'input')],
            out_reorder_index=None
        )
        # K has outlier should be kept.
        # Not reorder due to the RoPE embedding.
        m.self_attn.q_proj.reorder(
            in_reorder_index=reorder_index[nameTemplate.format(i, 'self_attn', 'q_proj', 'input')],
            out_reorder_index=None
        )
        m.self_attn.k_proj.reorder(
            in_reorder_index=reorder_index[nameTemplate.format(i, 'self_attn', 'k_proj', 'input')],
            out_reorder_index=None
        )
        m.self_attn.v_proj.reorder(
            in_reorder_index=reorder_index[nameTemplate.format(i, 'self_attn', 'v_proj', 'input')],
            out_reorder_index=None
        )
        m.self_attn.o_proj.reorder(
            in_reorder_index=reorder_index[nameTemplate.format(i, 'self_attn', 'o_proj', 'input')],
            out_reorder_index=None
        )
        m.input_layernorm.register_buffer('reorder_index',
            reorder_index[nameTemplate.format(i, 'self_attn', 'k_proj', 'input')] # Random choose one from k,q,v proj.
        )
        m.post_attention_layernorm.register_buffer('reorder_index',
            reorder_index[nameTemplate.format(i, 'mlp', 'gate_proj', 'input')]
        )
        m.self_attn.register_buffer('reorder_index', reorder_index[nameTemplate.format(i, 'self_attn', 'o_proj', 'input')])

        layers[i] = layers[i].cpu()
        layers[i] = m.cpu()
        del m
        torch.cuda.empty_cache()
    return model

@torch.no_grad()
def get_act_stats_llama(model, dataloader, device_, metric='hessian'):
    nsamples = len(dataloader)
    device = device_
    act_scales = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).detach()

        if metric == 'hessian':
            tensorH = math.sqrt(2 / nsamples) * tensor.float().t()
            comming_H = tensorH.matmul(tensorH.t())
            comming_scales = torch.diag(comming_H)
        else:
            # Here we use abs since symmetric quantization use absmax.
            comming_scales = torch.mean(tensor.abs(), dim=0).float().cpu()

        if name in act_scales:
            if metric == 'hessian':
                act_scales[name] += comming_scales
            else:
                act_scales[name] = torch.max(act_scales[name], comming_scales)
        else:
            act_scales[name] = comming_scales

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
            assert isinstance(x, torch.Tensor)
        if isinstance(y, tuple):
            y = y[0]
            assert isinstance(y, torch.Tensor)
        stat_tensor(name + ".input", x)
        stat_tensor(name + ".output", y)

    hooks = []
    for name, m in model.model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)
                )
            )

    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    if not model.model.norm.weight.is_meta:
        model.model.norm = model.model.norm.to(device)
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.self_attn = module.self_attn
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.squeeze(0)
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    assert cache['i'] == nsamples, "Captured samples should be equal to nsamples"

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if not model.model.norm.weight.is_meta:
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    for i in tqdm(range(len(layers))):
        layer = layers[i].to(device)
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        inps, outs = outs, inps

    for h in hooks:
        h.remove()

    return act_scales
