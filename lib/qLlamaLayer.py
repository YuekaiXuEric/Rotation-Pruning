import torch
from torch import nn
from typing import List, Optional, Tuple
import math
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaAttention, LlamaMLP
from .quant import Quantizer
from .qLinearLayer import QLinearLayer
from .utils import DEV

torch.autograd.set_detect_anomaly(True)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    print(f"rotate_half: x device: {x.device}")
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    print(f"apply_rotary_pos_emb: q device: {q.device}, k device: {k.device}, cos device: {cos.device}, sin device: {sin.device}")
    cos = cos.unsqueeze(unsqueeze_dim).to(DEV)
    sin = sin.unsqueeze(unsqueeze_dim).to(DEV)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep)."""
    print(f"repeat_kv: hidden_states device: {hidden_states.device}")
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states.to(DEV)
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class QLlamaDecoderLayer(nn.Module):
    def __init__(self, originalLayer: LlamaDecoderLayer, args):
        super().__init__()
        self.args = args
        self.hidden_size = originalLayer.hidden_size
        self.self_attn = QLlamaAttention(originalLayer.self_attn, args)
        self.mlp = QLlamaMLP(originalLayer.mlp, args)
        self.input_layernorm = QLlamaRMSNorm(originalLayer.input_layernorm, args)
        self.post_attention_layernorm = QLlamaRMSNorm(originalLayer.post_attention_layernorm, args)

    def to(self, *args, **kwargs):
        super(QLlamaDecoderLayer, self).to(*args, **kwargs).to(DEV)
        self.self_attn = self.self_attn.to(*args, **kwargs)
        self.input_layernorm = self.input_layernorm.to(*args, **kwargs)
        self.post_attention_layernorm = self.post_attention_layernorm.to(*args, **kwargs)
        self.mlp = self.mlp.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        padding_mask=None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        hidden_states = hidden_states.to(DEV)
        print(f"QLlamaDecoderLayer: hidden_states device: {hidden_states.device}")
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class QLlamaRMSNorm(nn.Module):
    def __init__(self, originalNorm: LlamaRMSNorm, args):
        super().__init__()
        self.originalNorm = originalNorm
        self.act_quant = Quantizer(args=args)
        self.register_buffer("reorder_index", None)
        self.args = args

    @torch.no_grad()
    def forward(self, hidden_states):
        hidden_states = hidden_states.to(DEV)
        print(f"QLlamaRMSNorm: hidden_states device: {hidden_states.device}")
        result = self.originalNorm(hidden_states)

        # Ensure reorder_index is on the same device as result
        if self.reorder_index is not None:
            self.reorder_index = self.reorder_index.to(DEV)
            assert result.shape[result.dim() - 1] == self.reorder_index.shape[0]
            result = torch.index_select(result, result.dim() - 1, self.reorder_index)

        if self.args.abits < 16:
            result = self.act_quant(result)

        return result

    def to(self, *args, **kwargs):
        super(QLlamaRMSNorm, self).to(*args, **kwargs)
        self.originalNorm = self.originalNorm.to(*args, **kwargs)
        self.act_quant = self.act_quant.to(*args, **kwargs)
        if self.reorder_index is not None:
            self.reorder_index = self.reorder_index.to(*args, **kwargs)
        return self

class QLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, originalAttn: LlamaAttention, args):
        super().__init__()
        self.abits = args.abits
        self.q_kv_cache = args.kv_cache
        self.config = originalAttn.config
        self.hidden_size = originalAttn.hidden_size
        self.num_heads = originalAttn.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = originalAttn.num_key_value_heads
        self.num_key_value_groups = originalAttn.num_key_value_groups
        self.max_position_embeddings = originalAttn.max_position_embeddings
        self.rope_theta = originalAttn.rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = QLinearLayer(originalAttn.q_proj, args).to(DEV)
        self.k_proj = QLinearLayer(originalAttn.k_proj, args).to(DEV)
        self.v_proj = QLinearLayer(originalAttn.v_proj, args).to(DEV)
        self.o_proj = QLinearLayer(originalAttn.o_proj, args).to(DEV)
        self.rotary_emb = originalAttn.rotary_emb.to(DEV)
        self.act_quant = Quantizer(args=args).to(DEV)
        self.v_quant = Quantizer(args=args).to(DEV)
        self.k_quant = Quantizer(args=args).to(DEV)
        self.register_buffer("reorder_index", None)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def to(self, *args, **kwargs):
        super(QLlamaAttention, self).to(*args, **kwargs).to(DEV)
        self.q_proj = self.q_proj.to(*args, **kwargs).to(DEV)
        self.k_proj = self.k_proj.to(*args, **kwargs).to(DEV)
        self.v_proj = self.v_proj.to(*args, **kwargs).to(DEV)
        self.o_proj = self.o_proj.to(*args, **kwargs).to(DEV)
        self.rotary_emb = self.rotary_emb.to(*args, **kwargs).to(DEV)
        self.act_quant = self.act_quant.to(*args, **kwargs).to(DEV)
        self.v_quant = self.v_quant.to(*args, **kwargs).to(DEV)
        self.k_quant = self.k_quant.to(*args, **kwargs).to(DEV)
        if self.reorder_index is not None:
            self.reorder_index = self.reorder_index.to(*args, **kwargs).to(DEV)
        return self

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        hidden_states = hidden_states.to(DEV)
        print(f"QLlamaAttention: hidden_states device: {hidden_states.device}")
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2).to(DEV)
        print(f"QLlamaAttention: query_states device: {query_states.device}")
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2).to(DEV)
        print(f"QLlamaAttention: key_states device: {key_states.device}")
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2).to(DEV)
        print(f"QLlamaAttention: value_states device: {value_states.device}")

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        # Ensure position_ids is on the same device as value_states
        if position_ids is not None:
            position_ids = position_ids.to(DEV)
        print(f"QLlamaAttention: position_ids device: {position_ids.device}")

        # Fake quantize the key_states.
        # Preserve the position embedding info by first quantize.
        if self.q_kv_cache:
            key_states = self.k_quant(key_states).to(DEV)
        print(f"QLlamaAttention: key_states after quant device: {key_states.device}")

        value_states = value_states.to('cpu')
        position_ids = position_ids.to('cpu')
        print(f"QLlamaAttention: value_states after quant device: {value_states.device}")
        print(f"QLlamaAttention: position_ids after quant device: {position_ids.device}")
        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2).to(DEV)
            value_states = torch.cat([past_key_value[1], value_states], dim=2).to(DEV)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups).to(DEV)
        value_states = repeat_kv(value_states, self.num_key_value_groups).to(DEV)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)).to(DEV) / math.sqrt(self.head_dim)
        print(f"QLlamaAttention: attn_weights device: {attn_weights.device}")

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attention_mask = attention_mask.to(DEV)
            print(f"QLlamaAttention: attention_mask device: {attention_mask.device}")
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype).to(DEV)
        print(f"QLlamaAttention: attn_weights after softmax device: {attn_weights.device}")

        # Fake quantize the value_states
        if self.q_kv_cache:
            value_states = self.v_quant(value_states).to(DEV)
        print(f"QLlamaAttention: value_states after quant device: {value_states.device}")

        attn_output = torch.matmul(attn_weights, value_states).to(DEV)
        print(f"QLlamaAttention: attn_output device: {attn_output.device}")

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).to(DEV)
        print(f"QLlamaAttention: attn_output reshaped device: {attn_output.device}")

        # Reorder the BMM output to feed into o.proj
        if self.reorder_index is not None:
            self.reorder_index = self.reorder_index.to(DEV)
            attn_output = torch.index_select(attn_output, 2, self.reorder_index).to(DEV)

        # Quantize the attention output
        attn_output = self.act_quant(attn_output).to(DEV)
        attn_output = self.o_proj(attn_output).to(DEV)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class QLlamaMLP(nn.Module):
    def __init__(self, originalMLP: LlamaMLP, args):
        super().__init__()
        self.gate_proj = QLinearLayer(originalMLP.gate_proj, args)
        self.down_proj = QLinearLayer(originalMLP.down_proj, args)
        self.up_proj = QLinearLayer(originalMLP.up_proj, args)
        self.act_fn = originalMLP.act_fn
        self.act_quant = Quantizer(args=args)
        # self.register_buffer("act_shifts", None)

    def to(self, *args, **kwargs):
        super(QLlamaMLP, self).to(*args, **kwargs)
        self.gate_proj = self.gate_proj.to(*args, **kwargs)
        self.down_proj = self.down_proj.to(*args, **kwargs)
        self.up_proj = self.up_proj.to(*args, **kwargs)
        self.act_quant = self.act_quant.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        x = x.to(DEV)
        print(f"QLlamaMLP: x device: {x.device}")
        # input X: [b, seq, dim]: quantized
        tmpResult = self.act_fn(self.gate_proj(x)).to(DEV) * self.up_proj(x).to(DEV)
        print(f"QLlamaMLP: tmpResult device: {tmpResult.device}")
        # Quantize the activations and feed into down_proj
        tmpResult = self.act_quant(tmpResult).to(DEV)
        print(f"QLlamaMLP: tmpResult after quant device: {tmpResult.device}")
        return self.down_proj(tmpResult).to(DEV)
