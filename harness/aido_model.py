"""aido_model.py — AIDO.RNA (genbio-ai/AIDO.RNA-650M, -1.6B) as embed_mm.py's tokenizer + model pair.

The forward is genbio's RNABert layout written out in plain torch: no embedding norm and no position table; per
block, pre-LN attention (q, k, v, o with bias; rotate-half RoPE over the whole head at base 10000) added to the
block's input, then pre-LN SwiGLU (gate, up, down with bias) added again; a final LayerNorm. Read straight off the
checkpoint's keys. Checked on mrl-hl rows: masked-base accuracy 0.37-0.49 through the released MLM head (chance 0.25).
Adding the attention branch to the normed input instead scores 0.19-0.29.
Their vocabulary: [PAD] 0, [MASK] 1, [CLS] 2, [SEP] 3, [UNK] 4, A 5, G 6, C 7, T 8, U 9, N 10; the model was
trained on T (the U row sits with the unused ones), so U is read as T.
"""
import glob
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

SIZES = {"650m": (1280, 33, 20), "1.6b": (2048, 32, 32)}
IDS = {"A": 5, "G": 6, "C": 7, "T": 8, "U": 8}


class AidoTokenizer:
    """[CLS] + bases + [SEP] per string, right-padded with [PAD]; the slice of embed_mm's tokenizer it uses."""

    def __call__(self, texts, return_tensors="pt", padding=True):
        rows = [[2] + [IDS.get(c, 10) for c in t.upper()] + [3] for t in texts]
        width = max(len(r) for r in rows)
        ids = torch.zeros(len(rows), width, dtype=torch.long)
        mask = torch.zeros(len(rows), width, dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, :len(r)] = torch.tensor(r)
            mask[i, :len(r)] = 1
        return {"input_ids": ids, "attention_mask": mask}


class AidoModel(nn.Module):
    def __init__(self, pattern: str, size: str, dtype=torch.bfloat16):
        super().__init__()
        self.H, self.layers, self.heads = SIZES[size]
        state = {}
        for path in sorted(glob.glob(pattern)):
            state.update(load_file(path))
        self.w = {k: v.to(dtype) for k, v in state.items() if not k.startswith("cls.")}
        self.config = SimpleNamespace(hidden_size=self.H, _attn_implementation="sdpa")

    def to(self, device):
        self.w = {k: v.to(device) for k, v in self.w.items()}
        return self

    def eval(self):
        return self

    def forward(self, input_ids, attention_mask):
        w, H, heads = self.w, self.H, self.heads
        B, L = input_ids.shape
        hd = H // heads
        x = w["bert.embeddings.word_embeddings.weight"][input_ids]
        inv = 1.0 / (10000 ** (torch.arange(0, hd, 2, dtype=torch.float32, device=x.device) / hd))
        ang = torch.outer(torch.arange(L, dtype=torch.float32, device=x.device), inv)
        ang = torch.cat([ang, ang], -1)
        cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
        keys = attention_mask.bool()[:, None, None, :]

        def rope(t):
            t1, t2 = t.chunk(2, -1)
            return t * cos + torch.cat([-t2, t1], -1) * sin

        def lin(h, name):
            return F.linear(h, w[name + ".weight"], w.get(name + ".bias"))

        for i in range(self.layers):
            p = f"bert.encoder.layer.{i}."
            h = F.layer_norm(x, (H,), w[p + "attention.ln.weight"], w[p + "attention.ln.bias"], 1e-5)
            q, k, v = (lin(h, p + f"attention.self.{n}").view(B, L, heads, hd).transpose(1, 2) for n in ("query", "key", "value"))
            a = F.scaled_dot_product_attention(rope(q), rope(k), v, attn_mask=keys).transpose(1, 2).reshape(B, L, H)
            x = x + lin(a, p + "attention.output.dense")
            h = F.layer_norm(x, (H,), w[p + "ln.weight"], w[p + "ln.bias"], 1e-5)
            x = x + lin(F.silu(lin(h, p + "mlp.gate_proj")) * lin(h, p + "mlp.up_proj"), p + "mlp.down_proj")
        x = F.layer_norm(x, (H,), w["bert.encoder.ln.weight"], w["bert.encoder.ln.bias"], 1e-5)
        return SimpleNamespace(last_hidden_state=x)

    __call__ = forward
