"""Modernized GPT architecture in plain PyTorch.
Features:
- Rotary Position Embeddings (RoPE)
- RMSNorm
- SwiGLU MLP
- Weight Tying
- Scaled Residual Initialization
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    vocab_size = 256      # default byte-level
    block_size = 256
    n_layer = 5
    n_head = 8
    n_embd = 160
    dropout = 0.0
    tie_weights = True
    use_rope = True
    use_rmsnorm = True
    use_swiglu = True


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm_x * self.weight


def apply_rotary_emb(x, cos, sin):
    # x: [B, n_head, T, head_dim]
    d = x.shape[-1]
    x1 = x[..., :d//2]
    x2 = x[..., d//2:]
    rotated_x = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotated_x * sin)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len):
        return self.cos_cached[:seq_len, :], self.sin_cached[:seq_len, :]


class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.use_rope = getattr(cfg, "use_rope", True)
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos=None, sin=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_rope and cos is not None and sin is not None:
            q = apply_rotary_emb(q, cos[None, None, :T, :], sin[None, None, :T, :])
            k = apply_rotary_emb(k, cos[None, None, :T, :], sin[None, None, :T, :])

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2 * (4 * dim) / 3)
            hidden_dim = 64 * ((hidden_dim + 63) // 64)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        use_rmsnorm = getattr(cfg, "use_rmsnorm", True)
        use_swiglu = getattr(cfg, "use_swiglu", True)

        self.norm1 = RMSNorm(cfg.n_embd) if use_rmsnorm else nn.LayerNorm(cfg.n_embd)
        self.attn = SelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd) if use_rmsnorm else nn.LayerNorm(cfg.n_embd)

        if use_swiglu:
            self.mlp = SwiGLU(cfg.n_embd, dropout=cfg.dropout)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
                nn.GELU(),
                nn.Linear(4 * cfg.n_embd, cfg.n_embd),
                nn.Dropout(cfg.dropout)
            )

    def forward(self, x, cos=None, sin=None):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)

        self.use_rope = getattr(cfg, "use_rope", True)
        if self.use_rope:
            head_dim = cfg.n_embd // cfg.n_head
            self.rope = RotaryEmbedding(head_dim, max_seq_len=cfg.block_size * 2)
            self.pos_emb = None
        else:
            self.rope = None
            self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)

        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        use_rmsnorm = getattr(cfg, "use_rmsnorm", True)
        self.ln_f = RMSNorm(cfg.n_embd) if use_rmsnorm else nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        if getattr(cfg, "tie_weights", True):
            self.head.weight = self.tok_emb.weight

        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            std = 0.02
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if self.use_rope:
            cos, sin = self.rope(T)
            cos, sin = cos.to(idx.device), sin.to(idx.device)
            x = self.drop(self.tok_emb(idx))
        else:
            cos, sin = None, None
            pos = torch.arange(T, device=idx.device)
            x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None, :, :])

        for blk in self.blocks:
            x = blk(x, cos, sin)

        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
