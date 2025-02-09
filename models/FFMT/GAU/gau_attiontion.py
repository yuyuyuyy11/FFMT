import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
import copy
from einops import rearrange

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def padding_to_multiple_of(n, mult):
    remainder = n % mult
    if remainder == 0:
        return 0
    return mult - remainder

class ScaleNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim = -1, keepdim = True) * self.scale
        return x / norm.clamp(min = self.eps) * self.g

# absolute positional encodings

class ScaledSinuEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1,))
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        n, device = x.shape[1], x.device
        t = torch.arange(n, device = device).type_as(self.inv_freq)
        sinu = einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((sinu.sin(), sinu.cos()), dim = -1)
        return emb * self.scale

# T5 relative positional bias

class T5RelativePositionBias(nn.Module):
    def __init__(
        self,
        scale,
        causal = False,
        num_buckets = 32,
        max_distance = 128
    ):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, 1)

    @staticmethod
    def _relative_position_bucket(
        relative_position,
        causal = True,
        num_buckets = 32,
        max_distance = 128
    ):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, x):
        i, j, device = *x.shape[-2:], x.device
        q_pos = torch.arange(i, dtype = torch.long, device = device)
        k_pos = torch.arange(j, dtype = torch.long, device = device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, causal = self.causal, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j 1 -> i j')
        return bias * self.scale

# class

class OffsetScale(nn.Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        self.beta = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.gamma, std = 0.02)

    def forward(self, x):
        out = einsum('... d, h d -> ... h d', x, self.gamma) + self.beta
        return out.unbind(dim = -2)

# activation functions

class ReLUSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2

class LaplacianAttnFn(nn.Module):
    """ https://arxiv.org/abs/2209.10655 claims this is more stable than Relu squared """

    def forward(self, x):
        mu = math.sqrt(0.5)
        std = math.sqrt((4 * math.pi) ** -1)
        return (1 + torch.special.erf((x - mu) / (std * math.sqrt(2)))) * 0.5

# gated attention unit


class GAU(nn.Module):
    def __init__(
        self,
        *,
        dim,
        query_key_dim = 128,
        expansion_factor = 2,
        add_residual = True,
        causal = False,
        dropout = 0.,
        laplace_attn_fn = False,
        rel_pos_bias = False,
        norm_klass = nn.LayerNorm
    ):
        super().__init__()
        hidden_dim = int(expansion_factor * dim)

        self.norm = norm_klass(dim)
        self.causal = causal
        self.dropout = nn.Dropout(dropout)

        self.attn_fn = ReLUSquared() if not laplace_attn_fn else LaplacianAttnFn()

        self.rel_pos_bias = T5RelativePositionBias(scale = dim ** 0.5, causal = causal)

        self.to_hidden = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),
            nn.SiLU()
        )

        self.to_qk = nn.Sequential(
            nn.Linear(dim, query_key_dim),
            nn.SiLU()
        )

        self.offsetscale = OffsetScale(query_key_dim, heads = 2)

        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

        self.add_residual = add_residual

    def forward(self,x1,x2,rel_pos_bias = None,mask = None):
        
        seq_len, device = x1.shape[-2], x1.device

        normed_x1 = self.norm(x1)
        normed_x2 = self.norm(x2)
        v, gate = self.to_hidden(normed_x1).chunk(2, dim = -1)

        qk = self.to_qk(normed_x2)
        q, k = self.offsetscale(qk)

        sim = einsum('b i d, b j d -> b i j', q, k)

        # if exists(self.rel_pos_bias):
        #     sim = sim + self.rel_pos_bias(sim)

        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias

        attn = self.attn_fn(sim / seq_len)
        attn = self.dropout(attn)

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 j')
            attn = attn.masked_fill(~mask, 0.)

        if self.causal:
            causal_mask = torch.ones((seq_len, seq_len), dtype = torch.bool, device = device).triu(1)
            attn = attn.masked_fill(causal_mask, 0.)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = out * gate

        out = self.to_out(out)

        if self.add_residual:
            out = out + x1

        return out
    

class GAU_attn(nn.Module):
    def __init__(self,config) -> None:
        super(GAU_attn,self).__init__()
        self.config = config
        self.d_model = config['d_model']
        self.layer_names = config['layer_names']
        self.query_key_dim = config['query_key_dim']
        self.laplace_attn_fn = config['laplace_attn_fn']

        self.causal = True
        self.expansion_factor = 2
        
        
        self.GAU = GAU(dim = self.d_model,query_key_dim=self.query_key_dim,causal=self.causal,expansion_factor=self.expansion_factor,laplace_attn_fn=self.laplace_attn_fn)
        self.att_layer = nn.ModuleList([copy.deepcopy(self.GAU) for _ in range(len(self.layer_names))])
    
    def forward(self, feat0, feat1,rel_pos_bias = None,mask = None):
        for layer, name in zip(self.att_layer, self.layer_names):
            if name == 'self':
                feat0 = layer(feat0, feat0, rel_pos_bias, mask)
                feat1 = layer(feat1, feat1, rel_pos_bias, mask)
            elif name == 'cross':
                feat0 = layer(feat0, feat1, rel_pos_bias, mask)
                feat1 = layer(feat1, feat0, rel_pos_bias, mask)
            else:
                raise KeyError

        return feat0, feat1
    
    
def test01():
    gau = GAU(
    dim = 512,
    query_key_dim = 128,     # query / key dimension
    causal = True,           # autoregressive or not
    expansion_factor = 2,    # hidden dimension = dim * expansion_factor
    laplace_attn_fn = True   # new Mega paper claims this is more stable than relu squared as attention function
    )

    x1 = torch.randn(2, 1024, 512)
    x2 = torch.randn(2, 1024, 512)
    out = gau(x1,x2) # (1, 1024, 512)
    print(out.size())

import sys 
sys.path.append('/home/yue/Projects/pointMBF')
from utils.time import time_it

@time_it
def test02():
    it = 1024
    d_model = 256
    model = GAU_attn({
        'd_model':d_model,
        'layer_names':['self','cross']*2
    })
    a = torch.randn(1,it,d_model)
    b = torch.randn(1,it,d_model)
    c = model(a,b)
    print(c[0].size(),c[1].size())

@time_it
def test03():
    from thop import profile, clever_format
    it = 16384
    d_model = 32
    model = GAU_attn({
        'd_model':d_model,
        'layer_names':['self','cross']
    })
    a = torch.randn(2,it,d_model)
    
    flops, params = profile(model, inputs=(a,a))
    flops, params = clever_format([flops, params], "%.3f")
    print(f"FLOPs: {flops}")
    print(f"Parameters: {params}")
    
    
if __name__ == "__main__":
    # test02()
    test03()