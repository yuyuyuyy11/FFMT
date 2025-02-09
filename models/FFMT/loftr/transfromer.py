import copy
import torch
import torch.nn as nn
# from .linear_attention import LinearAttention, FullAttention
import sys 
# print(sys.path)
sys.path.append('/home/yue/Projects/pointMBF')
from models.FFMT.loftr.linear_attention import LinearAttention, FullAttention
from utils.time import time_it
class LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(LocalFeatureTransformer, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        encoder_layer = LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'])
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == 'cross':
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError

        return feat0, feat1


class SelfAttention(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(SelfAttention, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        encoder_layer = LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'])
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == 'cross':
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError

        return feat0, feat1



def test01():
    config = {'d_model':32,
                'nhead':8,
                'layer_names':['self', 'cross'] * 4,
                'attention':'linear'}
    loftr_coarse = LocalFeatureTransformer(config)
    a = torch.randn([1,16,32])
    b = torch.randn([1,16,32])
    c = loftr_coarse(a,b)
    print(c[0].shape,c[1].shape,len(c))


@time_it
def test02():
    d_model = 128
    config = {'d_model':d_model,
                'nhead':8,
                'layer_names':['cross'],
                'attention':'linear'}
    loftr_coarse = LocalFeatureTransformer(config)
    a = torch.randn([1,131072,d_model])
    b = torch.randn([1,131072,d_model])
    c = loftr_coarse(a,b)
    print(c[0].shape,c[1].shape,len(c))

def test03():
    d_model = 16
    config = {'d_model':d_model,
                'nhead':8,
                'layer_names':['self'],
                'attention':'linear'}
    loftr_coarse = LocalFeatureTransformer(config)
    a = torch.randn([1,2,d_model])
    b = torch.randn([1,2,d_model])
    c = loftr_coarse(a,a)
    print(c[0].shape,c[1].shape,len(c))
    print(c[0])
    print(c[1])

# def test04():
#     config = {'d_model':16,
#               'nhead':8,
#               'layer_names':['self', 'cross'] * 4,
#               'attention':'linear'}
#     loftr_coarse = LocalFeatureTransformer(config)
#     a = torch.randn([1,16,16])
#     b = torch.randn([1,16,16])
#     c = loftr_coarse(a,b)
#     print(c[0].shape)

@time_it
def test05():
    d_model = 256
    it = 1024
    config = {'d_model':d_model,
                'nhead':8,
                'layer_names':['self', 'cross'] * 2,
                'attention':'linear'}
    loftr_coarse = LocalFeatureTransformer(config)
    a = torch.randn([1,it,d_model])
    b = torch.randn([1,it,d_model])
    c = loftr_coarse(a,a)
    print(c[0].shape,c[1].shape,len(c))
    # print(c[0])
    # print(c[1])

def test06():
    from thop import profile, clever_format
    it = 1024
    d_model = 256
    model = LocalFeatureTransformer({
        'd_model':d_model,
        'nhead':8,
        'layer_names':['self','cross'],
        'attention':'linear'
    })
    a = torch.randn(1,it,d_model)
    
    flops, params = profile(model, inputs=(a,a))
    flops, params = clever_format([flops, params], "%.3f")
    print(f"FLOPs: {flops}")
    print(f"Parameters: {params}")
    
def test07():
    self_attn_p1 = LoFTREncoderLayer(d_model=128,nhead=8)
    p = torch.rand([100,128])
    p1 = torch.chunk(p,2,dim=0)
    p1 = [torch.reshape(p1[i],(1,p1[i].size()[0],p1[i].size()[1])) for i in range(2)]
    p1 = [self_attn_p1(p1[i],p1[i]).squeeze() for i in range(2)]
    p2 = torch.cat(p1,dim=0)

    

if __name__ == "__main__":
    # test01()
    # test02()
    # test03()
    # test04()
    # test06()
    test07()