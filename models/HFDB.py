from models import common
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.wave import DWT_2D, IDWT_2D
from einops import rearrange
import numbers

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y





class Dense(nn.Module):
    def __init__(self, in_channels):
        super(Dense, self).__init__()

        # self.norm = nn.LayerNorm([in_channels, 128, 128])  # Assuming input size is [224, 224]
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,stride=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,stride=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,stride=1)
        self.conv4 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,stride=1)
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,stride=1)
        self.conv6 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)

        self.gelu = nn.GELU()

    def forward(self, x):

        x1 = self.conv1(x)
        x1 = self.gelu(x1+x)

        x2 = self.conv2(x1)
        x2 = self.gelu(x2+x1+x)

        x3 = self.conv3(x2)
        x3 = self.gelu(x3+x2+x1+x)

        x4 = self.conv4(x3)
        x4 = self.gelu(x4+x3+x2+x1+x)

        x5 = self.conv5(x4)
        x5 = self.gelu(x5+x4+x3+x2+x1+x)

        x6= self.conv6(x5)
        x6 = self.gelu(x6+x5+x4+x3+x2+x1+x)

        return x6



class ResNet(nn.Module):
    def __init__(self, in_channels):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        out1 = F.gelu(self.conv1(x))
        out2 = F.gelu(self.conv2(out1))
        out2 += x  # Residual connection
        return out2








class Fusion(nn.Module):
    def __init__(self, in_channels, wave):
        super(Fusion, self).__init__()
        self.dwt = DWT_2D(wave)
        self.convh1 = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.high = ResNet(in_channels)
        self.convh2 = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1, stride=1, padding=0, bias=True)
        self.convl = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.low = ResNet(in_channels)


        self.idwt = IDWT_2D(wave)


    def forward(self, x1,x2):
        b, c, h, w = x1.shape
        x_dwt = self.dwt(x1)
        ll, lh, hl, hh = x_dwt.split(c, 1)
        high = torch.cat([lh, hl, hh], 1)
        high1=self.convh1(high)
        high2= self.high(high1)
        highf=self.convh2(high2)
        b1, c1, h1, w1 = ll.shape
        b2, c2, h2, w2 = x2.shape

        #
        if(h1!=h2):
            x2 =F.pad(x2, (0, 0, 1, 0), "constant", 0)


        low=torch.cat([ll, x2], 1)
        low = self.convl(low)
        lowf=self.low(low)

        out = torch.cat((lowf, highf), 1)
        out_idwt = self.idwt(out)

        return out_idwt

class UNet(nn.Module):
    def __init__(self, in_channels, wave):
        super(UNet, self).__init__()
        # Define the layers
        self.trans1 = TransformerBlock(in_channels,8, 2.66, False, 'WithBias')
        self.trans2 = TransformerBlock(in_channels,8, 2.66, False, 'WithBias')
        self.trans3 = TransformerBlock(in_channels,8, 2.66, False, 'WithBias')
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2)

        self.upsample1 = Fusion(in_channels, wave)
        self.upsample2 = Fusion(in_channels, wave)


    def forward(self, x):
        x1=x
        # print(x1.shape)
        x1_r = self.trans1(x)
        x2 = self.avgpool1(x1)
        # print(x2.shape)
        x2_r = self.trans2(x2)
        x3 = self.avgpool2(x2)
        # print(x3.shape)
        x3_r = self.trans3(x3)

        x4 = self.upsample1(x2_r,x3_r)


        out=self.upsample2(x1_r,x4)
        b1, c1, h1, w1 = out.shape
        b2, c2, h2, w2 = x.shape

        if (h1 != h2):
            out = F.pad(out, (0, 0, 1, 0), "constant", 0)

        return out+x




##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x




class HPB(nn.Module):
    def __init__(self, n_feats, wave):
        super(HPB, self).__init__()
        self.down = nn.AvgPool2d(kernel_size=2)
        self.dense=Dense(n_feats)
        self.unet=UNet(n_feats, wave)


        self.alise1= nn.Conv2d(2 * n_feats, n_feats, 1, 1, 0)  # one_module(n_feats)
        self.alise2 = nn.Conv2d(n_feats, n_feats, 3, 1, 1)  # one_module(n_feats)

        self.att = CALayer(n_feats)


    def forward(self, x):
        low = self.down(x)
        high = x - F.interpolate(low, size=x.size()[-2:], mode='bilinear', align_corners=True)

        lowf=self.unet(low)
        highfeat = self.dense(high)
        lowfeat = F.interpolate(lowf, size=x.size()[-2:], mode='bilinear', align_corners=True)

        out=self.alise2(self.att(self.alise1(torch.cat([highfeat, lowfeat], dim=1)))) + x

        return out



class UN(nn.Module):
    def __init__(self, n_feats):
        super(UN, self).__init__()

        self.encoder1 = HPB(n_feats, 'haar')
        self.encoder2 = HPB(n_feats, 'haar')
        self.encoder3 = HPB(n_feats, 'haar')
        self.conv=nn.Conv2d(n_feats * 3, n_feats, kernel_size=1)



    def forward(self, x):


        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        out=torch.cat((x1,x2,x3),1)
        out=self.conv(out)

        return out



