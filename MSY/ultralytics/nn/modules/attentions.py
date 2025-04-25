import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad, ScConv
from .transformer import TransformerBlock

__all__ = (
    "DoubleAttention",
    "BAMBlock",
    # "S2Attention",
    "SimAM",
    "MultiQueryAttentionLayerWithDownSampling",
)


class DoubleAttention(nn.Module):

    def __init__(self, in_channels,c_m=128,c_n=128,reconstruct = True):
        super().__init__()
        self.in_channels = in_channels
        self.reconstruct = reconstruct
        self.c_m = c_m
        self.c_n = c_n
        self.convA = nn.Conv2d(in_channels,c_m,1)
        self.convB = nn.Conv2d(in_channels,c_n,1)
        self.convV = nn.Conv2d(in_channels,c_n,1)
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv2d(c_m, in_channels, kernel_size = 1)
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        assert c == self.in_channels
        A = self.convA(x)  # b,c_m,h,w
        B = self.convB(x)  # b,c_n,h,w
        V = self.convV(x)  # b,c_n,h,w
        tmpA = A.view(b, self.c_m,-1)
        attention_maps = F.softmax(B.view(b, self.c_n, -1), 1)
        attention_vectors = F.softmax(V.view(b, self.c_n, -1), 1)
        # step 1: feature gating
        global_descriptors = torch.bmm(tmpA, attention_maps.permute(0, 2, 1))  # b.c_m,c_n
        # step 2: feature distribution
        tmpZ = global_descriptors.matmul(attention_vectors)  # b,c_m,h*w
        tmpZ = tmpZ.view(b, self.c_m, h, w)  # b,c_m,h,w
        if self.reconstruct:
            tmpZ = self.conv_reconstruct(tmpZ)

        return tmpZ

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16, num_layers=3):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        gate_channels = [channel]
        gate_channels += [channel // reduction] * num_layers
        gate_channels += [channel]

        self.ca = nn.Sequential()
        self.ca.add_module('flatten', Flatten())
        for i in range(len(gate_channels) - 2):
            self.ca.add_module('fc%d' % i, nn.Linear(gate_channels[i], gate_channels[i + 1]))
            self.ca.add_module('bn%d' % i, nn.BatchNorm1d(gate_channels[i + 1]))
            self.ca.add_module('relu%d' % i, nn.ReLU())
        self.ca.add_module('last_fc', nn.Linear(gate_channels[-2], gate_channels[-1]))

    def forward(self, x):
        res = self.avgpool(x)
        res = self.ca(res)
        res = res.unsqueeze(-1).unsqueeze(-1).expand_as(x)
        return res


class SpatialAttention(nn.Module):
    def __init__(self, channel, reduction=16, num_layers=3, dia_val=2):
        super().__init__()
        self.sa = nn.Sequential()
        self.sa.add_module('conv_reduce1',
                           nn.Conv2d(kernel_size=1, in_channels=channel, out_channels=channel // reduction))
        self.sa.add_module('bn_reduce1', nn.BatchNorm2d(channel // reduction))
        self.sa.add_module('relu_reduce1', nn.ReLU())
        for i in range(num_layers):
            self.sa.add_module('conv_%d' % i, nn.Conv2d(kernel_size=3, in_channels=channel // reduction,
                                                        out_channels=channel // reduction, padding=autopad(3, None, dia_val), dilation=dia_val))
            self.sa.add_module('bn_%d' % i, nn.BatchNorm2d(channel // reduction))
            self.sa.add_module('relu_%d' % i, nn.ReLU())
        self.sa.add_module('last_conv', nn.Conv2d(channel // reduction, 1, kernel_size=1))

    def forward(self, x):
        res = self.sa(x)
        res = res.expand_as(x)
        return res


class BAMBlock(nn.Module):
    def __init__(self, channel=512, reduction=16, dia_val=2):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(channel=channel, reduction=reduction, dia_val=dia_val)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        sa_out = self.sa(x)
        ca_out = self.ca(x)
        weight = self.sigmoid(sa_out + ca_out)
        out = (1 + weight) * x
        return out



class SimAM(torch.nn.Module):
    def __init__(self, e_lambda=1e-4, *args):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


# def spatial_shift1(x):
#     b, w, h, c = x.size()
#     x[:, 1:, :, :c // 4] = x[:, :w - 1, :, :c // 4]
#     x[:, :w - 1, :, c // 4:c // 2] = x[:, 1:, :, c // 4:c // 2]
#     x[:, :, 1:, c // 2:c * 3 // 4] = x[:, :, :h - 1, c // 2:c * 3 // 4]
#     x[:, :, :h - 1, 3 * c // 4:] = x[:, :, 1:, 3 * c // 4:]
#     return x
#
#
# def spatial_shift2(x):
#     b, w, h, c = x.size()
#     x[:, :, 1:, :c // 4] = x[:, :, :h - 1, :c // 4]
#     x[:, :, :h - 1, c // 4:c // 2] = x[:, :, 1:, c // 4:c // 2]
#     x[:, 1:, :, c // 2:c * 3 // 4] = x[:, :w - 1, :, c // 2:c * 3 // 4]
#     x[:, :w - 1, :, 3 * c // 4:] = x[:, 1:, :, 3 * c // 4:]
#     return x


# class SplitAttention(nn.Module):
#     def __init__(self, channel=512, k=3):
#         super().__init__()
#         self.channel = channel
#         self.k = k
#         self.mlp1 = nn.Linear(channel, channel, bias=False)
#         self.gelu = nn.GELU()
#         self.mlp2 = nn.Linear(channel, channel * k, bias=False)
#         self.softmax = nn.Softmax(1)
#
#     def forward(self, x_all):
#         b, k, h, w, c = x_all.shape
#         x_all = x_all.reshape(b, k, -1, c)
#         a = torch.sum(torch.sum(x_all, 1), 1)
#         hat_a = self.mlp2(self.gelu(self.mlp1(a)))
#         hat_a = hat_a.reshape(b, self.k, c)
#         bar_a = self.softmax(hat_a)
#         attention = bar_a.unsqueeze(-2)
#         out = attention * x_all
#         out = torch.sum(out, 1).reshape(b, h, w, c)
#         return out
#
#
# class S2Attention(nn.Module):
#
#     def __init__(self, channels=512, *args, **kwargs):
#         super().__init__()
#         self.mlp1 = nn.Linear(channels, channels * 3)
#         self.mlp2 = nn.Linear(channels, channels)
#         self.split_attention = SplitAttention()
#
#     def forward(self, x):
#         b, c, w, h = x.size()
#         x = x.permute(0, 2, 3, 1)
#         x = self.mlp1(x)
#         x1 = spatial_shift1(x[:, :, :, :c])
#         x2 = spatial_shift2(x[:, :, :, c:c * 2])
#         x3 = x[:, :, :, c * 2:]
#         x_all = torch.stack([x1, x2, x3], 1)
#         a = self.split_attention(x_all)
#         x = self.mlp2(a)
#         x = x.permute(0, 3, 1, 2)
# #         return x
#
#
# class S2_MLPv2(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.mlp1 = nn.Linear(channels, channels * 3)
#         self.mlp2 = nn.Linear(channels, channels)
#         self.split_attention = SplitAttention()
#     def forward(self, x):
#         b, w, h, c = x.size()
#         x = self.mlp1(x)
#         x1 = spatial_shift1(x[:, :, :, :c//3])
#         x2 = spatial_shift2(x[:, :, :, c//3:c//3*2])
#         x3 = x[:, :, :, c//3*2:]
#         a = self.split_attention(x1, x2, x3)
#         x = self.mlp2(a)
#         return x


# def conv2d(in_channels, out_channels, kernel_size=3, stride=1, groups=1, bias=False, norm=True, act=True):
#     conv = nn.Sequential()
#     padding = (kernel_size - 1) // 2
#     conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, groups=groups))
#     if norm:
#         conv.append(nn.BatchNorm2d(out_channels))
#     if act:
#         conv.append(nn.ReLU6())
#     return conv
#
# class MultiQueryAttentionLayerWithDownSampling(nn.Module):
#     def __init__(self, in_channels, num_heads, key_dim, value_dim, query_h_strides, query_w_strides, kv_strides, dw_kernel_size=3, dropout=0.0):
#         """Multi Query Attention with spatial downsampling.
#         Referenced from here https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_blocks.py
#
#         3 parameters are introduced for the spatial downsampling:
#         1. kv_strides: downsampling factor on Key and Values only.
#         2. query_h_strides: vertical strides on Query only.
#         3. query_w_strides: horizontal strides on Query only.
#
#         This is an optimized version.
#         1. Projections in Attention is explict written out as 1x1 Conv2D.
#         2. Additional reshapes are introduced to bring a up to 3x speed up.
#         """
#         super(MultiQueryAttentionLayerWithDownSampling, self).__init__()
#         self.num_heads = num_heads
#         self.key_dim = key_dim
#         self.value_dim = value_dim
#         self.query_h_strides = query_h_strides
#         self.query_w_strides = query_w_strides
#         self.kv_strides = kv_strides
#         self.dw_kernel_size = dw_kernel_size
#         self.dropout = dropout
#
#         self.head_dim = self.key_dim // num_heads
#
#         if self.query_h_strides > 1 or self.query_w_strides > 1:
#             self._query_downsampling_norm = nn.BatchNorm2d(in_channels)
#         self._query_proj = conv2d(in_channels, self.num_heads * self.key_dim, 1, 1, norm=False, act=False)
#
#         if self.kv_strides > 1:
#             self._key_dw_conv = conv2d(in_channels, in_channels, dw_kernel_size, kv_strides, groups=in_channels,
#                                        norm=True, act=False)
#             self._value_dw_conv = conv2d(in_channels, in_channels, dw_kernel_size, kv_strides, groups=in_channels,
#                                          norm=True, act=False)
#         self._key_proj = conv2d(in_channels, key_dim, 1, 1, norm=False, act=False)
#         self._value_proj = conv2d(in_channels, key_dim, 1, 1, norm=False, act=False)
#         self._output_proj = conv2d(num_heads * key_dim, in_channels, 1, 1, norm=False, act=False)
#
#         self.dropout = nn.Dropout(p=dropout)
#
#     def forward(self, x):
#         bs, seq_len, _, _ = x.size()
#         # print(x.size())
#         if self.query_h_strides > 1 or self.query_w_strides > 1:
#             q = F.avg_pool2d(self.query_h_strides, self.query_w_strides)
#             q = self._query_downsampling_norm(q)
#             q = self._query_proj(q)
#         else:
#             q = self._query_proj(x)
#         px = q.size(2)
#         q = q.view(bs, self.num_heads, -1, self.key_dim)  # [batch_size, num_heads, seq_len, key_dim]
#
#         if self.kv_strides > 1:
#             k = self._key_dw_conv(x)
#             k = self._key_proj(k)
#             v = self._value_dw_conv(x)
#             v = self._value_proj(v)
#         else:
#             k = self._key_proj(x)
#             v = self._value_proj(x)
#         k = k.view(bs, 1, self.key_dim, -1)   # [batch_size, 1, key_dim, seq_length]
#         v = v.view(bs, 1, -1, self.key_dim)    # [batch_size, 1, seq_length, key_dim]
#
#         # calculate attention score
#         # print(q.shape, k.shape, v.shape)
#         attn_score = torch.matmul(q, k) / (self.head_dim ** 0.5)
#         attn_score = self.dropout(attn_score)
#         attn_score = F.softmax(attn_score, dim=-1)
#
#         # context = torch.einsum('bnhm,bmv->bnhv', attn_score, v)
#         # print(attn_score.shape, v.shape)
#         context = torch.matmul(attn_score, v)
#         context = context.view(bs, self.num_heads * self.key_dim, px, px)
#         output = self._output_proj(context)
#         # print(output.shape)
#         return output




def conv_2d(inp, oup, kernel_size=3, stride=1, groups=1, bias=False, norm=True, act=True):
    conv = nn.Sequential()
    padding = (kernel_size - 1) // 2
    conv.add_module('conv', nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=bias, groups=groups))
    if norm:
        conv.add_module('BatchNorm2d', nn.BatchNorm2d(oup))
    if act:
        conv.add_module('Activation', nn.ReLU6())
    return conv


class MultiQueryAttentionLayerWithDownSampling(nn.Module):
    def __init__(self, inp, num_heads, key_dim, value_dim, query_h_strides, query_w_strides, kv_strides,
                 dw_kernel_size=3, dropout=0.0):
        """Multi Query Attention with spatial downsampling.
        Referenced from here https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_blocks.py
        3 parameters are introduced for the spatial downsampling:
        1. kv_strides: downsampling factor on Key and Values only.
        2. query_h_strides: vertical strides on Query only.
        3. query_w_strides: horizontal strides on Query only.
        This is an optimized version.
        1. Projections in Attention is explict written out as 1x1 Conv2D.
        2. Additional reshapes are introduced to bring a up to 3x speed up.
        """
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.query_h_strides = query_h_strides
        self.query_w_strides = query_w_strides
        self.kv_strides = kv_strides
        self.dw_kernel_size = dw_kernel_size
        self.dropout = dropout

        self.head_dim = key_dim // num_heads

        if self.query_h_strides > 1 or self.query_w_strides > 1:
            self._query_downsampling_norm = nn.BatchNorm2d(inp)
        self._query_proj = conv_2d(inp, num_heads * key_dim, 1, 1, norm=False, act=False)

        if self.kv_strides > 1:
            self._key_dw_conv = conv_2d(inp, inp, dw_kernel_size, kv_strides, groups=inp, norm=True, act=False)
            self._value_dw_conv = conv_2d(inp, inp, dw_kernel_size, kv_strides, groups=inp, norm=True, act=False)
        self._key_proj = conv_2d(inp, key_dim, 1, 1, norm=False, act=False)
        self._value_proj = conv_2d(inp, key_dim, 1, 1, norm=False, act=False)

        self._output_proj = conv_2d(num_heads * key_dim, inp, 1, 1, norm=False, act=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        batch_size, seq_length, _, _ = x.size()
        if self.query_h_strides > 1 or self.query_w_strides > 1:
            q = F.avg_pool2d(self.query_h_stride, self.query_w_stride)
            q = self._query_downsampling_norm(q)
            q = self._query_proj(q)
        else:
            q = self._query_proj(x)
        px = q.size(2)
        q = q.view(batch_size, self.num_heads, -1, self.key_dim)  # [batch_size, num_heads, seq_length, key_dim]

        if self.kv_strides > 1:
            k = self._key_dw_conv(x)
            k = self._key_proj(k)
            v = self._value_dw_conv(x)
            v = self._value_proj(v)
        else:
            k = self._key_proj(x)
            v = self._value_proj(x)
        k = k.view(batch_size, self.key_dim, -1)  # [batch_size, key_dim, seq_length]
        v = v.view(batch_size, -1, self.key_dim)  # [batch_size, seq_length, key_dim]

        # calculate attn score
        attn_score = torch.matmul(q, k) / (self.head_dim ** 0.5)
        attn_score = self.dropout(attn_score)
        attn_score = F.softmax(attn_score, dim=-1)

        context = torch.matmul(attn_score, v)
        context = context.view(batch_size, self.num_heads * self.key_dim, px, px)
        output = self._output_proj(context)
        return output