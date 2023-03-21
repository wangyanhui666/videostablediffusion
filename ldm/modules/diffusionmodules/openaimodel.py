from abc import abstractmethod
import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from ldm.modules.attention import SpatialTransformer
from ldm.util import exists


# dummy replace
def convert_module_to_f16(x):
    pass

def convert_module_to_f32(x):
    pass


## go
# class AttentionPool2d(nn.Module):
#     """
#     Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
#     """

#     def __init__(
#         self,
#         spacial_dim: int,
#         embed_dim: int,
#         num_heads_channels: int,
#         output_dim: int = None,
#     ):
#         super().__init__()
#         self.positional_embedding = nn.Parameter(th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5)
#         self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
#         self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
#         self.num_heads = embed_dim // num_heads_channels
#         self.attention = QKVAttention(self.num_heads)

#     def forward(self, x):
#         b, c, *_spatial = x.shape
#         x = x.reshape(b, c, -1)  # NC(HW)
#         x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
#         x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
#         x = self.qkv_proj(x)
#         x = self.attention(x)
#         x = self.c_proj(x)
#         return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x

def spatial_temporal_forward(x,spatial_layers,temporal_layers,identity_layer=nn.Identity(),emb=None,context=None):
    """
    fisrt do spacial forward
    then do temporal forward
    have skip connection
    in the temporal layers not change channel
    """
    b,c,*_,h,w=x.shape
    x=rearrange(x,'b c t h w -> (b t) c h w').contiguous()
    if isinstance(spatial_layers, nn.Module):
        # If the spatial_layers is a single convolutional layer or an nn.Sequential object
        if isinstance(spatial_layers, TimestepEmbedSequential):
            x = spatial_layers(x,emb,context)
        else:
            x = spatial_layers(x)
    elif isinstance(spatial_layers, list):
        # If the spatial_layers is a list of convolutional layers
        for layer in spatial_layers:
            x = layer(x)
    else:
        raise TypeError("spatial_layers argument must be a single convolutional layer, an nn.Sequential object, or a list of convolutional layers")
    
    bt,c,h,w=x.shape
    x=rearrange(x,'(b t) c h w -> (b h w) c t',b=b).contiguous()
    identity=identity_layer(x)
    if isinstance(temporal_layers, nn.Module):
        # If the temporal_layers is a single convolutional layer or an nn.Sequential object
        if isinstance(temporal_layers, TimestepEmbedSequential):
            x = temporal_layers(x,emb,context)
        else:
            x = temporal_layers(x)
    elif isinstance(temporal_layers, list):
        # If the temporal_layers is a list of convolutional layers
        for layer in temporal_layers:
            x = layer(x)
    elif temporal_layers==None:
        x=th.zeros_like(identity,device=identity.device,dtype=identity.dtype)
    else:
        raise TypeError("temporal_layers argument must be a single convolutional layer, an nn.Sequential object, a list of convolutional layers or None")
    x=x+identity
    x=rearrange(x, '(b h w) c t -> b c t h w', h=h, w=w).contiguous()
    return x


class Upsample(nn.Module):
    """
    can handle image/video as input
    temporal conv is zero initialized
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    timeupscale: up sample scale of time dim
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1, timeupscale=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.timeupscale=timeupscale

        if use_conv:
            if dims!=3:
                self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)
            else:
                self.conv = conv_nd(2,self.channels,self.out_channels,3,padding=padding)
                self.conv_temporal = zero_module(conv_nd(1,self.out_channels,self.out_channels,3,padding=padding))
    def forward(self, x):
        assert x.shape[1] == self.channels
        is_video=(len(x.shape)==5)
        if is_video:
            x = F.interpolate(
                x, (x.shape[2]*self.timeupscale, x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            if is_video:
                x=spatial_temporal_forward(x,self.conv,self.conv_temporal)
            else:
                x = self.conv(x)
        return x

# class TransposedUpsample(nn.Module):
#     'Learned 2x upsampling without padding'
#     def __init__(self, channels, out_channels=None, ks=5):
#         super().__init__()
#         self.channels = channels
#         self.out_channels = out_channels or channels

#         self.up = nn.ConvTranspose2d(self.channels,self.out_channels,kernel_size=ks,stride=2)

#     def forward(self,x):
#         return self.up(x)


class Downsample(nn.Module):
    """
    can handle image/video input
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    out_channels: default None (outputs channel same as input)
    timedownscale: 1 or 2 downsample scale of time dim
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None,padding=1,timedownscale=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.timedownscale=timedownscale
        if use_conv:
            stride = 2
            if dims!=3:
                self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=padding)
            else:
                self.op = conv_nd(2, self.channels, self.out_channels, 3, stride=stride, padding=padding)
                self.conv_temporal=zero_module(conv_nd(1, self.out_channels, self.out_channels, 3, stride=self.timedownscale, padding=padding))
                if self.timedownscale==1:
                    self.identity=nn.Identity()
                elif self.timedownscale==2:
                    self.identity=avg_pool_nd(1,kernel_size=stride,stride=stride)
        else:
            assert self.channels == self.out_channels
            if dims!=3:
                stride=2
                self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)
            else: 
                self.op = avg_pool_nd(2, kernel_size=2, stride=2)
                stride=(self.timedownscale,2,2)
                self.op_3d=avg_pool_nd(3,kernel_size=stride,stride=stride)
    def forward(self, x):
        assert x.shape[1] == self.channels
        is_video=(len(x.shape)==5)
        if self.use_conv:
            if is_video:
                x=spatial_temporal_forward(x,self.op,self.conv_temporal,self.identity)
            else:
                x = self.op(x)
        else:
            if is_video:
                x=self.op_3d(x)
            else:
                x=self.op(x)
        return x


class ResBlock(TimestepBlock):
    """
    Can handle image/video input
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        
        if dims!=3:
            self.in_layers = nn.Sequential(
                normalization(channels),
                nn.SiLU(),
                conv_nd(dims, channels, self.out_channels, 3, padding=1),
            )
        else:
            self.in_layers = nn.Sequential(
                normalization(channels),
                nn.SiLU(),
                conv_nd(2, channels, self.out_channels, 3, padding=1),
            )
            self.in_layers_temporal=nn.Sequential(
                normalization(self.out_channels),
                nn.SiLU(),
                zero_module(conv_nd(1, self.out_channels, self.out_channels, 3, padding=1)),
            )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()
        
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        if dims!=3:
            self.out_layers = nn.Sequential(
                normalization(self.out_channels),
                nn.SiLU(),
                nn.Dropout(p=dropout),
                zero_module(
                    conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
                ),
            )
        else:
            self.out_layers = nn.Sequential(
                normalization(self.out_channels),
                nn.SiLU(),
                nn.Dropout(p=dropout),
                zero_module(
                    conv_nd(2, self.out_channels, self.out_channels, 3, padding=1)
                ),
            )
            self.out_layers_temporal = nn.Sequential(
                normalization(self.out_channels),
                nn.SiLU(),
                nn.Dropout(p=dropout),
                zero_module(
                    conv_nd(1, self.out_channels, self.out_channels, 3, padding=1)
                ),
            )
        if self.out_channels == channels:
            if dims!=3:
                self.skip_connection = nn.Identity()
            else:
                self.skip_connection = nn.Identity()
                self.skip_connection_temporal = None
        elif use_conv:
            if dims!=3:
                self.skip_connection = conv_nd(
                    dims, channels, self.out_channels, 3, padding=1
                )
            else:
                self.skip_connection = conv_nd(2, channels, self.out_channels, 3, padding=1)
                self.skip_connection_temporal=zero_module(conv_nd(1,self.out_channels,self.out_channels,3,padding=1))
        else:
            if dims!=3:
                self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)
            else:
                self.skip_connection = conv_nd(2, channels, self.out_channels, 1)
                self.skip_connection_temporal = zero_module(conv_nd(1, self.out_channels, self.out_channels, 1))

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )


    def _forward(self, x, emb):
        is_video=(len(x.shape)==5)
        identity = x
        if self.updown:
            if is_video:
                in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
                # in_rest_t, in_conv_t = self.in_layers_temporal[:-1], self.in_layers_temporal[-1]
                # 3d
                x = in_rest(x)
                x = self.h_upd(x)
                # 2d+1d
                x = spatial_temporal_forward(x,in_conv,self.in_layers_temporal)
                # 3d
                identity = self.x_upd(identity)
            else:
                in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
                x = in_rest(x)
                x = self.h_upd(x)
                x = in_conv(x)
                identity = self.x_upd(identity)
        else:
            if is_video:
                x=spatial_temporal_forward(x,self.in_layers,self.in_layers_temporal)
            else:
                x = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(x.dtype)
        while len(emb_out.shape) < len(x.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            if is_video:
                out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
                scale, shift = th.chunk(emb_out, 2, dim=1)
                #3d
                x = out_norm(x) * (1 + scale) + shift
                # 2d+1d
                x = spatial_temporal_forward(x,out_rest,self.out_layers_temporal)
                
            else:
                out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
                scale, shift = th.chunk(emb_out, 2, dim=1)
                x = out_norm(x) * (1 + scale) + shift
                x = out_rest(x)
        else:
            if is_video:
                x= x + emb_out
                #2d + 1d
                x=spatial_temporal_forward(x,self.out_layers,self.out_layers_temporal)
            else:
                x = x + emb_out
                x = self.out_layers(x)
        if is_video:
            identity=spatial_temporal_forward(identity,self.skip_connection,self.skip_connection_temporal)
            x=identity+x
        else:
            x=self.skip_connection(identity)+x
        return x


class AttentionBlock(nn.Module):
    """
    3D attentionBlock
    Can handle image/video input
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    dims: 2 2Dmodel, 3 3Dmodel
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
        dims=2,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if dims==3:
            self.norm_temporal = normalization(channels)
            self.qkv_temporal = conv_nd(1, channels, channels * 3, 1)
            if use_new_attention_order:
                # split qkv before split heads
                self.attention_temporal = QKVAttention(self.num_heads)
            else:
                # split heads before split qkv
                self.attention_temporal = QKVAttentionLegacy(self.num_heads)
            self.proj_out_temporal=zero_module(conv_nd(1, channels, channels, 1))
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)   # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
        #return pt_checkpoint(self._forward, x)  # pytorch

    def _forward(self, x):
        is_video=(len(x.shape)==5)
        if is_video:
            b,c,t,h,w = x.shape
            # spatial attension
            x = rearrange(x,'b c t h w -> (b t) c (h w)')
            identity=x
            x=self.qkv(self.norm(x))
            x=self.attention(x)
            x=self.proj_out(x)
            x=x+identity
            x= rearrange(x, '(b t) c (h w) -> (b h w) c t', t=t,h=h,w=w)
            identity=x
            x=self.qkv_temporal(self.norm_temporal(x))
            x=self.attention_temporal(x)
            x=self.proj_out_temporal(x)
            x=x+identity
            x= rearrange(x,'(b h w) c t -> b c t h w' ,h=h,w=w)
        else:
            b, c, *spatial = x.shape
            x = x.reshape(b, c, -1)
            identity=x
            x = self.qkv(self.norm(x))
            x = self.attention(x)
            x= self.proj_out(x)
            x = x+identity
            x = x.reshape(b,c,*spatial)
        return x


def count_flops_attn(model, _x, y):
    """
    Need to add spatial temporal attension
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    if len(y[0].shape)==5:
        raise ValueError("not support spatial temporal attension")
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class UNetModel(nn.Module):
    """
    Unet3d
    Can handle image/video input.
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    timescale: time up/down scale
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        use_bf16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        timescale=1,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.dtype = th.bfloat16 if use_bf16 else self.dtype
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            else:
                raise ValueError()
        if dims!=3:
            self.input_blocks = nn.ModuleList(
                [
                    TimestepEmbedSequential(
                        conv_nd(dims, in_channels, model_channels, 3, padding=1)
                    )
                ]
            )
        else:
            self.input_blocks = nn.ModuleList(
                [
                    TimestepEmbedSequential(
                        conv_nd(2, in_channels, model_channels, 3, padding=1)
                    )
                ]
            )
            self.input_blocks_temporal=TimestepEmbedSequential(
                    zero_module(conv_nd(1, model_channels, model_channels, 3, padding=1))
                )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                                dims=dims
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint,dims=dims,
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch,timedownscale=timescale
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
                dims=dims,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                            disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                            use_checkpoint=use_checkpoint,dims=dims
                        ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or i < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_upsample,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                                dims=dims,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint,dims=dims
                            )
                        )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch,timeupscale=timescale)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
        if dims!=3:
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
            )
        else:
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                zero_module(conv_nd(2, model_channels, out_channels, 3, padding=1)),
            )
            self.out_temporal=nn.Sequential(
                # normalization(out_channels),
                nn.SiLU(),
                zero_module(conv_nd(1, out_channels, out_channels, 3, padding=1)),
            )
        if self.predict_codebook_ids:
            if dims!=3:
                self.id_predictor = nn.Sequential(
                normalization(ch),
                conv_nd(dims, model_channels, n_embed, 1),
                #nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
                )
            else:
                self.id_predictor = nn.Sequential(
                normalization(ch),
                conv_nd(2, model_channels, n_embed, 1),
                #nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
                )
                self.id_predictor_temporal=nn.Sequential(
                # normalization(model_channels),
                zero_module(conv_nd(1, n_embed, n_embed, 1)),
                #nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
                )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps=None, context=None, y=None,**kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        is_video=(len(x.shape)==5)
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for layer,module in enumerate(self.input_blocks):
            if layer==0 and is_video:
                h=spatial_temporal_forward(h,module,self.input_blocks_temporal,emb=emb,context=context)
                hs.append(h)
            else:
                h = module(h, emb, context)
                hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            if is_video:
                h=spatial_temporal_forward(h,self.id_predictor,self.id_predictor_temporal)
                return h
            else:
                return self.id_predictor(h)
        else:
            if is_video:
                h=spatial_temporal_forward(h,self.out,self.out_temporal)
                return h
            else:
                return self.out(h)

if __name__=="__main__":
    input=th.randn(2,4,64,64).cuda()
    input_emd=th.randn(2).cuda()
    use_checkpoint=True
    use_fp16=True
    image_size=32 # unused
    in_channels=4
    out_channels=4
    model_channels=320
    attention_resolutions=[ 4, 2, 1 ]
    num_res_blocks=2
    channel_mult=[ 1, 2, 4, 4 ]
    num_head_channels=64 # need to fix for flash-attn
    use_spatial_transformer=True
    use_linear_in_transformer=True
    transformer_depth=1
    context_dim=1024
    legacy=False
    # test_upsample=ResBlock(channels=32,emb_channels=32,dropout=0.5,out_channels=32,dims=3,use_conv=True)
    model=UNetModel(use_checkpoint=True,
    use_fp16=False,
    image_size=32, # unused
    in_channels=4,
    out_channels=4,
    model_channels=320,
    attention_resolutions=[ 4, 2, 1 ],
    num_res_blocks=2,
    channel_mult=[ 1, 2, 4, 4 ],
    num_head_channels=64, # need to fix for flash-attn
    use_spatial_transformer=True,
    use_linear_in_transformer=True,
    transformer_depth=1,
    context_dim=1024,
    legacy=False,
    dims=3,
    timescale=2)
    model.cuda()
    context=th.randn(2,77,1024).cuda()
    output=model(input,timesteps=input_emd,context=context)
    print(output.shape)
# if __name__=='__main__':
#     input=th.randn(2,320,8,64,64)
#     t_emb=th.randn(2,1024)
#     net=ResBlock(
#         channels=320,
#         emb_channels=1024,
#         dropout=0.5,
#         out_channels=320,
#         use_conv=True,
#         use_scale_shift_norm=True,
#         dims=3,
#         use_checkpoint=False,
#         up=False,
#         down=False,)
#     output=net(input,t_emb)
#     print(output.shape)