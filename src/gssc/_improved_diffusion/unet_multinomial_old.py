from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from autoencoder.simpleAE import SimpleAE_v2, Encoder
from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    SiLU,
    conv_nd,
    depthwise_conv_nd,
    linear,
    avg_pool_nd,
    max_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    lidar_embedding,
    checkpoint,
)

from spconv.pytorch.modules import SparseModule, SparseSequential
from spconv.pytorch.conv import SubMConv3d, SparseConv3d
from spconv.pytorch.core import SparseConvTensor

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb, lidar_emb=None, bev_emb=None, uncond_flag=0):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, lidar_emb=None, bev_emb=None, uncond_flag=0):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb, lidar_emb, bev_emb, uncond_flag)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, channels, channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2] * 2, x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (2, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(dims, stride)

    def forward(self, x):
        #print(self.dims, x.shape, self.op(x).shape)
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
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
    """

    def __init__(
        self,
        channels,
        emb_channels,
        model_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        mult=1,
        use_lidar=False,
        use_depthwise_conv=False,
        sparse_conv=False,
        use_bev=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.sparse_conv = sparse_conv
        self.use_bev = use_bev

        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1) if not use_depthwise_conv else depthwise_conv_nd(dims, channels, self.out_channels),
        )
        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        if use_bev:
            self.bev_emb_layers = nn.Sequential(
                max_pool_nd(dims, mult),
                conv_nd(dims, model_channels, self.out_channels, 3, padding=1) if not use_depthwise_conv else depthwise_conv_nd(dims, model_channels, self.out_channels),
                SiLU(),
                normalization(self.out_channels),
        )
        if use_lidar:
            if sparse_conv:
                self.lidar_emb_layers = SparseSequential(
                    SparseConv3d(model_channels, self.out_channels, 3, stride=mult, padding=1, bias=False, indice_key=None),
                    SiLU(),
                    SubMConv3d(self.out_channels, self.out_channels, 3, padding=1, bias=False, indice_key=None),
                    SiLU(),
                    normalization(self.out_channels),
                )
            else:
                self.lidar_emb_layers = nn.Sequential(
                    max_pool_nd(dims, mult),
                    conv_nd(dims, model_channels, self.out_channels, 3, padding=1) if not use_depthwise_conv else depthwise_conv_nd(dims, model_channels, self.out_channels),
                    SiLU(),
                    normalization(self.out_channels),
                )
        #print(channels, self.out_channels)
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1) if not use_depthwise_conv else depthwise_conv_nd(dims, self.out_channels, self.out_channels),
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1) if not use_depthwise_conv else depthwise_conv_nd(dims, channels, self.out_channels),
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1) 

    def forward(self, x, emb, lidar_emb=None, bev_emb=None, uncond_flag=0):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb, lidar_emb, bev_emb, uncond_flag), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb, lidar_emb=None, bev_emb=None, uncond_flag=0):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        if lidar_emb != None:
            lidar_emb = self.lidar_emb_layers(lidar_emb)
            if self.sparse_conv:
                lidar_emb = lidar_emb.dense()
            if uncond_flag == 1:
                lidar_emb = lidar_emb * 0
            h = h + lidar_emb
        if bev_emb != None:
            bev_emb = self.bev_emb_layers(bev_emb)
            h = h + bev_emb
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, channels, num_heads=1, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint

        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = th.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        return th.einsum("bts,bcs->bct", weight, v)

    @staticmethod
    def count_flops(model, _x, y):
        """
        A counter for the `thop` package to count the operations in an
        attention operation.

        Meant to be used like:

            macs, params = thop.profile(
                model,
                inputs=(inputs, timestamps),
                custom_ops={QKVAttention: QKVAttention.count_flops},
            )

        """
        b, c, *spatial = y[0].shape
        num_spatial = int(np.prod(spatial))
        # We perform two matmuls with the same number of ops.
        # The first computes the weight matrix, the second computes
        # the combination of the value vectors.
        matmul_ops = 2 * b * (num_spatial ** 2) * c
        model.total_ops += th.DoubleTensor([matmul_ops])


class UNetModel(nn.Module):
    """
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
    """

    def __init__(
        self,
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
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        encoder_config=None,
        use_lidar=False,
        binary=False,
        use_depthwise_conv=False,
        sparse_conv=False,
        use_attn=False,
        downsample_resolution=1,
        analog_bits=False,
        one_hot=False,
        use_bev=False,
        use_bev_vol=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample
        self.binary = binary
        self.sparse_conv = sparse_conv
        self.use_bev = use_bev
        self.use_bev_vol = use_bev_vol

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        
        self.class_embedding = nn.Embedding(num_classes, model_channels)
        if use_lidar:
            #self.lidar_pooling = nn.AvgPool3d(4)
            if sparse_conv:
                self.lidar_embedding = SparseSequential(
                    SubMConv3d(model_channels, model_channels, 3, padding=1, bias=False, indice_key=None),
                    SiLU(),
                    SubMConv3d(model_channels, model_channels, 3, padding=1, bias=False, indice_key=None),
                    SiLU(),
                    normalization(model_channels),
                )

            else:
                self.lidar_embedding = nn.Sequential(
                    conv_nd(dims, model_channels, model_channels, 1, padding=0),
                    SiLU(),
                    conv_nd(dims, model_channels, model_channels, 1, padding=0),
                    normalization(model_channels),
                )

        if use_bev:
            self.bev_height_embed = nn.Embedding(12,8)
        
            self.bev_embed_2D = nn.Sequential(
                conv_nd(2, model_channels, model_channels, 3, padding=1),
                SiLU(),     
                conv_nd(2, model_channels, model_channels, 3, padding=1),
                normalization(model_channels),
            )
                
            self.bev_embed_3D = nn.Sequential(
                conv_nd(3, model_channels, model_channels, 3, padding=1),
                SiLU(),     
                conv_nd(3, model_channels, model_channels, 3, padding=1),
                normalization(model_channels),
            )
            
        if use_bev_vol:
            
            self.bev_embed = SparseSequential(
                conv_nd(3, model_channels, model_channels, 3, padding=1, stride=downsample_resolution),
                SiLU(), 
                conv_nd(3, model_channels, model_channels, 3, padding=1),
                SiLU(),
                normalization(model_channels),
            )
        use_bev = use_bev or use_bev_vol
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, model_channels, model_channels, 3, padding=1)
                )
            ]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        model_channels,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        mult=2**level,
                        use_lidar=use_lidar,
                        use_depthwise_conv=use_depthwise_conv,
                        sparse_conv=sparse_conv,
                        use_bev=use_bev,
                    )
                ]
                ch = mult * model_channels
                if level in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch, use_checkpoint=use_checkpoint, num_heads=num_heads
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims))
                )
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                model_channels,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                mult=2**level,
                use_lidar=use_lidar,
                use_depthwise_conv=use_depthwise_conv,
                sparse_conv=sparse_conv,
                use_bev=use_bev,
            ),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads),
            ResBlock(
                ch,
                time_embed_dim,
                model_channels,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                mult=2**level,
                use_lidar=use_lidar,
                use_depthwise_conv=use_depthwise_conv,
                sparse_conv=sparse_conv,
                use_bev=use_bev,
            ),
        ) if use_attn else TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                model_channels,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                mult=2**level,
                use_lidar=use_lidar,
                use_depthwise_conv=use_depthwise_conv,
                sparse_conv=sparse_conv,
                use_bev=use_bev,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                model_channels,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                mult=2**level,
                use_lidar=use_lidar,
                use_depthwise_conv=use_depthwise_conv,
                sparse_conv=sparse_conv,
                use_bev=use_bev,
            ),
        )
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        model_channels,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        mult=2**(level),
                        use_lidar=use_lidar,
                        use_depthwise_conv=use_depthwise_conv,
                        sparse_conv=sparse_conv,
                        use_bev=use_bev,
                    )
                ]
                ch = model_channels * mult
                if level in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)) if not use_depthwise_conv else zero_module(depthwise_conv_nd(dims, model_channels, out_channels)),
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

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    def forward(self, x, timesteps, x_paths=None, y=None, bev=None, classifier_free=None, condition_scale=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of voxel feature.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N x C x ...] Tensor of input Lidar data, binary.
        :param x_paths: an [N] list of paths of 
        :return: an [N x C x ...] Tensor of outputs.
        """

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        lidar_emb = None
        if y != None:
            if self.sparse_conv:
                #prepare input tensor
                uncond_flag = 0
                if (y == 0).all():
                    uncond_flag = 1
                    z=14
                    y[:,-z:,-z:,-z:] = 1
                h,w,l=y.shape[-3:]
                spatial_shape = [h,w,l]
                batch_size = x.shape[0]
                voxel_coords = th.stack(th.meshgrid(th.arange(0,h,1), th.arange(0,w,1), th.arange(0,l,1))).to(x.device).unsqueeze(0).repeat(batch_size,1,1,1,1)
                voxel_batches = th.ones(batch_size,1,h,w,l, device=x.device) * th.arange(0,batch_size,1,device=x.device).view(-1,1,1,1,1)
                voxel_coords = th.cat([voxel_batches, voxel_coords], dim=1)
                lidar_mask = y != 0
                lidar_voxels = y[lidar_mask].clone()
                lidar_coords = voxel_coords.permute(0,2,3,4,1)[lidar_mask].clone()
                #lidar_features = lidar_voxels.unsqueeze(-1)
                lidar_features = self.class_embedding(lidar_voxels)
                input_sp_lidar = SparseConvTensor(
                    features=lidar_features.clone().detach(),
                    indices=lidar_coords.clone().detach().int(),
                    spatial_shape=spatial_shape,
                    batch_size=batch_size
                )
                lidar_emb = self.lidar_embedding(input_sp_lidar)
                
            else:
                assert False # not finished yet
                lidar_emb = self.lidar_embedding(lidar_embedding(y, self.model_channels))
            #print(y.shape, lidar_emb.shape,emb.shape)

        bev_emb=None
        if self.use_bev:
            bev_index = bev.squeeze(1)
            bev_volume = self.bev_height_embed(bev_index.type(th.long)).permute(0,3,1,2).contiguous()
            bev = self.class_embedding(bev).permute(0,3,1,2).contiguous()
            bev = self.bev_embed_2D(bev)
            bev_volume = bev_volume.unsqueeze(1) * bev.unsqueeze(2)
            
            bev_emb = self.bev_embed_3D(bev_volume)
            
        if self.use_bev_vol:
            bev = self.class_embedding(bev).permute(0,4,1,2,3).contiguous()
            bev_emb = self.bev_embed(bev)

        #if self.num_classes is not None:
        #    assert y.shape == (x.shape[0],)
        #    emb = emb + self.label_emb(y)
        x = x.type(th.float)
        #h = x.type(self.inner_dtype)
        free_mask = (x == 0).type(th.float).unsqueeze(1).expand(-1,self.model_channels,-1,-1,-1)
        h = self.class_embedding(x.type(th.long)).permute(0,4,1,2,3).type(self.inner_dtype).contiguous()
        free_voxels = th.zeros_like(h)
        h = h * (1-free_mask) + free_voxels * free_mask
        for module in self.input_blocks:
            h = module(h, emb, lidar_emb, bev_emb, uncond_flag)
            hs.append(h)
        h = self.middle_block(h, emb, lidar_emb, bev_emb, uncond_flag)
        for module in self.output_blocks:
            h_ = hs.pop()
            cat_in = th.cat([h, h_], dim=1)
            h = module(cat_in, emb, lidar_emb, bev_emb, uncond_flag)
        h = h.type(x.dtype)
        out = self.out(h)
        return out

    def get_feature_vectors(self, x, timesteps, y=None):
        """
        Apply the model and return all of the intermediate tensors.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: a dict with the following keys:
                 - 'down': a list of hidden state tensors from downsampling.
                 - 'middle': the tensor of the output of the lowest-resolution
                             block in the model.
                 - 'up': a list of hidden state tensors from upsampling.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        result = dict(down=[], up=[])
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            result["down"].append(h.type(x.dtype))
        h = self.middle_block(h, emb)
        result["middle"] = h.type(x.dtype)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
            result["up"].append(h.type(x.dtype))
        return result


class SuperResModel(UNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)

    def get_feature_vectors(self, x, timesteps, low_res=None, **kwargs):
        _, new_height, new_width, _ = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().get_feature_vectors(x, timesteps, **kwargs)

