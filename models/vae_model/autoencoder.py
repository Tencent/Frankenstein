import enum
import torch
from torch import nn

from einops import reduce

from typing import List, TypeVar
Tensor = TypeVar("torch.tensor")

from models.vae_model.archs.attention import SpatialTransformer, checkpoint, SpatialTransformer_
from models.vae_model.archs.fpn import FPN_down_g, FPN_up_g, GroupConv
import numpy as np




def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels, group_layer_num=32):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(group_layer_num, channels)


class ResBlock(nn.Module):
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
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        dropout,
        out_channels=None,
        use_conv=False,
        dims=2,
        use_checkpoint=False,
        group_layer_num_in=32,
        group_layer_num_out=32,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels if out_channels else channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint

        self.in_layers = nn.Sequential(
            normalization(channels, group_layer_num_in),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels, group_layer_num_out),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
             self._forward, [x], self.parameters(), self.use_checkpoint
        )

    def _forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        h = self.skip_connection(x) + h
        return h


class GroupConvTranspose(nn.Module):
    def __init__(self, in_channels,out_channels, kernel_size, stride=1,padding=0,output_padding=0) -> None:
        super(GroupConvTranspose, self).__init__()
        self.conv = nn.ConvTranspose2d(3*in_channels, 3*out_channels, kernel_size, stride, padding,output_padding,groups=3)
    def forward(self, data: Tensor, **kwargs) -> Tensor:
        data = torch.concat(torch.chunk(data,3,dim=-1),dim=1)
        data = self.conv(data)
        data = torch.concat(torch.chunk(data,3,dim=1),dim=-1)
        return data


class ResBlock_g(nn.Module):
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
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        dropout,
        out_channels=None,
        use_conv=False,
        dims=2,
        use_checkpoint=False,
        group_layer_num_in=1,
        group_layer_num_out=1,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels if out_channels else channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.in_layers = nn.Sequential(
            normalization(channels, group_layer_num_in),
            nn.SiLU(),
            GroupConv(channels, self.out_channels, 3, padding=1)
        )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels, group_layer_num_out),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                GroupConv(self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = GroupConv(channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = GroupConv(channels, self.out_channels,1)
    def forward(self, x):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
             self._forward, [x], self.parameters(), self.use_checkpoint
        )

    def _forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        h = self.skip_connection(x) + h
        return h



# add feature pyramid fusion module, to save more detailed low-level information
class BetaVAERolloutTransformer_room_v1(nn.Module):
    def __init__(self, vae_config) -> None:
        super(BetaVAERolloutTransformer_room_v1, self).__init__()
        print("vae type: BetaVAERolloutTransformer_room_v1")

        kl_std = vae_config.get("kl_std", 0.25)
        kl_weight = vae_config.get("kl_weight", 0.001)
        plane_shape = vae_config.get("plane_shape", [3, 32, 256, 256])
        z_shape = vae_config.get("z_shape", [4, 64, 64])
        num_heads = vae_config.get("num_heads", 16)
        transform_depth = vae_config.get("transform_depth", 1)

        self.plane_dim = len(plane_shape) + 1
        self.plane_shape = plane_shape
        self.z_shape = z_shape

        self.kl_std = kl_std
        self.kl_weight = kl_weight

        # hidden_dims = [512, 512, 1024, 1024, 1024, 1024, 1024, 2 * self.z_shape[0]]
        hidden_dims = [256, 256, 512, 512, 512, 512, 512, 2 * self.z_shape[0]] ###
        # feature size:  64,  32,  16,   8,    4,    8,   16,       32
        # feature_size = [64, 32, 16, 8, 4, 8, 16, 32]
        feature_size = [80, 40, 20, 10, 5, 10, 20, 40] ###
        #

        # hidden_dims_decoder = [1024, 1024, 1024, 1024, 1024, 512, 512]
        # feature size:  16,    8,   4,   8,    16,  32,  64

        self.in_layer = nn.Sequential(ResBlock_g(
            32,
            dropout=0,
            out_channels=128,
            use_conv=True,
            dims=2,
            use_checkpoint=False,
            group_layer_num_in=1
        ),
            nn.BatchNorm2d(128),
            nn.SiLU())

        #
        # self.spatial_modulation = nn.Linear(128*3, 128*3)

        # Build Encoder
        self.encoders_down = nn.ModuleList()
        in_channels = 128
        for i, h_dim in enumerate(hidden_dims[:1]):
            stride = 2
            modules = []
            modules.append(
                nn.Sequential(
                    # nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                    GroupConv(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.SiLU(),
                    ResBlock_g(
                        h_dim,
                        dropout=0,
                        out_channels=h_dim,
                        use_conv=True,
                        dims=2,
                        use_checkpoint=False,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.SiLU()),
            )
            in_channels = h_dim
            self.encoders_down.append(nn.Sequential(*modules))

        for i, h_dim in enumerate(hidden_dims[1:5]):
            dim_head = h_dim // num_heads
            self.encoders_down.append(nn.Sequential(
                GroupConv(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.SiLU(),
                SpatialTransformer_(h_dim,
                                   num_heads,
                                   dim_head,
                                   depth=transform_depth,
                                   context_dim=h_dim,
                                   disable_self_attn=False,
                                   use_linear=True,
                                   attn_type="linear",
                                   use_checkpoint=True,
                                   layer=feature_size[i + 1]
                                   ),
                nn.BatchNorm2d(h_dim),
                nn.SiLU()
            ))
            in_channels = h_dim

        # self.encoder_fpn = FPN_down([512, 512, 1024, 1024], [512, 1024, 1024])
        # self.encoder_fpn = FPN_down_g([512, 512, 1024, 1024], [512, 1024, 1024])
        self.encoder_fpn = FPN_down_g([256, 256, 512, 512], [256, 512, 512]) ###
        self.encoders_up = nn.ModuleList()
        for i, h_dim in enumerate(hidden_dims[5:]):
            modules = []
            if i > 0:
                in_channels = in_channels * 2
            dim_head = h_dim // num_heads
            modules.append(nn.Sequential(
                                         GroupConvTranspose(in_channels,
                                                            h_dim,
                                                            kernel_size=3,
                                                            stride=2,
                                                            padding=1,
                                                            output_padding=1),
                                         nn.BatchNorm2d(h_dim),
                                         nn.SiLU()))
            if i == 2:
                modules.append(nn.Sequential(ResBlock_g(
                    h_dim,
                    dropout=0,
                    out_channels=2 * z_shape[0],
                    use_conv=True,
                    dims=2,
                    use_checkpoint=False,
                ),
                    nn.BatchNorm2d(2 * z_shape[0]),
                    nn.SiLU()))
                in_channels = z_shape[0]
            else:
                modules.append(nn.Sequential(SpatialTransformer_(h_dim,
                                                                num_heads,
                                                                dim_head,
                                                                depth=transform_depth,
                                                                context_dim=h_dim,
                                                                disable_self_attn=False,
                                                                use_linear=True,
                                                                attn_type="linear",
                                                                use_checkpoint=True,
                                                                layer=feature_size[i + 5]
                                                                ),
                                             nn.BatchNorm2d(h_dim),
                                             nn.SiLU()))
                in_channels = h_dim
            self.encoders_up.append(nn.Sequential(*modules))

        ## build decoder
        # hidden_dims_decoder = [1024, 1024, 1024, 1024, 1024, 512, 512]
        hidden_dims_decoder = [512, 512, 512, 512, 512, 256, 256] ###
        # feature size:  16,    8,   4,   8,    16,  32,  64

        # feature_size_decoder = [16, 8, 4, 8, 16, 32, 64]
        feature_size_decoder = [20, 10, 5, 10, 20, 40, 80] ###

        self.decoder_in_layer = nn.Sequential(ResBlock_g(
            self.z_shape[0],
            dropout=0,
            out_channels=256, ### 512
            use_conv=True,
            dims=2,
            use_checkpoint=False,
            group_layer_num_in=1
        ),
            nn.BatchNorm2d(256), ### 512
            nn.SiLU())

        self.decoders_down = nn.ModuleList()
        in_channels = 256 ### 512
        for i, h_dim in enumerate(hidden_dims_decoder[0:3]):
            dim_head = h_dim // num_heads
            stride = 2
            self.decoders_down.append(nn.Sequential(
                # nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                GroupConv(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.SiLU(),
                SpatialTransformer_(h_dim,
                                   num_heads,
                                   dim_head,
                                   depth=transform_depth,
                                   context_dim=h_dim,
                                   disable_self_attn=False,
                                   use_linear=True,
                                   attn_type="linear",
                                   use_checkpoint=True,
                                   layer=feature_size_decoder[i]
                                   ),
                nn.BatchNorm2d(h_dim),
                nn.SiLU()
            ))
            in_channels = h_dim

        # self.decoder_fpn = FPN_up([1024, 1024, 1024, 512], [1024, 1024, 512])
        # self.decoder_fpn = FPN_up_g([1024, 1024, 1024, 512], [1024, 1024, 512])
        self.decoder_fpn = FPN_up_g([512, 512, 512, 256], [512, 512, 256]) ###
        self.decoders_up = nn.ModuleList()
        for i, h_dim in enumerate(hidden_dims_decoder[3:]):
            modules = []
            if i > 0 and i < 4:
                in_channels = in_channels * 2
            dim_head = h_dim // num_heads
            modules.append(nn.Sequential(
                                        GroupConvTranspose( in_channels,
                                                            h_dim,
                                                            kernel_size=3,
                                                            stride=2,
                                                            padding=1,
                                                            output_padding=1),
                                         nn.BatchNorm2d(h_dim),
                                         nn.SiLU()))
            if i < 4:
                modules.append(nn.Sequential(SpatialTransformer_(h_dim,
                                                                num_heads,
                                                                dim_head,
                                                                depth=transform_depth,
                                                                context_dim=h_dim,
                                                                disable_self_attn=False,
                                                                use_linear=True,
                                                                attn_type="linear",
                                                                use_checkpoint=True,
                                                                layer=feature_size_decoder[i + 3]
                                                                ),
                                             nn.BatchNorm2d(h_dim),
                                             nn.SiLU()))
                in_channels = h_dim
            else:
                modules.append(nn.Sequential(ResBlock_g(
                    h_dim,
                    dropout=0,
                    out_channels=h_dim,
                    use_conv=True,
                    dims=2,
                    use_checkpoint=False,
                ),
                    nn.BatchNorm2d(h_dim),
                    nn.SiLU()))
                in_channels = h_dim
            self.decoders_up.append(nn.Sequential(*modules))

        self.decoders_up.append(nn.Sequential(
            GroupConvTranspose(in_channels,
                               in_channels,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            ResBlock_g(
                in_channels,
                dropout=0,
                out_channels=self.plane_shape[1],
                use_conv=True,
                dims=2,
                use_checkpoint=False,
            ),
            nn.BatchNorm2d(self.plane_shape[1]),
            nn.Tanh()))

    def encode(self, enc_input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :enc_input: (Tensor) Input tensor to encoder [B x D x resolution x resolution]
        :return: (Tensor) List of latent codes
        """
        result = enc_input
        if self.plane_dim == 5:
            result = torch.concat(torch.chunk(result,3,dim=1),dim=-1).squeeze(1)
        elif self.plane_dim == 4:
            result = torch.concat(torch.chunk(result,3,dim=1),dim=-1)
        
        feature = self.in_layer(result)

        # hidden_dims = [512, 512, 1024, 1024, 1024, 1024, 1024]
        # #feature size:  64,  32,  16,   8,    4,    8,   16

        features_down = []
        for i, module in enumerate(self.encoders_down):
            feature = module(feature)
            # print("3:", i, feature.shape)
            if i in [0, 1, 2, 3]:
                features_down.append(feature)
        
        features_down = self.encoder_fpn(features_down)

        feature = self.encoders_up[0](feature)
        feature = torch.cat([feature, features_down[-1]], dim=1)
        feature = self.encoders_up[1](feature)
        feature = torch.cat([feature, features_down[-2]], dim=1)
        feature = self.encoders_up[2](feature)

        encode_channel = self.z_shape[0]
        log_var = feature[:, encode_channel:, ...]
        mu = feature[:, :encode_channel, ...]

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        '''
        z
        '''

        # hidden_dims_decoder = [1024, 1024, 1024, 1024, 1024, 512, 512]
        #         #feature size:  16,    8,   4,   8,    16,  32,  64
        x = self.decoder_in_layer(z)
        feature_down = [x]
        for i, module in enumerate(self.decoders_down):
            x = module(x)
            feature_down.append(x)

        feature_down = self.decoder_fpn(feature_down[::-1])

        for i, module in enumerate(self.decoders_up):
            if i in [1, 2, 3]:
                x = torch.cat([x, feature_down[-i]], dim=1)
                x = module(x)
            else:
                x = module(x)

        if self.plane_dim == 5:
            plane_w = self.plane_shape[-1]
            x = torch.concat([x[..., 0:plane_w].unsqueeze(1),
                              x[..., plane_w: plane_w*2].unsqueeze(1),
                              x[..., plane_w*2: plane_w*3].unsqueeze(1),], dim=1)
        elif self.plane_dim == 4:
            plane_w = self.plane_shape[-1]
            x = torch.concat([x[..., 0: plane_w],
                              x[..., plane_w: plane_w*2],
                              x[..., plane_w*2: plane_w*3],], dim=1)
        return x


    def forward(self, data: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(data)
        z = self.reparameterize(mu, log_var)
        result = self.decode(z)

        return [result, data, mu, log_var, z]

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough to compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    # only using VAE loss
    def loss_function(self,
                      *args) -> dict:
        mu = args[2]
        log_var = args[3]
        # print("recon, data shape: ", recons.shape, data.shape)
        # recons_loss = F.mse_loss(recons, data)

        # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        if self.kl_std == 'zero_mean':
            latent = self.reparameterize(mu, log_var)
            # print("latent shape: ", latent.shape) # (B, dim)
            l2_size_loss = torch.sum(torch.norm(latent, dim=-1))
            kl_loss = l2_size_loss / latent.shape[0]

        else:
            std = torch.exp(torch.clamp(0.5 * log_var, max=10)) + 1e-6
            gt_dist = torch.distributions.normal.Normal(torch.zeros_like(mu), torch.ones_like(std) * self.kl_std)
            sampled_dist = torch.distributions.normal.Normal(mu, std)
            # gt_dist = normal_dist.sample(log_var.shape)
            # print("gt dist shape: ", gt_dist.shape)

            kl = torch.distributions.kl.kl_divergence(sampled_dist, gt_dist)  # reversed KL
            kl_loss = reduce(kl, 'b ... -> b (...)', 'mean').mean()

        return self.kl_weight * kl_loss

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

    def get_latent(self, x):
        '''
        given input x, return the latent code
        x:  [B x C x H x W]
        return: [B x latent_dim]
        '''
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return z


    def sample(self,
               num_samples: int,
               **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """

        # # return samples
        # z_rollout_shape = [self.z_shape[0], self.z_shape[1], self.z_shape[2] * 3]
        # eps = torch.randn(num_samples, *(z_rollout_shape)).cuda()
        # z = eps * self.kl_std
        # samples = self.decode(z)
        # return samples, z

        z_rollout_shape = [self.z_shape[0], self.z_shape[1], self.z_shape[2] * 3]
        gt_dist = torch.distributions.normal.Normal(torch.zeros(num_samples, *(z_rollout_shape)),
                                                    torch.ones(num_samples, *(z_rollout_shape)) * self.kl_std)

        z = gt_dist.sample().cuda()
        samples = self.decode(z)
        return samples, z


class BetaVAERolloutTransformer_vroid_128(nn.Module):
    def __init__(self, vae_config) -> None:
        super(BetaVAERolloutTransformer_vroid_128, self).__init__()
        print("vae type: BetaVAERolloutTransformer_vroid_128")

        kl_std = vae_config.get("kl_std", 0.25)
        kl_weight = vae_config.get("kl_weight", 0.001)
        plane_shape = vae_config.get("plane_shape", [3, 32, 128, 128])
        z_shape = vae_config.get("z_shape", [256, 32, 32])
        num_heads = vae_config.get("num_heads", 16)
        transform_depth = vae_config.get("transform_depth", 1)


        self.plane_dim = len(plane_shape) + 1
        self.plane_shape = plane_shape
        self.z_shape = z_shape
        
        self.kl_std = kl_std
        self.kl_weight = kl_weight

        hidden_dims = [512, 512, 1024, 1024, 1024, 1024, 1024, 2*self.z_shape[0]]
        #feature size:  64,  32,  16,   8,    4,    8,   16,       32

        

        hidden_dims_decoder = [1024, 1024, 1024, 1024, 1024, 512, 512]
                #feature size:  16,    8,   4,   8,    16,  32,  64


        self.in_layer = nn.Sequential(ResBlock(
                            32,
                            dropout=0,
                            out_channels=128,
                            use_conv=True,
                            dims=2,
                            use_checkpoint=False,
                            group_layer_num_in=1
                        ),
                        nn.BatchNorm2d(128),
                        nn.SiLU())
        # Build Encoder
        self.encoders_down = nn.ModuleList()
        in_channels = 128
        for i, h_dim in enumerate(hidden_dims[:2]):
            stride = 2
            modules = []
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.SiLU(),
                    ResBlock(
                        h_dim,
                        dropout=0,
                        out_channels=h_dim,
                        use_conv=True,
                        dims=2,
                        use_checkpoint=False,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.SiLU()),
            )
            in_channels = h_dim
            self.encoders_down.append(nn.Sequential(*modules))
        
        for i, h_dim in enumerate(hidden_dims[2:5]):
            dim_head = h_dim // num_heads
            self.encoders_down.append(nn.Sequential(
                                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                                    nn.BatchNorm2d(h_dim),
                                    nn.SiLU(),
                                    SpatialTransformer(h_dim,
                                                        num_heads,
                                                        dim_head,
                                                        depth=transform_depth,
                                                        context_dim=h_dim,
                                                        disable_self_attn=False,
                                                        use_linear=True,
                                                        attn_type="softmax-xformers",
                                                        use_checkpoint=True,
                                                        ),
                                    nn.BatchNorm2d(h_dim),
                                    nn.SiLU()
                                    ))
            in_channels = h_dim


        self.encoders_up = nn.ModuleList()
        for i, h_dim in enumerate(hidden_dims[5:]):
            modules = []
            if i > 0:
                in_channels = in_channels * 2
            dim_head = h_dim // num_heads
            modules.append(nn.Sequential(nn.ConvTranspose2d(in_channels,
                                                        h_dim,
                                                        kernel_size=3,
                                                        stride = 2,
                                                        padding=1,
                                                        output_padding=1),
                            nn.BatchNorm2d(h_dim),
                            nn.SiLU()))
            if i == 2:
                modules.append(nn.Sequential(ResBlock(
                                        h_dim,
                                        dropout=0,
                                        out_channels=2*z_shape[0],
                                        use_conv=True,
                                        dims=2,
                                        use_checkpoint=False,
                                    ),
                                nn.BatchNorm2d(2*z_shape[0]),
                                nn.SiLU()))
                in_channels = z_shape[0]
            else:
                modules.append(nn.Sequential(SpatialTransformer(h_dim,
                                                num_heads,
                                                dim_head,
                                                depth=transform_depth,
                                                context_dim=h_dim,
                                                disable_self_attn=False,
                                                use_linear=True,
                                                attn_type="softmax-xformers",
                                                use_checkpoint=True,
                                                ),
                                nn.BatchNorm2d(h_dim),
                                nn.SiLU()))
                in_channels = h_dim
            self.encoders_up.append(nn.Sequential(*modules))


        ## build decoder
        hidden_dims_decoder = [1024, 1024, 1024, 1024, 1024, 512, 512]
                #feature size:  16,    8,   4,   8,    16,  32,  64

        self.decoder_in_layer = nn.Sequential(ResBlock(
                            self.z_shape[0],
                            dropout=0,
                            out_channels=512,
                            use_conv=True,
                            dims=2,
                            use_checkpoint=False,
                            group_layer_num_in=1
                        ),
                        nn.BatchNorm2d(512),
                        nn.SiLU())
        
        self.decoders_down = nn.ModuleList()
        in_channels = 512
        for i, h_dim in enumerate(hidden_dims_decoder[0:3]):
            dim_head = h_dim // num_heads
            stride = 2
            self.decoders_down.append(nn.Sequential(
                                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                                    nn.BatchNorm2d(h_dim),
                                    nn.SiLU(),
                                    SpatialTransformer(h_dim,
                                                        num_heads,
                                                        dim_head,
                                                        depth=transform_depth,
                                                        context_dim=h_dim,
                                                        disable_self_attn=False,
                                                        use_linear=True,
                                                        attn_type="softmax-xformers",
                                                        use_checkpoint=True,
                                                        ),
                                    nn.BatchNorm2d(h_dim),
                                    nn.SiLU()
                                    ))
            in_channels = h_dim

        self.decoders_up = nn.ModuleList()
        for i, h_dim in enumerate(hidden_dims_decoder[3:]):
            modules = []
            if i > 0 and i < 4:
                in_channels = in_channels * 2
            dim_head = h_dim // num_heads
            modules.append(nn.Sequential(nn.ConvTranspose2d(in_channels,
                                                        h_dim,
                                                        kernel_size=3,
                                                        stride = 2,
                                                        padding=1,
                                                        output_padding=1),
                            nn.BatchNorm2d(h_dim),
                            nn.SiLU()))
            if i < 3:
                modules.append(nn.Sequential(SpatialTransformer(h_dim,
                                                num_heads,
                                                dim_head,
                                                depth=transform_depth,
                                                context_dim=h_dim,
                                                disable_self_attn=False,
                                                use_linear=True,
                                                attn_type="softmax-xformers",
                                                use_checkpoint=True,
                                                ),
                                nn.BatchNorm2d(h_dim),
                                nn.SiLU()))
                in_channels = h_dim
            else:
                modules.append(nn.Sequential(ResBlock(
                                        h_dim,
                                        dropout=0,
                                        out_channels=h_dim,
                                        use_conv=True,
                                        dims=2,
                                        use_checkpoint=False,
                                    ),
                                nn.BatchNorm2d(h_dim),
                                nn.SiLU()))
                in_channels = h_dim
            self.decoders_up.append(nn.Sequential(*modules))

        self.decoders_up.append(nn.Sequential(
                                    nn.ConvTranspose2d(in_channels,
                                                        in_channels,
                                                        kernel_size=3,
                                                        stride = 2,
                                                        padding=1,
                                                        output_padding=1),
                                    nn.BatchNorm2d(in_channels),
                                    nn.SiLU(),
                                    ResBlock(
                                        in_channels,
                                        dropout=0,
                                        out_channels=self.plane_shape[1],
                                        use_conv=True,
                                        dims=2,
                                        use_checkpoint=False,
                                    ),
                                    nn.BatchNorm2d(self.plane_shape[1]),
                                    nn.Tanh()))


    def encode(self, enc_input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :enc_input: (Tensor) Input tensor to encoder [B x D x resolution x resolution]
        :return: (Tensor) List of latent codes
        """
        result = enc_input  # [B,3,32,256,256]
        if self.plane_dim == 5:
            plane_list = []
            for i in range(self.plane_shape[0]):
                plane_list.append(result[:, i, :, :, :])
            result = torch.concat(plane_list, dim=-1)
        elif self.plane_dim == 4:
            plane_channel = result.shape[1] // 3
            result = torch.concat([result[:, 0:plane_channel ,...],
                                result[:, plane_channel:plane_channel*2 ,...],
                                result[:, plane_channel*2:plane_channel*3 ,...]], dim=-1)
        
        feature = self.in_layer(result)  # [B,32,256,768] -> [B,128,256,768]

        # hidden_dims = [512, 512, 1024, 1024, 1024, 1024, 1024]
        # #feature size:  64,  32,  16,   8,    4,    8,   16
        # [B,128,256,768] -> 0 [B,512,128,384] -> 1 [B,512,64,192] -> 2 [B,1024,32,96] -> 3 [B,1024,16,48] -> [B,1024,8,24]
        features_down = []
        for i, module in enumerate(self.encoders_down):
            feature = module(feature)
            if i in [2, 3]:
                features_down.append(feature)

        feature = self.encoders_up[0](feature)  # [B,1024,16,48]
        feature = torch.cat([feature, features_down[-1]], dim=1)  # [B,2048,16,48]
        feature = self.encoders_up[1](feature)  # [B,1024,32,96]
        feature = torch.cat([feature, features_down[-2]], dim=1)  # [B,2048,32,96]
        feature = self.encoders_up[2](feature)  # [B,512,64,192]

        encode_channel = self.z_shape[0]
        mu = feature[:, :encode_channel, ...]
        log_var = feature[:, encode_channel:, ...]
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        '''
        z
        '''

        # hidden_dims_decoder = [1024, 1024, 1024, 1024, 1024, 512, 512]
        #         #feature size:  16,    8,   4,   8,    16,  32,  64
        x = self.decoder_in_layer(z)
        feature_down = [x]
        for i, module in enumerate(self.decoders_down):
            x = module(x)
            if i in [0, 1]:
                feature_down.append(x)

        for i, module in enumerate(self.decoders_up):
            if i in [1, 2, 3]:
                x = torch.cat([x, feature_down[-i]], dim=1)
                x = module(x)
            else:
                x = module(x)

        plane_w = self.plane_shape[-1]
        if self.plane_dim == 5:
            x = torch.concat([x[..., 0:plane_w].unsqueeze(1),
                              x[..., plane_w: plane_w*2].unsqueeze(1),
                              x[..., plane_w*2: plane_w*3].unsqueeze(1),], dim=1)
        elif self.plane_dim == 4:
            x = torch.concat([x[..., 0: plane_w],
                              x[..., plane_w: plane_w*2],
                              x[..., plane_w*2: plane_w*3],], dim=1)
        return x

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough to compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, data: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(data)
        z = self.reparameterize(mu, log_var)
        result = self.decode(z)

        return  [result, data, mu, log_var, z]

    # only using VAE loss
    def loss_function(self,
                      *args) -> dict:
        mu = args[2]
        log_var = args[3]
      
        if self.kl_std == 'zero_mean':
            latent = self.reparameterize(mu, log_var) 
            #print("latent shape: ", latent.shape) # (B, dim)
            l2_size_loss = torch.sum(torch.norm(latent, dim=-1))
            kl_loss = l2_size_loss / latent.shape[0]

        else:
            std = torch.exp(torch.clamp(0.5 * log_var, max=10)) + 1e-6
            gt_dist = torch.distributions.normal.Normal( torch.zeros_like(mu), torch.ones_like(std)*self.kl_std )
            sampled_dist = torch.distributions.normal.Normal( mu, std )
            #gt_dist = normal_dist.sample(log_var.shape)
            #print("gt dist shape: ", gt_dist.shape)

            kl = torch.distributions.kl.kl_divergence(sampled_dist, gt_dist) # reversed KL
            kl_loss = reduce(kl, 'b ... -> b (...)', 'mean').mean()

        return self.kl_weight * kl_loss

    def sample(self,
               num_samples:int,
                **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """

        raise NotImplementedError

        # # return samples
        # z_rollout_shape = [self.z_shape[0], self.z_shape[1], self.z_shape[2] * 3]
        # eps = torch.randn(num_samples, *(z_rollout_shape)).cuda()
        # z = eps * self.kl_std
        # samples = self.decode(z)
        # return samples, z

        z_rollout_shape = [self.z_shape[0], self.z_shape[1], self.z_shape[2] * 3]
        gt_dist = torch.distributions.normal.Normal(torch.zeros(num_samples, *(z_rollout_shape)), 
                                                    torch.ones(num_samples, *(z_rollout_shape)) * self.kl_std)

        z = gt_dist.sample().cuda()
        samples = self.decode(z)
        return samples, z


    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

    def get_latent(self, x):
        '''
        given input x, return the latent code
        x:  [B x C x H x W]
        return: [B x latent_dim]
        '''
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return z





if __name__ == "__main__":
    vae_config = {"kl_std": 0.25,
                "kl_weight": 0.001,
                "plane_shape": [3, 32, 160, 160],
                # "z_shape": [32, 64, 64],
                "z_shape": [4, 40, 40],
                "num_heads": 16,
                "transform_depth": 1}

    vae_model = BetaVAERolloutTransformer_room_v1(vae_config).cuda()
    

    input_tensor = torch.randn(4, 3, 32, 160, 160).cuda()
    out = vae_model(input_tensor)
    input_tensor = torch.randn(4, 3, 32, 160, 160).cuda()
    import time 
    t1 = time.time()
    out = vae_model(input_tensor)
    print("time consume:", time.time()-t1)
    loss = vae_model.loss_function(*out)
    print("loss: {}".format(loss))
    print("z shape: {}".format(out[-1].shape))
    print("reconstruct shape: {}".format(out[0].shape))
    # samples = vae_model.sample(2)
    # print("samples shape: {}".format(samples[0].shape))
    # import pdb;pdb.set_trace()
