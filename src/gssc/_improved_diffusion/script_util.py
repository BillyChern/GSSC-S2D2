#import ipdb; ipdb.set_trace()
import argparse
import inspect
import torch as th

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .multinomial_diffusion import MultinomialDiffusion
from .unet import SuperResModel, UNetModel
from .unet_factorized import UNetModel as UNetModel_fa
from .unet_old_fullres_baseline import UNetModel as UNetModel_old
from .unet_multinomial import UNetModel as UNetModel_multinomial # error
from .unet_sparse import UNetModel as UNetModel_sparse
from .transformer import PointDiffusionTransformer
from .dist_util import DEBUG

NUM_CLASSES = 1000


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    return dict(
        image_size=64,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions="",
        dropout=0.0,
        learn_sigma=False,
        sigma_small=False,
        class_cond=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        dataset="kitti",
        diffusion_type="gaussian",
    )

def create_model_and_diffusion(
    image_size,
    class_cond,
    dataset,
    diffusion_type,
    learn_sigma,
    sigma_small,
    num_channels,
    num_res_blocks,
    num_heads,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    use_lidar,
    binary,
    use_depthwise_conv,
    sparse_conv,
    factorized_conv,
    old,
    label_diffusion,
    downsample_resolution,
    pos_neg_adjust_thresh,
    analog_bits,
    one_hot,
    use_bev,
    use_bev_vol,
    use_multiclass_bev,
    use_point_encoder,
    recon_x0,
    sparse_unet,
):
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        use_lidar=use_lidar,
        binary=binary,
        use_depthwise_conv=use_depthwise_conv,
        sparse_conv=sparse_conv,
        factorized_conv=factorized_conv,
        old=old,
        label_diffusion=label_diffusion,
        downsample_resolution=downsample_resolution,
        analog_bits=analog_bits,
        one_hot=one_hot,
        diffusion_type=diffusion_type,
        dataset=dataset,
        use_bev=use_bev,
        use_bev_vol=use_bev_vol,
        use_multiclass_bev=use_multiclass_bev,
        use_point_encoder=use_point_encoder,
        sparse_unet=sparse_unet,
    )
    diffusion = create_gaussian_diffusion(
        diffusion_type=diffusion_type,
        dataset=dataset,
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
        pos_neg_adjust_thresh=pos_neg_adjust_thresh,
        recon_x0=recon_x0,
    )
    return model, diffusion

def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
    use_lidar,
    binary,
    use_depthwise_conv,
    sparse_conv,
    factorized_conv,
    old,
    label_diffusion,
    downsample_resolution,
    analog_bits,
    one_hot,
    diffusion_type,
    dataset,
    use_bev,
    use_bev_vol,
    use_multiclass_bev,
    use_point_encoder,
    sparse_unet,
):
    
    if image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
    elif image_size == 1: # for SSC
        channel_mult = (1, 1, 2, 2)
    elif image_size == 2:
        channel_mult = (1, 2, 2, 4)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    if attention_resolutions != "":
        use_attn = True
    else:
        use_attn = False
    #for res in attention_resolutions.split(","):
    #    attention_ds.append(image_size // int(res))
    
    if attention_resolutions != "":
        for level in attention_resolutions.split(","):
            attention_ds.append(int(level))
    
    if label_diffusion:
        in_channel = 19
    elif binary:
        in_channel = 1
    elif analog_bits:
        if dataset == "kitti":
            in_channel = 5
        elif dataset == "carla":
            in_channel = 4
    elif one_hot:
        in_channel = 20
    else:
        in_channel = 64
    out_channel = in_channel
    
    if factorized_conv:
        return UNetModel_fa(
            in_channels=in_channel,
            model_channels=num_channels,
            #out_channels=(3 if not learn_sigma else 6),
            out_channels=out_channel if not learn_sigma else out_channel*2,
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=dropout,
            channel_mult=channel_mult,
            num_classes=(NUM_CLASSES if class_cond else None),
            use_checkpoint=use_checkpoint,
            num_heads=num_heads,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            #encoder_config=encoder_config,
            dims=3,
            use_lidar=use_lidar,
            binary=binary,
            use_depthwise_conv=use_depthwise_conv,
            sparse_conv=sparse_conv,
        )
    elif old:
        return UNetModel_old(
            in_channels=in_channel,
            model_channels=num_channels,
            #out_channels=(3 if not learn_sigma else 6),
            out_channels=out_channel if not learn_sigma else out_channel*2,
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=dropout,
            channel_mult=channel_mult,
            num_classes=(NUM_CLASSES if class_cond else None),
            use_checkpoint=use_checkpoint,
            num_heads=num_heads,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            #encoder_config=encoder_config,
            dims=3,
            use_lidar=use_lidar,
            binary=binary,
            use_depthwise_conv=use_depthwise_conv,
            sparse_conv=sparse_conv,
        )
    elif diffusion_type == "multinomial":
        if dataset == "kitti":
            num_classes = 20
        elif dataset == "carla":
            num_classes = 11
        if sparse_unet:
            return UNetModel_sparse(
                in_channels=in_channel,
                model_channels=num_channels,
                #out_channels=(3 if not learn_sigma else 6),
                out_channels=num_classes if not learn_sigma else num_classes*2,
                num_res_blocks=num_res_blocks,
                attention_resolutions=tuple(attention_ds),
                dropout=dropout,
                channel_mult=channel_mult,
                num_classes=num_classes,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_heads_upsample=num_heads_upsample,
                use_scale_shift_norm=use_scale_shift_norm,
                #encoder_config=encoder_config,
                dims=3,
                use_lidar=use_lidar,
                binary=binary,
                use_depthwise_conv=use_depthwise_conv,
                sparse_conv=sparse_conv,
                use_attn=use_attn,
                downsample_resolution=downsample_resolution,
                analog_bits=analog_bits,
                one_hot=one_hot,
                use_bev=use_bev,
                use_bev_vol=use_bev_vol,
                use_multiclass_bev=use_multiclass_bev,
                use_point_encoder=use_point_encoder,
            )
        else:
            return UNetModel_multinomial(
                in_channels=in_channel,
                model_channels=num_channels,
                #out_channels=(3 if not learn_sigma else 6),
                out_channels=num_classes if not learn_sigma else num_classes*2,
                num_res_blocks=num_res_blocks,
                attention_resolutions=tuple(attention_ds),
                dropout=dropout,
                channel_mult=channel_mult,
                num_classes=num_classes,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_heads_upsample=num_heads_upsample,
                use_scale_shift_norm=use_scale_shift_norm,
                #encoder_config=encoder_config,
                dims=3,
                use_lidar=use_lidar,
                binary=binary,
                use_depthwise_conv=use_depthwise_conv,
                sparse_conv=sparse_conv,
                use_attn=use_attn,
                downsample_resolution=downsample_resolution,
                analog_bits=analog_bits,
                one_hot=one_hot,
                use_bev=use_bev,
                use_bev_vol=use_bev_vol,            
                use_multiclass_bev=use_multiclass_bev,
                use_point_encoder=use_point_encoder,
            )
    else:
        return UNetModel(
            in_channels=in_channel,
            model_channels=num_channels,
            #out_channels=(3 if not learn_sigma else 6),
            out_channels=out_channel if not learn_sigma else out_channel*2,
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=dropout,
            channel_mult=channel_mult,
            num_classes=(NUM_CLASSES if class_cond else None),
            use_checkpoint=use_checkpoint,
            num_heads=num_heads,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            #encoder_config=encoder_config,
            dims=3,
            use_lidar=use_lidar,
            binary=binary,
            use_depthwise_conv=use_depthwise_conv,
            sparse_conv=sparse_conv,
            use_attn=use_attn,
            downsample_resolution=downsample_resolution,
            analog_bits=analog_bits,
            one_hot=one_hot,
            use_bev=use_bev,
            use_bev_vol=use_bev_vol,
            use_multiclass_bev=use_multiclass_bev,
        )

def sr_model_and_diffusion_defaults():
    res = model_and_diffusion_defaults()
    res["large_size"] = 256
    res["small_size"] = 64
    arg_names = inspect.getfullargspec(sr_create_model_and_diffusion)[0]
    for k in res.copy().keys():
        if k not in arg_names:
            del res[k]
    return res


def sr_create_model_and_diffusion(
    large_size,
    small_size,
    class_cond,
    learn_sigma,
    num_channels,
    num_res_blocks,
    num_heads,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
):
    model = sr_create_model(
        large_size,
        small_size,
        num_channels,
        num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def sr_create_model(
    large_size,
    small_size,
    num_channels,
    num_res_blocks,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
):
    _ = small_size  # hack to prevent unused variable

    if large_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif large_size == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported large size: {large_size}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(large_size // int(res))

    return SuperResModel(
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
    )


def create_gaussian_diffusion(
    *,
    diffusion_type="gaussian",
    dataset="kitti",
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
    point_diffusion=False,
    binary=False,
    use_kld=False,
    pos_neg_adjust_thresh=-1,
    recon_x0=False,
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if point_diffusion:
        loss_type = gd.LossType.MSE_CE
    elif use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    if diffusion_type == "gaussian":
        return SpacedDiffusion(
            use_timesteps=space_timesteps(steps, timestep_respacing),
            betas=betas,
            model_mean_type=(
                gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
            ),
            model_var_type=(
                (
                    gd.ModelVarType.FIXED_LARGE
                    if not sigma_small
                    else gd.ModelVarType.FIXED_SMALL
                )
                if not learn_sigma
                else gd.ModelVarType.LEARNED_RANGE
            ),
            loss_type=loss_type,
            rescale_timesteps=rescale_timesteps,
            binary=binary,
            kld=use_kld,
            pos_neg_adjust_thresh=pos_neg_adjust_thresh,
        )
    elif diffusion_type == "multinomial":
        if dataset == 'kitti':
            num_classes = 20
        elif dataset == 'carla':
            num_classes = 11
        return MultinomialDiffusion(
            num_classes=num_classes,
            diffusion_steps=steps,
            recon_loss=False,
            recon_x0=recon_x0,
            multi_criterion=None,
        )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
