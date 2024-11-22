import torch.nn as nn
import torch.nn.init as init
from models.vae_model import * 


def get_decoder_model(specs):
    decoder_config = specs["decoder_config"]

    if decoder_config["decoder_type"] == "sdfvroid":
        decoder_model = SdfSemModel(decoder_config["config_json"])
        if "train_params" in specs.keys() and specs["train_params"] == "vae":
            decoder_model.eval()
            for p in decoder_model.parameters():
                p.requires_grad = False
    
    else:
        print("did not recogonize decoder name")
        exit(1)

    return decoder_model


def get_vae_model(specs):
    vae_config = specs["vae_config"]
    vae_type = vae_config["vae_type"]
    kl_std = vae_config.get("kl_std", 0.001)
    kl_weight = vae_config.get("kl_weight", 1.0)
    plane_shape = vae_config["plane_shape"]
    z_shape = vae_config["z_shape"]
    if vae_type == "BetaVAERolloutTransformer_room_v1":
        vae_model = BetaVAERolloutTransformer_room_v1(vae_config)
    elif vae_type == "BetaVAERolloutTransformer_vroid_128":
        vae_model = BetaVAERolloutTransformer_vroid_128(vae_config)
    else:
        raise NotImplementedError

    vae_model.apply(weight_init)
    return vae_model


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            # init.normal_(m.bias.data)
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            # init.normal_(m.bias.data)
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            # init.normal_(m.bias.data)
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            # init.normal_(m.bias.data)
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            # init.normal_(m.bias.data)
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            # init.normal_(m.bias.data)
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            # init.normal_(m.bias.data)
            init.constant_(m.bias.data, 0)