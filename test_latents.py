import os, pdb
import numpy as np 
from tqdm import tqdm 
import json
import torch

from models.vae_model import CombinedModel
from models.utils import mesh
from models.dataloader.triplane_stats import unnormalize



def test(args):

    specs = json.load(open(os.path.join(args.exp_dir, "specs_test.json")))

    # load statistics of latent tri-plane
    stats_dir = os.path.join(args.exp_dir, "stats")

    # load models
    if args.resume == "last":
        ckpt = "last.ckpt"
    else:
        ckpt = f"epoch={args.resume}.ckpt"
    resume = os.path.join(args.exp_dir, ckpt)
    print("Load from ", resume)
    model = CombinedModel.load_from_checkpoint(resume, specs=specs, args=args).cuda().eval()

    # load latent tri-plane
    lats = torch.from_numpy(np.load(os.path.join(args.lat_dir, 'lats_sample.npy'))).cuda()
    print(f"Load latents with shape: {lats.shape}")

    # vae-decode the latent tri-plane and extract mesh from sdfs
    for i in tqdm(range(lats.shape[0])):
        savedir = os.path.join(args.lat_dir, f"recon_{i:03d}")
        os.makedirs(savedir, exist_ok=True)

        # vae-decode 
        # e.g. latent tri-plane [1,256,32,96] -> normalized raw tri-plane [1,3,32,128,128]
        out = model.vae_model.decode(lats[i:i+1]) 
        reconstructed_plane_feature = unnormalize(out, stats_dir) # unnormalized raw tri-plane
        
        # extract mesh from sdfs
        mesh.create_mesh(
            model=model.decoder_model, 
            shape_feature=reconstructed_plane_feature, 
            filename=savedir, 
            sem_labels=specs['sem_labels'],
            N=reconstructed_plane_feature.shape[-1] * 2, 
            max_batch=2**21, 
            from_plane_features=True, 
            render_scene=args.render_scene,
            wall_remesh=args.wall_remesh
        )



if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--exp_dir", "-e", required=True,
        help="This directory should include experiment specifications in 'specs_test.json'.",
    )
    arg_parser.add_argument(
        "--lat_dir", "-l", required=True,
        help="This directory should include 'lats_sample.npy'.",
    )
    arg_parser.add_argument(
        "--resume", "-r", default='last',
        help="continue from previous saved logs, integer value or 'last'.",
    )
    arg_parser.add_argument(
        "--render_scene", "-s", action="store_true",
        help="render the holistic scene as scene.ply."
    )
    arg_parser.add_argument(
        "--wall_remesh", "-w", action="store_true",
        help="activate wall remesh algo."
    )

    args = arg_parser.parse_args()

    test(args)
