import time
import numpy as np
import json
import os
import torch
from shutil import copyfile

from models.dataloader.triplane_stats import unnormalize
from diffusers.pipelines.ddpm.pipeline_ddpm_triplane import DDPMPipelineTriplane

def test(args):
    stats_dir = os.path.join(args.exp_dir, "stats")
    save_test_dir = os.path.join(args.exp_dir, "triplane_" + time.strftime('%Y-%m-%d-%H:%M:%S'))
    os.makedirs(save_test_dir, exist_ok=True)
    
    # diffusion pipeline
    generator = DDPMPipelineTriplane.from_pretrained(args.exp_dir, use_safetensors=True).to("cuda")

    lats_list = []
    imgs_list = []

    if len(args.userdef)>0 : # layout map cond
        assert args.num_samples == 1
        print("Generate from user-defined layout maps...")
        xz_image = torch.from_numpy(np.load(os.path.join(args.userdef, "layout_maps.npy"))).float().squeeze()

        assert args.userdef[-1] == '/'
        render_dir = args.userdef.split('/')[-2]

        latent_image = torch.stack(
            [
                torch.zeros_like(xz_image),
                xz_image, 
                torch.zeros_like(xz_image),
            ], dim=0  # [3,N,40,40,L]
        ).permute(1,0,4,2,3)   # [N,3,L,40,40]
        latent_image = torch.cat(
            [latent_image[:,0], latent_image[:,1], latent_image[:,2]], dim=-1
        ).cuda()  # [N,3,40,120]

        latent_width = latent_image.shape[2] 
        lats_sample = generator(batch_size=len(xz_image), cond=latent_image)
        lats_sample = unnormalize(lats_sample, stats_dir)

        lats_list = lats_sample
        imgs_list = latent_image[:,:,:,latent_width:-latent_width]  # [N,3,40,40]
        
        # save layout visualization
        for i in range(len(imgs_list)):
            save_map_name = render_dir+f'_{i:03d}.png'
            copyfile(os.path.join(args.userdef, save_map_name), os.path.join(save_test_dir, save_map_name))

    else: # uncond
        print("Generate from sampled random noises...")
        for i, batch in enumerate(range(args.num_samples)):
            
            lats_sample = generator(batch_size=args.batch_size)

            lats_sample = unnormalize(lats_sample, stats_dir)
            lats_list.append(lats_sample)
        lats_list = torch.concat(lats_list)


    # save latent tri-plane
    np.save(os.path.join(save_test_dir, f"lats_sample.npy"), lats_list.detach().cpu().numpy())

    


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--exp_dir", "-e", required=True, 
        help="This directory of exp"
    )
    arg_parser.add_argument("--num_samples", "-n", default=1, type=int, 
        help='number of samples to generate and reconstruct'
    )
    arg_parser.add_argument("--batch_size", "-b", default=5, type=int, 
        help='number of batch size to generate and reconstruct'
    )
    arg_parser.add_argument("--userdef", "-u", type=str, default='', 
        help='user-defined layout maps'
    )
    
    args = arg_parser.parse_args()

    test(args)
