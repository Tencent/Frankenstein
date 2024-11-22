# Frankenstein: Generating Semantic-Compositional 3D Scenes in One Tri-Plane

### SIGGRAPH Asia 2024 (Conference Track)

### [Project page](https://wolfball.github.io/frankenstein/) | [arXiv](https://arxiv.org/abs/2403.16210) | [Video](https://youtu.be/lRn-HqyCrLI)

![image](images/repre_image.jpg)


### News

- [2024.11] Inference code released.




### Installation

```bash
conda create -n frankenstein python=3.9 -y
conda activate frankenstein
pip install torch==2.0.1 torchvision==0.15.2
pip install -r requirements.txt
```

We slightly modify the diffusers:
- add "num_layers_mid" to UNet2DModel in diffusers/models/unet_2d.py
- add pipeline_ddpm_triplane.py in diffusers/pipelines/ddpm/


### Pretrained Model

For bedroom:
- Download [bedroom vae (2.3G)](https://drive.google.com/file/d/1rgCFJH5cDz1OwikRqHLNPLG0eSH2ElUL/view?usp=sharing) to vae_training/configs/vae_vroid/last.ckpt
- Downlaod [bedroom diffusion (383.2M)](https://drive.google.com/file/d/1aor6-AgVo3lGmF8_ANfqHfQWuotxeYcl/view?usp=sharing) to diff_denoising/configs/diff_vroid/unet/diffusion_pytorch_model.safetensors

For livingroom:
- Download [livingroom vae (2.3G)](https://drive.google.com/file/d/1C5Jhon-I9hmt9bxIS-RDtgNZRenS4Q1b/view?usp=sharing) to vae_training/configs/vae_vroid/last.ckpt
- Downlaod [livingroom diffusion (383.2M)](https://drive.google.com/file/d/1gJvJIKoWf4trGs3P7myKzpGI1FAiKO3W/view?usp=sharing) to diff_denoising/configs/diff_vroid/unet/diffusion_pytorch_model.safetensors

For vroid:
- Download [vroid vae (4.4G)](https://drive.google.com/file/d/1amtqwby7o-PjTvL9aOWdwvTGb3Xgg-a9/view?usp=sharing) to vae_training/configs/vae_vroid/last.ckpt
- Downlaod [vroid diffusion (1.1G)](https://drive.google.com/file/d/1N-j8yB-LlamEIf4xLW1IIiy9R4UPX-G9/view?usp=sharing) to diff_denoising/configs/diff_vroid/unet/diffusion_pytorch_model.safetensors

### Inference

For bedroom (layout condition):
```bash
# generate layout
# move 1-th (-o 1) cabinet (-i 2) along z-axis (-d 2)
python paint_layout.py -b 62d0964d-a9c3-4f54-a1d5-4709a289193f_MasterBedroom-11117 -t bedroom -d 2 -i 2 -o 1

# generate latent tri-plane conditioned by layout
python test_diff.py -e diff_denoising/configs/diff_bedroom/ -u painting_bedroom/62d0964d-a9c3-4f54-a1d5-4709a289193f_MasterBedroom-11117/

# decode to 3D mesh
python test_latents.py -e vae_training/configs/vae_bedroom/ -s -l diff_denoising/configs/diff_bedroom/triplane_xxxxxx/
```

For livingroom (layout condition + wall remeshing):
```bash
# generate layout
# move 0-th (-o 0) cabinet (-i 1) along x-axis (-d 0)
python paint_layout.py -b de513c79-61f9-4689-b671-fad361e605a3_LivingDiningRoom-16466 -t livingroom -d 0 -i 1 -o 0

# generate latent tri-plane conditioned by layout
python test_diff.py -e diff_denoising/configs/diff_livingroom/ -u painting_livingroom/de513c79-61f9-4689-b671-fad361e605a3_LivingDiningRoom-16466/

# decode to 3D mesh ('-w' activate the wall remeshing algorithm)
python test_latents.py -e vae_training/configs/vae_livingroom/ -w -l diff_denoising/configs/diff_livingroom/triplane_xxxxxx/
```

For avatar (uncondition):
```bash
# generate latent tri-plane from random noise (unconditional)
python test_diff.py -e diff_denoising/configs/diff_vroid/

# decode to 3D mesh
python test_latents.py -e vae_training/configs/vae_vroid/ -s -l diff_denoising/configs/diff_vroid/triplane_xxxxxx/
```




### Citation
If you find our code or paper helps, please consider citing:

```text
@article{yan2024frankenstein,
    author    = {Han, Yan and Yang, Li and Zhennan, Wu and Shenzhou, Chen and Weixuan, Sun and Taizhang, Shang and Weizhe, Liu and Tian, Chen and Xiaqiang, Dai and Chao, Ma and Hongdong, Li and Pan, Ji},
    title     = {Frankenstein: Generating Semantic-Compositional 3D Scenes in One Tri-Plane},
    journal   = {ACM SIGGRAPH Asia Conference Proceedings},
    year      = {2024},
}
```


### Acknowledgments

Some source codes are borrowed from [DiffusionSDF](https://github.com/princeton-computational-imaging/Diffusion-SDF), [Diffusers](https://github.com/huggingface/diffusers), [IDR](https://github.com/lioryariv/idr).