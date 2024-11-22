import torch
import time
import pytorch_lightning as pl
import os
import numpy as np

from models.vae_model.lr_scheduler.transformer_lr_scheduler import TransformerStayLRScheduler
from models.vae_model.utils_model import get_decoder_model, get_vae_model
from models.dataloader.triplane_stats import normalize, unnormalize


class CombinedModel(pl.LightningModule):
    def __init__(self, specs, args):
        super().__init__()
        self.specs = specs
        self.cur_epoch = 0
        self.args = args
        
        self.stats_dir = None
        if os.path.exists(os.path.join(self.args.exp_dir, "stats")):
            self.stats_dir = os.path.join(self.args.exp_dir, "stats")
            min_values = np.load(f'{self.stats_dir}/lower_bound.npy').astype(np.float32).reshape(1, -1, 1, 1)  # should be (1, 96, 1, 1)
            max_values = np.load(f'{self.stats_dir}/upper_bound.npy').astype(np.float32).reshape(1, -1, 1, 1)
            self._range = max_values - min_values
            self.middle = (min_values + max_values) / 2

        self.latent_dir = os.path.join(self.args.exp_dir, "modulations" + time.strftime('%Y-%m-%d-%H:%M:%S'))
        self.task = specs['training_task']
        self.n_labels = len(specs["sem_labels"].keys())


        if "decoder_config" in specs:
            self.decoder_model = get_decoder_model(specs)

        if "vae_config" in specs:
            self.vae_model = get_vae_model(specs)
            
        if "loss_config" in specs:
            self.loss_config = specs["loss_config"]




    def training_step(self, x, idx):
        if self.current_epoch > self.cur_epoch:
            self.cur_epoch = self.current_epoch
            torch.cuda.empty_cache()
        if self.task == 'vae_sdfvroid':
            return self.train_vae_sdf(x)
        elif self.task == 'vae_sdfroom':
            return self.train_vae_sdf(x)


    def save_triplane(self, triplane, filenames, triplane_name="triplane.pt"):
        for plane, filedir in zip(triplane, filenames):
            save_path = os.path.join(filedir, triplane_name)
            torch.save(plane.cpu(), save_path)


    def configure_optimizers(self):
        if self.task in ['vae_sdfvroid']:
            if "train_params" in self.specs.keys() and self.specs["train_params"] == "all":
              params_list = [
                      { 'params': self.parameters(), 'lr':self.specs['lr_init'] }
                  ]
            elif "train_params" in self.specs.keys() and self.specs["train_params"] == "vae":
              params_list = [
                      { 'params': self.vae_model.parameters(), 'lr':self.specs['lr_init'] }
                  ]
            else:
                exit("train_params are not specified")
        
        elif self.task in ["vae_sdfroom012"]:
            if "train_params" in self.specs.keys() and self.specs["train_params"] == "all":
              params_list = [
                      { 'params': self.parameters(), 'lr':self.specs['lr_init'] }
                  ]
            elif "train_params" in self.specs.keys() and self.specs["train_params"] == "vae":
              params_list = [
                      { 'params': self.vae_model.parameters(), 'lr':self.specs['lr_init'] }
                  ]
            else:
                exit("train_params are not specified")
            optimizer = torch.optim.Adam(params_list)
            return {"optimizer": optimizer}
            

        optimizer = torch.optim.Adam(params_list)

        lr_scheduler = TransformerStayLRScheduler(optimizer = optimizer,
                    init_lr=self.specs['lr_init'],
                    peak_lr=self.specs['lr'],
                    final_lr=self.specs['final_lr'],
                    final_lr_scale=self.specs['final_lr_scale'],
                    warmup_steps=self.specs['warmup_steps'],
                    stay_steps=self.specs['stay_steps'],
                    decay_steps=self.specs['decay_steps'])


        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "monitor": "total_loss",
                    "frequency": 1,
                    "interval": "epoch"
                }
              }


    def train_vae_sdf(self, data):
        points_surface = data["surface_points"].cuda()   # [B,N,3]
        normal_surface = data["surface_normals"].cuda()  # [B,N,3]
        labels_surface = data["surface_labels"].cuda()   # [B,N,1]
        points_empty = data["sdf_points"].cuda()         # [B,N,3]
        sdf_empty_gt = data["sdf_sdfs"].cuda()           # [B,N,9]
        plane_features = data["sdf_triplane"].cuda()     # [B,3,32,256,256]

        # STEP 1: obtain reconstructed plane feature and latent code 
        random_points = torch.from_numpy(
                            np.random.uniform(-1.0, 1.0, size=points_empty.shape).astype(np.float32)
                        ).cuda()
        points_surface_num = points_surface.shape[1]
        # # STEP 1: obtain reconstructed plane feature and latent code 
        # plane_features = plane_features.clamp(-1.0, 1.0)

        # normalize
        if self.stats_dir is not None:
            plane_features = normalize(plane_features, self.stats_dir, self.middle, self._range)
        else:
            raise RuntimeError
        ### training
        
        out = self.vae_model(plane_features) # out = [self.decode(z), input, mu, log_var, z]
        reconstructed_plane_feature, latent = out[0], out[-1]

        #####  vae loss: l1 + kl loss
        try:
            vae_loss = self.vae_model.loss_function(*out)
        except:
            print("vae loss is nan at epoch {}...".format(self.current_epoch))
            return None # skips this batch

        loss_l1 = (reconstructed_plane_feature - plane_features).abs().mean()

        if self.stats_dir is not None:
            reconstructed_plane_feature = unnormalize(reconstructed_plane_feature, self.stats_dir, self.middle, self._range)
        
        ###### sdf loss
        sdf_empty_pred = self.decoder_model.forward_sdf(reconstructed_plane_feature, points_empty)
        surface_sdf = self.decoder_model.forward_sdf(reconstructed_plane_feature, points_surface)
        
        sdf_surf_nei = self.gradient_tri(self.decoder_model.forward_sdf, reconstructed_plane_feature, points_surface)  # [B,6,N,L]
        sdf_rndm_nei = self.gradient_tri(self.decoder_model.forward_sdf, reconstructed_plane_feature, random_points)  # [B,6,N,L]
        eps_s = 1e-6
        eps_v = 1e-6
        surface_norm_pred = torch.cat([
            0.5 * torch.gather(sdf_surf_nei[:,0] - sdf_surf_nei[:,1], 2, labels_surface) / eps_s,
            0.5 * torch.gather(sdf_surf_nei[:,2] - sdf_surf_nei[:,3], 2, labels_surface) / eps_s,
            0.5 * torch.gather(sdf_surf_nei[:,4] - sdf_surf_nei[:,5], 2, labels_surface) / eps_s,
        ], dim=-1)  # 【B,N,3]
        points_norm_rndm = torch.stack([
            0.5 * (sdf_rndm_nei[:,0] - sdf_rndm_nei[:,1]) / eps_v,
            0.5 * (sdf_rndm_nei[:,2] - sdf_rndm_nei[:,3]) / eps_v,
            0.5 * (sdf_rndm_nei[:,4] - sdf_rndm_nei[:,5]) / eps_v,
        ], dim=-1)


        eikonal_loss = ((points_norm_rndm.norm(2, dim=-1) - 1) ** 2).mean()

        normals_loss_sem = ((surface_norm_pred - normal_surface)).norm(2, dim=-1).reshape(-1)
        surface_loss_sem = torch.gather(surface_sdf, 2, labels_surface).abs().reshape(-1)  # [B,N,1] -> [B*N]

        sur_loss_list = [torch.tensor(0.).cuda()] * self.n_labels
        nor_loss_list = [torch.tensor(0.).cuda()] * self.n_labels
        for lb in range(self.n_labels):
            lb_mask = (labels_surface == lb).reshape(-1)
            if len(surface_loss_sem[lb_mask]) > 0:
                sur_loss_list[lb] = surface_loss_sem[lb_mask].mean()
            if len(normals_loss_sem[lb_mask]) > 0:
                nor_loss_list[lb] = normals_loss_sem[lb_mask].mean()

        surface_sdf_loss = torch.stack(sur_loss_list).sum()
        normals_loss = torch.stack(nor_loss_list).mean()

        spe_sdf_mask = sdf_empty_gt < 60
        psd_sdf_loss = (sdf_empty_pred[spe_sdf_mask] - sdf_empty_gt[spe_sdf_mask]).abs().mean()

        # ###### tv loss
        if self.current_epoch < 3:
            loss = loss_l1 * self.loss_config["loss_l1_weight"] + vae_loss * self.loss_config["loss_vae_weight"]
        else:
            loss = vae_loss * self.loss_config["loss_vae_weight"] + \
                   eikonal_loss * self.loss_config["loss_eikonal_weight"] + \
                   surface_sdf_loss * self.loss_config["loss_surface_sdf_weight"] +  \
                   normals_loss * self.loss_config["loss_normals_weight"] +  \
                   psd_sdf_loss * self.loss_config["loss_psd_sdf_weight"]

        loss_dict =  {
            "L": loss,
            "Leik": eikonal_loss,
            "Lsur": surface_sdf_loss,
            "Lnor": normals_loss,
            "Lsdf": psd_sdf_loss,
            "rec": loss_l1,
            "vae": vae_loss,
        }
        
        for lb in range(self.n_labels):
            loss_dict[f'Lsur{lb}'] = sur_loss_list[lb]
            loss_dict[f'Lnor{lb}'] = nor_loss_list[lb]
        
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False, logger=True)
        return loss


    def on_train_epoch_start(self):
        print("start time: {}, experiment: {}".format(time.strftime('%Y-%m-%d-%H:%M:%S'), self.args.exp_dir))
        return
    
    def on_train_epoch_end(self):
        pass

    def gradient_tri(self, net, plane_feature, x):
        b = x.shape[0]
        n = x.shape[1]
        eps = 1e-6
        x_nei = torch.stack([
            x + torch.as_tensor([[eps, 0.0, 0.0]]).to(x),
            x + torch.as_tensor([[-eps, 0.0, 0.0]]).to(x),
            x + torch.as_tensor([[0.0, eps, 0.0]]).to(x),
            x + torch.as_tensor([[0.0, -eps, 0.0]]).to(x),
            x + torch.as_tensor([[0.0, 0.0, eps]]).to(x),
            x + torch.as_tensor([[0.0, 0.0, -eps]]).to(x)
        ], dim=1).view(b, n*6, 3)
        sdf_nei = net(plane_feature, x_nei)
        sdf_nei = sdf_nei.view(b, 6, n, -1)
        return sdf_nei


    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']