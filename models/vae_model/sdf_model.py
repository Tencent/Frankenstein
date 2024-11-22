#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import pytorch_lightning as pl 

import json

from models.vae_model.archs.sdf_decoder import SdfDecoder


class SdfSemModel(pl.LightningModule):
    def __init__(self, config_json):
        super().__init__()
        with open(config_json, encoding='utf-8') as f:
            self.dataset_dict = json.load(f)
        
        self.config = self.dataset_dict["config"]["Config"]

        self.mlp_path = self.dataset_dict["config"]["MLP"]

        self.model_geo = SdfDecoder(d_in=self.config["channel"], 
                                  d_out=self.config["n_labels"], 
                                  d_hidden=self.config["n_hid"], 
                                  n_layers=self.config["n_layers"], 
                                  ).cuda()

        ckpt_state_dict = torch.load(self.mlp_path, map_location="cuda")
        self.model_geo.load_state_dict(ckpt_state_dict, strict=True)
        print("loaded Geo and Tex model from {}".format(self.mlp_path))
        print(self.model_geo)
        self.model_geo.eval()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.specs["sdf_lr"])
        return optimizer
    
    def normalize_coordinate2(self, p, padding=0.1, plane='xz'):
        ''' Normalize coordinate to [0, 1] for unit cube experiments

        Args:
            p (tensor): point
            padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
            plane (str): plane feature type, ['xz', 'xy', 'yz']
        '''
        if plane == 'zx':
            xy = p[:, :, [2, 0]]
        elif plane =='yx':
            xy = p[:, :, [1, 0]]
        else:
            xy = p[:, :, [1, 2]]

        return xy

    # sample_plane_feature function copied from /src/conv_onet/models/decoder.py
    # uses values from plane_feature and pixel locations from vgrid to interpolate feature
    def sample_plane_feature(self, query, plane_feature, plane, padding=0.1):
        xy = self.normalize_coordinate2(query.clone(), plane=plane, padding=padding)
        xy = xy[:, :, None].float()
        # vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        # vgrid = xy - 1.0
        vgrid = xy
        sampled_feat = F.grid_sample(plane_feature, vgrid, padding_mode='border', align_corners=True, mode='bilinear').squeeze(-1)
        return sampled_feat


    def get_points_plane_features(self, plane_features, query):
        # plane features shape: batch, dim*3, 64, 64
        fea = {}
        fea['yx'], fea['zx'], fea['yz'] = plane_features[:, 0, ...], plane_features[:, 1, ...], plane_features[:, 2, ...]
        #print("shapes: ", fea['xz'].shape, fea['xy'].shape, fea['yz'].shape) #([1, 256, 64, 64])
        plane_feat_sum = 0
        plane_feat_sum += self.sample_plane_feature(query, fea['yx'], 'yx')
        plane_feat_sum += self.sample_plane_feature(query, fea['zx'], 'zx')
        plane_feat_sum += self.sample_plane_feature(query, fea['yz'], 'yz')

        return plane_feat_sum.transpose(2,1)


    def forward(self, plane_features_list, xyz):
        raise NotImplementedError("Run forward_sdf instead of forward!!!")


    def forward_sdf(self, plane_features, xyz):
        sdf_features = self.get_points_plane_features(plane_features, xyz) # point_features: B, N, D
        pred_sdf = self.model_geo(sdf_features)
        return pred_sdf


