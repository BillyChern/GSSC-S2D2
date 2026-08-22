# Adapted from 2DPASS (https://github.com/yanx27/2DPASS),
# network/voxel_fea_generator.py -- MIT License, Copyright (c) 2022 Benny
# (the holder named in the upstream LICENSE; yanx27 is the repository owner).
# See THIRD_PARTY_NOTICES.md section 6 for the full notice and the measured
# overlap. A bare upstream filename does not identify a project, which is why
# this header names one.
# import ipdb; ipdb.set_trace()
import numpy as np
import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch_scatter


class voxelization(nn.Module):
    def __init__(self):
        super().__init__()
        
    def set_pc_config(self, coors_range_xyz, spatial_shape):
        self.spatial_shape = spatial_shape
        self.coors_range_xyz = coors_range_xyz

    @staticmethod
    def sparse_quantize(pc, coors_range, spatial_shape):
        idx = spatial_shape * (torch.clamp(pc, max=coors_range[1]-1e-6) - coors_range[0]) / (coors_range[1] - coors_range[0])
        return idx.long()

    #def forward(self, data_dict):
    def forward(self, points, batch_idx):
        
        ip = points
        
        # filter out points out of range
        min_bound = self.coors_range_xyz.type(torch.float32)[:,0].to(points.device)
        max_bound = self.coors_range_xyz.type(torch.float32)[:,1].to(points.device)
        valid_point_mask= torch.all(
            (points[:,:3] < max_bound) & (points[:,:3] >= min_bound), dim=1)
            
        valid_points = points[valid_point_mask, :].clone()
        valid_batchidx = batch_idx[valid_point_mask].clone()
        
        points = valid_points
        batch_idx = valid_batchidx
        
        pc = points[:, :3]

        xidx = self.sparse_quantize(pc[:, 0], self.coors_range_xyz[0], self.spatial_shape[0])
        yidx = self.sparse_quantize(pc[:, 1], self.coors_range_xyz[1], self.spatial_shape[1])
        zidx = self.sparse_quantize(pc[:, 2], self.coors_range_xyz[2], self.spatial_shape[2])
        xidx = torch.clamp(xidx, 0, self.spatial_shape[0]-1)
        yidx = torch.clamp(yidx, 0, self.spatial_shape[1]-1)
        zidx = torch.clamp(zidx, 0, self.spatial_shape[2]-1)

        bxyz_indx = torch.stack([batch_idx, xidx, yidx, zidx], dim=-1).long()
        unq, unq_inv, unq_cnt = torch.unique(bxyz_indx, return_inverse=True, return_counts=True, dim=0)
        '''print(max_bound, min_bound)
        print(torch.max(ip[:,0])<max_bound[0], torch.min(ip[:,0])>=min_bound[0])
        print(torch.max(ip[:,1])<max_bound[1], torch.min(ip[:,1])>=min_bound[1])
        print(torch.max(ip[:,2])<max_bound[2], torch.min(ip[:,2])>=min_bound[2])        
        print(torch.max(points[:,0])<max_bound[0], torch.min(points[:,0])>=min_bound[0])
        print(torch.max(points[:,1])<max_bound[1], torch.min(points[:,1])>=min_bound[1])
        print(torch.max(points[:,2])<max_bound[2], torch.min(points[:,2])>=min_bound[2])
        print(torch.max(ip[:,0]), torch.min(ip[:,0]))
        print(torch.max(ip[:,1]), torch.min(ip[:,1]))
        print(torch.max(ip[:,2]), torch.min(ip[:,2]))
        print(torch.max(ip[:,3]), torch.min(ip[:,3]))
        print(torch.max(points[:,0]), torch.min(points[:,0]))
        print(torch.max(points[:,1]), torch.min(points[:,1]))
        print(torch.max(points[:,2]), torch.min(points[:,2]))
        print(torch.max(points[:,3]), torch.min(points[:,3]))

        print(torch.max(unq[:,0]), torch.min(unq[:,0]))
        print(torch.max(unq[:,1]), torch.min(unq[:,1]))
        print(torch.max(unq[:,2]), torch.min(unq[:,2]))
        print(torch.max(unq[:,3]), torch.min(unq[:,3]))'''
        #unq = torch.cat([unq[:, 0:1], unq[:, [3, 2, 1]]], dim=1) #bzyx
        unq = torch.cat([unq[:, 0:1].clone(), unq[:, [3, 2, 1]].clone()], dim=1) #bzyx
        '''print(torch.max(unq[:,0]), torch.min(unq[:,0]))
        print(torch.max(unq[:,1]), torch.min(unq[:,1]))
        print(torch.max(unq[:,2]), torch.min(unq[:,2]))
        print(torch.max(unq[:,3]), torch.min(unq[:,3]))'''

        return points, bxyz_indx, unq, unq_inv


class voxel_3d_generator(nn.Module):
    def __init__(self, in_channels, out_channels, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.PPmodel = nn.Sequential(
            nn.Linear(in_channels + 6, out_channels),
            nn.ReLU(True),
            nn.Linear(out_channels, out_channels)
        )
        
    def set_pc_config(self, coors_range_xyz, spatial_shape, batch_size):
        self.spatial_shape = spatial_shape
        self.coors_range_xyz = coors_range_xyz
        self.batch_size = batch_size

    def prepare_input(self, point, grid_ind, inv_idx):
        # points: point cloud, N x 3+ (xyzr) (0-51.2,-25.6-25.6,-2-4)
        # grid_ind: the voxel index of each point, N x 3 (xyz) (0-256,0-256,0-32)
        # inv_idx: mapping each point to a occupied voxel, N
        
        pc_mean = torch_scatter.scatter_mean(point[:, :3], inv_idx, dim=0)[inv_idx]
        nor_pc = point[:, :3] - pc_mean

        coors_range_xyz = self.coors_range_xyz.clone()
        cur_grid_size = self.spatial_shape.clone()
        crop_range = coors_range_xyz[:, 1] - coors_range_xyz[:, 0]
        intervals = (crop_range / cur_grid_size).to(point.device)
        voxel_centers = grid_ind * intervals + coors_range_xyz[:, 0].to(point.device)
        center_to_point = point[:, :3] - voxel_centers

        pc_feature = torch.cat((point, nor_pc, center_to_point), dim=1)
        return pc_feature

    def forward(self, points, grid_ind, unq_idx, inv_idx):
        pt_fea = self.prepare_input(points, grid_ind[:,1:], inv_idx)
        pt_fea = self.PPmodel(pt_fea.clone().detach())

        features = torch_scatter.scatter_mean(pt_fea, inv_idx, dim=0)
        sparse_tensor = spconv.SparseConvTensor(
            features=features,
            indices=unq_idx.int(),
            spatial_shape=np.int32(self.spatial_shape.cpu().detach().numpy())[::-1].tolist(),
            batch_size=self.batch_size
        )
        out = sparse_tensor.dense()
        #sparse_tensor_t = torch.sparse_coo_tensor(unq_idx.int().permute(1,0), features, [1,32,120,120,64])#, [32,120,120])
        #out = sparse_tensor_t.to_dense().permute(0,4,1,2,3)

        

        '''try:
            out = sparse_tensor.dense()
        except:
            print(pt_fea.shape)
            print(features.shape)
            print(unq_idx.shape)
            print(torch.max(unq_idx[:,0]), torch.min(unq_idx[:,0]))
            print(torch.max(unq_idx[:,1]), torch.min(unq_idx[:,1]))
            print(torch.max(unq_idx[:,2]), torch.min(unq_idx[:,2]))
            print(torch.max(unq_idx[:,3]), torch.min(unq_idx[:,3]))
            print(self.spatial_shape.shape)
            print(self.spatial_shape)
            print(np.int32(self.spatial_shape.cpu().detach().numpy())[::-1].tolist())
            print(batch_size)
            print(features, unq_idx)
            assert False'''
        return out