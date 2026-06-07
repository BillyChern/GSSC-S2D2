import copy
import functools
import os
import time
from math import log

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from apex import amp
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    zero_grad,
)
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        resume_step,
        diffusion_type,
        num_classes,
        save_pth=None,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        use_lidar=False,
        binary=False,
        point_diffusion=False,
        n_points=0,
        lidar_point=True,
        kld=False,
        downsample_resolution=1,
        sparse_conv=False,
        label_diffusion=False,
        scale_input=False,
        refine=False,
        ignore_invalid=False,
        analog_bits=False,
        one_hot=False,
        use_bev=False,
        use_bev_vol=False,
        use_multiclass_bev=False,
        sparse_unet=False,
        bev_vol_num=0
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.point_diffusion = point_diffusion
        self.n_points = n_points
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.kld = kld
        self.save_pth = save_pth
        self.downsample_resolution = downsample_resolution
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.use_lidar = use_lidar
        self.binary = binary
        self.sparse_conv = sparse_conv
        self.label_diffusion = label_diffusion
        self.lidar_point = lidar_point
        self.scale_input = scale_input
        self.refine = refine
        self.ignore_invalid = ignore_invalid
        self.analog_bits = analog_bits
        self.one_hot = one_hot
        self.diffusion_type = diffusion_type
        self.use_bev = use_bev
        self.use_bev_vol = use_bev_vol
        self.use_multiclass_bev = use_multiclass_bev
        self.num_classes = num_classes
        self.sparse_unet = sparse_unet
        self.bev_vol_num = bev_vol_num

        self.step = 0
        self.resume_step = resume_step
        self.global_batch = self.batch_size * dist.get_world_size()

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        #if self.use_fp16:
            #self._setup_fp16()

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        if self.use_fp16:
            self.model, self.opt = amp.initialize(self.model, self.opt, opt_level="O1")

        if th.cuda.is_available() and not dist_util.DEBUG:
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self._state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        print(bf.dirname(main_checkpoint),bf.exists(opt_checkpoint))
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()
        
    def run_loop(self):
        total_time = 0
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            #batch, cond = next(self.data)
            file_names, batch_dict = next(self.data)

            timer = time.time()
            #print(len(batch_dict), len(file_names), file_names, batch_dict.keys())
            '''out = batch_dict['label'][0].permute(2,1,0).clone().cpu().numpy()
            np.save("vis_an/000000.label",out.astype(np.int16).reshape(-1))
            os.system("mv vis_an/000000.label.npy vis_an/000000.label")'''
            
            assert sum([self.point_diffusion,self.label_diffusion,self.one_hot,self.analog_bits,self.binary]) <= 1
            
            
            if self.diffusion_type == "multinomial":
                '''print(batch_dict['label'].shape, batch_dict['input'].shape, batch_dict['bev'].shape)
                print(batch_dict['label'].type(), batch_dict['input'].type(), batch_dict['bev'].type())
                print(th.unique(batch_dict['label']), th.unique(batch_dict['bev']))
                assert False'''
                height = batch_dict['label'].shape[1]
                batch = batch_dict['label'].clone()
                batch_cond = batch_dict['input'].clone().type(th.float)
                batch_bev = None
                if self.use_bev: #2D bev
                    batch_bev = batch_dict['bev'].clone().type(th.long)
                         
                if self.use_bev_vol: #3D bev
                    target = batch_dict['bev'].clone().type(th.long)
                    label_2_height = th.zeros(self.num_classes+1,height).type(th.long).to(target.device)
                    if self.bev_vol_num == 1:
                        label_2_height[1,:6] = 1 #building
                        label_2_height[2,:7] = 2 #fences
                        label_2_height[3,:3] = 3 #others
                        label_2_height[4,:5] = 4 #pedestrian
                        label_2_height[5,:7] = 5 #pole
                        label_2_height[6,:1] = 6 #road
                        label_2_height[7,:1] = 7 #ground
                        label_2_height[8,:1] = 8 #sidewalk
                        label_2_height[9,:6] = 9 #vegetation
                        label_2_height[10,:6] = 10 #vehicle
                        label_2_height[11,:8] = 11 #unconditioned generation
                    elif self.bev_vol_num == 2:
                        label_2_height[1,:8] = 1 #building
                        label_2_height[2,:8] = 2 #fences
                        label_2_height[3,:8] = 3 #others
                        label_2_height[4,:8] = 4 #pedestrian
                        label_2_height[5,:8] = 5 #pole
                        label_2_height[6,:1] = 6 #road
                        label_2_height[7,:1] = 7 #ground
                        label_2_height[8,:1] = 8 #sidewalk
                        label_2_height[9,:8] = 9 #vegetation
                        label_2_height[10,:8] = 10 #vehicle
                        label_2_height[11,:8] = 11 #unconditioned generation
                    elif self.bev_vol_num == 3:
                        for class_idx in range(self.num_classes):
                            label_2_height[class_idx+1,:height] = class_idx+1
                        '''label_2_height[1,:8] = 1 #building
                        label_2_height[2,:8] = 2 #fences
                        label_2_height[3,:8] = 3 #others
                        label_2_height[4,:8] = 4 #pedestrian
                        label_2_height[5,:8] = 5 #pole
                        label_2_height[6,:8] = 6 #road
                        label_2_height[7,:8] = 7 #ground
                        label_2_height[8,:8] = 8 #sidewalk
                        label_2_height[9,:8] = 9 #vegetation
                        label_2_height[10,:8] = 10 #vehicle
                        label_2_height[11,:8] = 11 #unconditioned generation'''

                    bev_vol = F.embedding(target, label_2_height).permute(0,3,1,2)
                    batch_bev = bev_vol
                
                if self.sparse_unet:
                    target = batch_dict['bev'].clone().type(th.long)
                    label_2_height = th.zeros(12,8).type(th.long).to(target.device)
                    label_2_height[0,:1] = 1 #ground for free space
                    label_2_height[1,:8] = 1 #building
                    label_2_height[2,:8] = 2 #fences
                    label_2_height[3,:8] = 3 #others
                    label_2_height[4,:7] = 4 #pedestrian
                    label_2_height[5,:8] = 5 #pole
                    label_2_height[6,:1] = 6 #road
                    label_2_height[7,:1] = 7 #ground
                    label_2_height[8,:1] = 8 #sidewalk
                    label_2_height[9,:8] = 9 #vegetation
                    label_2_height[10,:8] = 10 #vehicle
                    label_2_height[11,:8] = 11 #unconditioned generation
                    bev_mask = F.embedding(target, label_2_height).permute(0,3,1,2)
                    batch_bev_mask = bev_mask != 0
                    
                    batch_dict['invalid'] = (batch_bev_mask == 0)
                
                if self.use_multiclass_bev:
                    batch_bev = batch_dict['bev'].clone().type(th.float)
                    
                cond = {'y':batch_cond, 'bev':batch_bev, 'bev_mask':batch_bev_mask if self.sparse_unet else None}
                
                if 'pc_data' in batch_dict:
                    cond['pc_data'] = batch_dict['pc_data']
                else:
                    cond['pc_data'] = None
                if "invalid_aug" in batch_dict:
                    cond['invalid_aug'] = batch_dict["invalid_aug"]

                
            elif self.point_diffusion:
                downsampled_batch = th.nn.functional.interpolate(batch_dict['label'].unsqueeze(1).type(th.float), scale_factor=self.downsample_resolution, mode='nearest').squeeze(1)
                downsampled_cond = th.nn.functional.interpolate(batch_dict['input'].unsqueeze(1).type(th.float), scale_factor=self.downsample_resolution, mode='nearest').squeeze(1)
                #downsampled_cond = th.nn.functional.max_pool_3d(batch_dict['input'].unsqueeze(1).type(th.float))
                #print(downsampled_batch.shape)
                #bs, h, w, l = batch_dict['label'].shape
                bs, h, w, l = downsampled_batch.shape
                batch = []
                conds = []
                #for sample in batch_dict['label']:
                for sample in downsampled_batch:
                    voxel_coords = th.stack(th.meshgrid(th.arange(0,h,1)/(h-1), th.arange(0,w,1)/(w-1), th.arange(0,l,1)/(l-1))) * 2 - 1 # to [-1,1]
                    voxel_coords = voxel_coords.to(downsampled_batch.device)
                    #print(downsampled_batch.device, voxel_coords.device)
                    voxel_logits = th.zeros(20,h,w,l, device=downsampled_batch.device)
                    voxel_logits = voxel_logits.scatter_(0, sample.unsqueeze(0).type(th.long), 1)
                    voxel_valid = (sample != 0)
                    voxel_invalid = (sample == 0)
                    if self.binary:
                        
                        if self.kld:
                            voxel_logits_bin = th.zeros(2,h,w,l, device=downsampled_batch.device)
                            bin_sample = (sample != 0).type(th.float)
                            voxel_logits = voxel_logits_bin.scatter_(0, bin_sample.unsqueeze(0).type(th.long), 1)
                        else:
                            voxel_logits = (sample != 0).type(th.float).unsqueeze(0)
                            voxel_logits_msk = voxel_logits == 0
                            voxel_logits[voxel_logits_msk] = -1
                        #print(voxel_coords.shape, voxel_logits.shape)
                        
                        voxel_feats = th.cat([voxel_coords, voxel_logits], dim=0).permute(1,2,3,0)
                        
                        #voxel_feats = voxel_coords.permute(1,2,3,0)
                    else:
                        voxel_feats = th.cat([voxel_coords, voxel_logits], dim=0).permute(1,2,3,0)
                    occupied_point_sample = voxel_feats[voxel_valid].transpose(0,1)
                    unoccupied_point_sample = voxel_feats[voxel_invalid]
                    #print(self.n_points, occupied_point_sample.shape)
                    n_empty_points = self.n_points - occupied_point_sample.shape[-1]
                    unoccupied_indices = th.randint(0, unoccupied_point_sample.shape[0], (n_empty_points,))
                    unoccupied_point_sample = unoccupied_point_sample[unoccupied_indices].transpose(0,1)
                    point_sample = th.cat([occupied_point_sample, unoccupied_point_sample], dim=-1)
                    batch.append(point_sample.clone().detach())
                    '''vgrid = th.zeros(h,w,l)
                    print(h,w,l)
                    for v in occupied_point_sample[0:3,].transpose(0,1):
                        v = ((v + 1) / 2 * th.tensor([h,w,l])).type(th.long)
                        vgrid[v[0],v[1],v[2]] = 1
                    print((vgrid == voxel_valid).all())
                    assert False'''
                        
                    # Note: random empty-voxel sampling is intentionally not added here;
                    # the current path uses only occupied voxels for the point sample.
                    #print(voxel_coords.shape, voxel_logits.shape, point_sample.shape)
                
                if self.lidar_point:
                    conds = th.zeros(bs,512,3).to(downsampled_batch.device) #max len 512
                    attn_mask = th.ones(bs,512+self.n_points).to(downsampled_batch.device)
                    i = 0
                    for cond in downsampled_cond:
                        input_coords = (th.stack(th.meshgrid(th.arange(0,h,1)/(h-1), th.arange(0,w,1)/(w-1), th.arange(0,l,1)/(l-1))) * 2 - 1).permute(1,2,3,0).to(downsampled_batch.device)
                        input_valid = (cond != 0)
                        valid_input_coords = input_coords[input_valid].clone()
                        conds[i,:valid_input_coords.shape[0]] = valid_input_coords
                        attn_mask[i, valid_input_coords.shape[0]:512] = th.zeros(512-valid_input_coords.shape[0]).to(downsampled_batch.device)
                        i += 1
                        #conds.append(valid_input_coords.clone().detach())
                    attn_mask = attn_mask.unsqueeze(-1)
                    attn_mask = th.einsum("bpa,baq->bpq",attn_mask, attn_mask.transpose(-2,-1))
                batch = th.stack(batch, dim=0)
                #cond = {'y':th.stack(conds, dim=0)}
                cond = {'y':downsampled_cond} if not self.lidar_point else {'y':conds,'attn_mask':attn_mask}
            elif self.label_diffusion:
                batch_size = batch_dict['label'].shape
                target = batch_dict['label'].unsqueeze(1)
                batch = th.zeros(batch_dict['label'].shape, device=target.device).unsqueeze(1).repeat(1,20,1,1,1)
                batch = batch.scatter_(1, target, 1)
                batch = batch[:,1:].clone()
                
                batch_cond = batch_dict['input'].type(th.float)
                if self.downsample_resolution != 1:
                    m = th.nn.MaxPool3d(self.downsample_resolution)
                    #downsampled_label = torch.nn.functional.interpolate(v.unsqueeze(1).type(torch.float), scale_factor=0.5, mode='nearest').squeeze(1)
                    batch = m(batch)          
                    batch_cond = m(batch_cond.unsqueeze(1)).squeeze(1)  
                    
                if self.sparse_conv:
                    #assert self.downsample_resolution == 1 #currently dont know how to downsample a labelled voxel grid
                    if self.refine:
                        target = batch_dict['refine'].unsqueeze(1)
                        batch_cond = th.zeros(batch_dict['refine'].shape, device=target.device).unsqueeze(1).repeat(1,20,1,1,1)
                        batch_cond = batch_cond.scatter_(1, target, 1)
                        batch_cond = batch_cond[:,1:].clone()
                        if self.downsample_resolution != 1:
                            batch_cond = m(batch_cond).clone()
                    else:
                        input_mask = batch_dict['input'] == 0
                        target = batch_dict['label'].unsqueeze(1)
                        target[input_mask.unsqueeze(1)] = 0
                        batch_cond = th.zeros(batch_dict['label'].shape, device=target.device).unsqueeze(1).repeat(1,20,1,1,1)
                        batch_cond = batch_cond.scatter_(1, target, 1)
                        batch_cond = batch_cond[:,1:].clone()
                        if self.downsample_resolution != 1:
                            batch_cond = m(batch_cond).clone()
                    #batch_cond = batch_dict['label']
                    #batch_cond[input_mask] = 0
                            
                if self.use_lidar:
                    cond = {'y': batch_cond.detach()}
                else:
                    cond = None   
            elif self.one_hot:
                batch_size = batch_dict['label'].shape
                if self.downsample_resolution == 1:
                    target = batch_dict['label'].clone().unsqueeze(1).type(th.long)
                elif self.downsample_resolution == 2:
                    target = batch_dict['label_2'].clone().unsqueeze(1)
                else:
                    pass
                batch = th.zeros(target.shape, device=target.device).repeat(1,20,1,1,1)
                batch = batch.scatter_(1, target, 1).clone()
                
                batch_cond = batch_dict['input'].type(th.float)
                    
                if self.sparse_conv:
                    #assert self.downsample_resolution == 1 #currently dont know how to downsample a labelled voxel grid
                    if self.refine:
                        assert False
                        target = batch_dict['refine'].unsqueeze(1)
                        batch_cond = th.zeros(batch_dict['refine'].shape, device=target.device).unsqueeze(1).repeat(1,20,1,1,1)
                        batch_cond = batch_cond.scatter_(1, target, 1)
                        batch_cond = batch_cond[:,1:].clone()
                        if self.downsample_resolution != 1:
                            batch_cond = m(batch_cond).clone()
                    else:
                        input_mask = batch_dict['input'] == 0
                        target = batch_dict['label'].unsqueeze(1).type(th.long)
                        target[input_mask.unsqueeze(1)] = 0
                        batch_cond = th.zeros(batch_dict['label'].shape, device=target.device).unsqueeze(1).repeat(1,20,1,1,1)
                        batch_cond = batch_cond.scatter_(1, target, 1).clone()
                        #batch_cond = batch_cond[:,1:].clone()
                    #batch_cond = batch_dict['label']
                    #batch_cond[input_mask] = 0
                            
                if self.use_lidar:
                    cond = {'y': batch_cond.detach()}
                else:
                    cond = None   
            elif self.analog_bits:
                if self.downsample_resolution == 1:
                    batch = batch_dict['label'].clone().unsqueeze(1).type(th.float)
                elif self.downsample_resolution == 2:
                    batch = batch_dict['label_2'].clone().unsqueeze(1).type(th.float)
                else:
                    pass
                bs, c, z, y, x = batch.shape
                bits_cnt = int(log(self.num_classes,2)) + 1
                bits_batch = th.zeros(bs,bits_cnt,z,y,x)
                for k in range(0,bits_cnt):
                    bits_k = th.fmod(batch, 2)
                    #print(batch.shape, bits_k.shape, bits_batch[:,4-k:5-k].shape)
                    bits_batch[:,bits_cnt-1-k:bits_cnt-k] = bits_k
                    batch = batch.div(2, rounding_mode="trunc")

                batch = bits_batch
                
                bs, c, z, y, x = batch_dict['input'].unsqueeze(1).shape
                if self.sparse_conv:
                    #assert self.downsample_resolution == 1 #currently dont know how to downsample a labelled voxel grid
                    if self.refine:
                        input_mask = batch_dict['refine'] == 0
                        target = batch_dict['refine'].clone().unsqueeze(1)
                        target[input_mask.unsqueeze(1)] = 0
                        batch_cond_bits = th.zeros(bs,5,z,y,x)
                        for k in range(0,5):
                            bits_k = th.fmod(target, 2)
                            batch_cond_bits[:,4-k:5-k] = bits_k
                            target = target.div(2, rounding_mode="trunc")
                    
                    elif self.use_bev or self.use_bev_vol:
                        batch_cond_bits = batch_dict['input'].clone().unsqueeze(1)
                    else:
                        input_mask = batch_dict['input'] == 0
                        #target = batch_dict['label'].clone().unsqueeze(1)
                        target = batch_dict['input'].clone().unsqueeze(1)
                        target[input_mask.unsqueeze(1)] = 0
                        bits_cnt = int(log(self.num_classes,2)) + 1
                        batch_cond_bits = th.zeros(bs,bits_cnt,z,y,x)
                        for k in range(0,bits_cnt):
                            bits_k = th.fmod(target, 2)
                            batch_cond_bits[:,bits_cnt-1-k:bits_cnt-k] = bits_k
                            target = target.div(2, rounding_mode="trunc")
                    batch_cond = batch_cond_bits
                
                batch_bev_bits = None
                if self.use_bev: #2D bev
                    target = batch_dict['bev'].clone().unsqueeze(1)
                    bits_cnt = int(log(self.num_classes,2)) + 1
                    batch_bev_bits = th.zeros(bs,bits_cnt,y,x)
                    #print(target.shape, batch_bev_bits.shape, batch_bev_bits[:,bits_cnt-1-1:bits_cnt-1].shape)
                    for k in range(0,bits_cnt):
                        bits_k = th.fmod(target, 2)
                        batch_bev_bits[:,bits_cnt-1-k:bits_cnt-k] = bits_k
                        target = target.div(2, rounding_mode="trunc")
                         
                if self.use_bev_vol: #3D bev
                    target = batch_dict['bev'].clone().type(th.long)
                    label_2_height = th.zeros(12,8).type(th.long).to(target.device)
                    label_2_height[1,:6] = 1 #building
                    label_2_height[2,:7] = 2 #fences
                    label_2_height[3,:3] = 3 #others
                    label_2_height[4,:5] = 4 #pedestrian
                    label_2_height[5,:7] = 5 #pole
                    label_2_height[6,:1] = 6 #road
                    label_2_height[7,:1] = 7 #ground
                    label_2_height[8,:1] = 8 #sidewalk
                    label_2_height[9,:6] = 9 #vegetation
                    label_2_height[10,:6] = 10 #vehicle
                    label_2_height[11,:8] = 11 #unconditioned generation
                    bev_vol = F.embedding(target, label_2_height).permute(0,3,1,2)
                    target = bev_vol.unsqueeze(1)
                    
                    bits_cnt = int(log(self.num_classes,2)) + 1
                    batch_bev_bits = th.zeros(bs,bits_cnt,z,y,x)
                    #print(target.shape, batch_bev_bits.shape, batch_bev_bits[:,bits_cnt-1-1:bits_cnt-1].shape)
                    for k in range(0,bits_cnt):
                        bits_k = th.fmod(target, 2)
                        batch_bev_bits[:,bits_cnt-1-k:bits_cnt-k] = bits_k
                        target = target.div(2, rounding_mode="trunc")
                     
                cond = {'y': batch_cond.detach() if batch_cond is not None else None, 'bev': batch_bev_bits.detach() if batch_bev_bits is not None else None}
                       
                '''if self.use_lidar:
                    cond = {'y': batch_cond.detach(), 'bev': batch_bev_bits.detach() if batch_bev_bits is not None else None}
                else:
                    cond = None '''
            elif self.binary:
                batch = (batch_dict['label'] != 0).unsqueeze(1).type(th.float)
                batch_cond = batch_dict['input'].type(th.float)
                
                if self.downsample_resolution != 1:
                    m = th.nn.MaxPool3d(self.downsample_resolution)
                    #downsampled_label = torch.nn.functional.interpolate(v.unsqueeze(1).type(torch.float), scale_factor=0.5, mode='nearest').squeeze(1)
                    batch = m(batch)          
                    batch_cond = m(batch_cond.unsqueeze(1)).squeeze(1)  
                    
                if self.sparse_conv:
                    assert self.downsample_resolution == 1 #currently dont know how to downsample a labelled voxel grid
                    input_mask = batch_cond == 0
                    batch_cond = batch_dict['label']
                    batch_cond[input_mask] = 0
                            
                if self.use_lidar:
                    cond = {'y': batch_cond.detach()}
                else:
                    cond = None
                #print(batch.shape)
            else:
                batch = batch_dict['feat']
            if self.scale_input:
                batch = batch*2 - 1 # from [0,1] to [-1,1]
                #if cond is not None and 'y' in cond:
                #    cond['y'] = (cond['y'] * 2 - 1).detach()
            batch = batch.detach()
            #x_paths = batch_dict['feat_path']
            x_paths = ""
            
            if self.ignore_invalid:
                invalid = batch_dict['invalid'].clone().type(th.float)
                if self.downsample_resolution != 1:
                    m = th.nn.AvgPool3d(self.downsample_resolution)
                    invalid = (m(invalid) == 1).type(th.float).unsqueeze(1)
            else:
                invalid = None
            self.run_step(batch, cond, x_paths, invalid)
            time_per_step = time.time() - timer
            total_time += time_per_step
            if self.step % self.log_interval == 0:
                logger.logkv("total_time", total_time)
                logger.logkv("time_per_step", total_time/(self.step+1))
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond, x_paths, invalid=None):
        self.forward_backward(batch, cond, x_paths, invalid)
        if self.use_fp16:
            #self.optimize_fp16()
            self.optimize_normal()
        else:
            self.optimize_normal()
        self.log_step()

    def forward_backward(self, batch, cond, x_paths, invalid=None):
        zero_grad(self.model_params)
        
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            #if self.use_lidar:
            micro_cond = {}
            for k, v in cond.items():
                if type(v) == th.Tensor:
                    micro_cond[k] = v[i : i + self.microbatch].to(dist_util.dev())
                elif type(v) == dict:
                    v_dict = {}
                    for kk, vv in v.items():
                        if type(vv) == th.Tensor and len(vv.shape) > 2:
                            v_dict[kk] = vv[i : i + self.microbatch].to(dist_util.dev())
                        elif type(vv) == th.Tensor and len(vv.shape) == 2:
                            # should filter points regard batch_idx, but not implemented yet
                            v_dict[kk] = vv.to(dist_util.dev())
                        elif type(vv) == list:
                            v_dict[kk] = vv[i : i + self.microbatch]
                            
                    micro_cond[k] = v_dict
                    
                elif v == None:
                    micro_cond[k] = v
            #else:
            #    micro_cond = {}
            micro_x_paths = x_paths[i: i + self.microbatch]
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            if invalid is not None:
                invalid = invalid.to(dist_util.dev())

            micro_cond['x_paths'] = micro_x_paths
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
                invalid=invalid,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )
            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )

            if self.use_fp16:
                with amp.scale_loss(loss, self.opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

    def optimize_fp16(self):
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        sqsum = 0.0
        '''for name, param in self.model.named_parameters():
            if param.grad == None:
                print(name, param.shape, param.requires_grad)
        assert False'''
        for p in self.master_params:
            sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        #if self.use_fp16:
        #    logger.logkv("lg_loss_scale", self.lg_loss_scale)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                if self.save_pth:
                    save_pth = self.save_pth
                else:
                    save_pth = get_blob_logdir()
                with bf.BlobFile(bf.join(save_pth, filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            if self.save_pth:
                save_pth = self.save_pth
            else:
                save_pth = get_blob_logdir()
            with bf.BlobFile(
                bf.join(save_pth, f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()

    def _master_params_to_state_dict(self, master_params):
        #if self.use_fp16:
        #    master_params = unflatten_master_params(
        #        self.model.parameters(), master_params
        #    )
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        if self.use_fp16:
            #return make_master_params(params)
            return params
        else:
            return params


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
