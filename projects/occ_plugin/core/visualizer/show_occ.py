
import torch.nn.functional as F
import torch
import numpy as np
from os import path as osp
import os

def save_occ(pred_c, pred_f, img_metas, path, visible_mask=None, gt_occ=None, free_id=0, thres_low=0.4, thres_high=0.99):

    """
    visualization saving for paper:
    1. gt
    2. pred_f pred_c
    3. gt visible
    4. pred_f visible
    """
    if pred_f is not None:
        pred_f = F.softmax(pred_f, dim=1)
        pred_f = pred_f[0].cpu().numpy()  # C W H D
        _, W, H, D = pred_f.shape
    
    if pred_c is not None:
        pred_c = F.softmax(pred_c, dim=1)
        pred_c = pred_c[0].cpu().numpy()  # C W H D
        if pred_f is None:
            _, W, H, D = pred_c.shape
    
    if visible_mask is not None:
        visible_mask = visible_mask[0].cpu().numpy().reshape(-1) > 0  # WHD
    else:
        visible_mask = np.ones(W*H*D, dtype=bool)
        
    if gt_occ is not None:
        gt_occ = gt_occ.data[0][0].cpu().numpy()  # W H D
        gt_occ[gt_occ==255] = 0

    if pred_f is not None:
        coordinates_3D_f = np.stack(np.meshgrid(np.arange(W), np.arange(H), np.arange(D), indexing='ij'), axis=-1).reshape(-1, 3) # (W*H*D, 3)
        pred_f = np.argmax(pred_f, axis=0) # (W, H, D)
        occ_pred_f_mask = (pred_f.reshape(-1))!=free_id
        pred_f_save = np.concatenate([coordinates_3D_f[occ_pred_f_mask], pred_f.reshape(-1)[occ_pred_f_mask].reshape(-1, 1)], axis=1)[:, [2,1,0,3]]  # zyx cls
        pred_f_visible_save = np.concatenate([coordinates_3D_f[occ_pred_f_mask&visible_mask], pred_f.reshape(-1)[occ_pred_f_mask&visible_mask].reshape(-1, 1)], axis=1)[:, [2,1,0,3]]  # zyx cls
    
    if pred_c is not None:
        _, W_c, H_c, D_c = pred_c.shape
        coordinates_3D_c = np.stack(np.meshgrid(np.arange(W_c), np.arange(H_c), np.arange(D_c), indexing='ij'), axis=-1).reshape(-1, 3) # (W*H*D, 3)
        pred_c = np.argmax(pred_c, axis=0) # (W, H, D)
        occ_pred_c_mask = (pred_c.reshape(-1))!=free_id
        pred_c_save = np.concatenate([coordinates_3D_c[occ_pred_c_mask], pred_c.reshape(-1)[occ_pred_c_mask].reshape(-1, 1)], axis=1)[:, [2,1,0,3]]  # zyx cls
    
    scene_token = img_metas.data[0][0]['scene_token']
    lidar_token = img_metas.data[0][0]['lidar_token']
    save_path = osp.join(path, scene_token, lidar_token)
    if not osp.exists(save_path):
        os.makedirs(save_path)
    if pred_f is not None:
        save_pred_f_path = osp.join(save_path, 'pred_f.npy')
        save_pred_f_v_path = osp.join(save_path, 'pred_f_visible.npy')
        np.save(save_pred_f_path, pred_f_save)
        np.save(save_pred_f_v_path, pred_f_visible_save)
    if pred_c is not None:
        save_pred_c_path = osp.join(save_path, 'pred_c.npy')
        np.save(save_pred_c_path, pred_c_save)

    if gt_occ is not None:
        W_gt, H_gt, D_gt = gt_occ.shape
        coordinates_3D_gt = np.stack(np.meshgrid(np.arange(W_gt), np.arange(H_gt), np.arange(D_gt), indexing='ij'), axis=-1).reshape(-1, 3) 
        occ_gt_mask = (gt_occ.reshape(-1))!=free_id
        gt_save = np.concatenate([coordinates_3D_gt[occ_gt_mask], gt_occ.reshape(-1)[occ_gt_mask].reshape(-1, 1)], axis=1)[:, [2,1,0,3]]  # zyx cls
        
        if visible_mask is not None and len(visible_mask) == (W_gt * H_gt * D_gt):
            gt_visible_save = np.concatenate([coordinates_3D_gt[occ_gt_mask&visible_mask], gt_occ.reshape(-1)[occ_gt_mask&visible_mask].reshape(-1, 1)], axis=1)[:, [2,1,0,3]]  # zyx cls
            save_gt_v_path = osp.join(save_path, 'gt_visible.npy')
            np.save(save_gt_v_path, gt_visible_save)
            
        save_gt_path = osp.join(save_path, 'gt.npy')
        np.save(save_gt_path, gt_save)
