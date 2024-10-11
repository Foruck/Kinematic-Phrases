import os
import numpy as np
import joblib
import torch
import torch.nn as nn
import smplx
from utils.geometry import canonicalize_smplx

class JOI2KP(nn.Module):
    
    def __init__(self, input_type='smplx'):
        super().__init__()
        self.input_type = input_type.lower()
        meta       = joblib.load('sample/meta_info.pkl')
        
        if self.input_type in ['smplx']:
            JOINT_NAMES = [
                "pelvis",
                "left_hip",
                "right_hip",
                "spine1",
                "left_knee",
                "right_knee",
                "spine2",
                "left_ankle",
                "right_ankle",
                "spine3",
                "left_foot",
                "right_foot",
                "neck",
                "left_collar",
                "right_collar",
                "head",
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow",
                "left_wrist",
                "right_wrist",
                "jaw",
                "left_eye",
                "right_eye",
            ]
            joint_idx = {' '.join(k.split('_')): v for v, k in enumerate(JOINT_NAMES)}
        else:
            raise NotImplementedError
            
        axis_idx  = {
            'ud': 0,
            'rl': 1, 
            'fb': 2,
            'none': 3, 
            'left upper arm': 4,
            'left thigh': 5,
            'right upper arm': 6, 
            'right thigh': 7,
        }
        limbs = {
            'left lower arm': ('left wrist', 'left elbow'), 
            'left upper arm': ('left shoulder', 'left elbow'), 
            'left shank':     ('left foot', 'left knee'), 
            'left thigh':     ('left hip', 'left knee'), 
            'left body':      ('left shoulder', 'left hip'), 
            'right lower arm': ('right wrist', 'right elbow'), 
            'right upper arm': ('right shoulder', 'right elbow'), 
            'right shank':     ('right foot', 'right knee'), 
            'right thigh':     ('right hip', 'right knee'), 
            'right body':      ('right shoulder', 'right hip'), 
            'upper body':      ('pelvis', 'neck'),
        }
        fullLimbs = {
            'left arm':        ('left lower arm', 'left upper arm'), 
            'left leg':        ('left shank', 'left thigh'), 
            'left upper arm':  ('left body', 'left upper arm'),
            'left thigh':      ('upper body', 'left thigh'),
            'right arm':       ('right lower arm', 'right upper arm'), 
            'right leg':       ('right shank', 'right thigh'),
            'right upper arm': ('right body', 'right upper arm'),
            'right thigh':     ('upper body', 'right thigh'),
        }
        idx = [] # each item, j1 index, j2 index, axis index
        for i, info in enumerate(meta['IDX2META']):
            if info[0] == 'pp':
                # e.g.
                # (left hand, ud)
                # 1: left hand moves upwards
                # -1: left hand moves downwards
                part, ax = info[1] # 1: part moving upwards ()
                idx.append([joint_idx[part], 0, axis_idx[ax]])
            elif info[0] == 'pdp': 
                # e.g.
                # (left hand, right hand)
                # 1: lhand and rhand moves away from each other
                # -1: lhand and rhand moves closer
                ja, jb = info[1]
                idx.append([joint_idx[ja], joint_idx[jb], axis_idx['none']])
            elif info[0] in ['prpp', 'lop']:
                # e.g.
                # (left hand, right hand, ud)
                # 1: lhand above rhand
                # -1: lhand below rhand
                ja, jb, ax = info[1]
                idx.append([joint_idx[ja], joint_idx[jb], axis_idx[ax]])
            elif info[0] == 'lap':
                # e.g.
                # (left arm)
                # 1: left arm unbends
                # -1: left arm bends
                fLimb = fullLimbs[info[1]]
                idx.append([joint_idx[limbs[fLimb[0]][0]], joint_idx[limbs[fLimb[0]][1]], axis_idx[fLimb[1]]])
        print(len(idx))

        self.idx = np.array(idx)
        self.joint_idx = joint_idx
    
    def forward(self, joi, index=None):
        axis = torch.zeros(joi.shape[0], 8, 3, device=joi.device)
        axis[:, 0, -1] = 1. # ud
        axis[:, 1] = joi[:, self.joint_idx['right hip']] - joi[:, self.joint_idx['left hip']] # rl
        axis[:, 2] = torch.cross(axis[:, 0], axis[:, 1]) # fb
        axis[:, 3] = 1. # none
        axis[:, 4] = joi[:, self.joint_idx['left shoulder']]  - joi[:, self.joint_idx['left elbow']]  # left upper arm
        axis[:, 5] = joi[:, self.joint_idx['left hip']]    - joi[:, self.joint_idx['left knee']]   # left thigh
        axis[:, 6] = joi[:, self.joint_idx['right shoulder']] - joi[:, self.joint_idx['right elbow']] # right upper arm
        axis[:, 7] = joi[:, self.joint_idx['right hip']]   - joi[:, self.joint_idx['right knee']]  # right thigh
        axis = axis / torch.norm(axis, p=2, dim=2, keepdim=True)
        if index is None:
            ind1 = torch.sum((joi[:, self.idx[:381, 0]] - joi[:, self.idx[:381, 1]]) * axis[:, self.idx[:381, 2]], axis=-1)
            ind2 = torch.arccos(torch.sum((joi[:, self.idx[381:, 0]] - joi[:, self.idx[381:, 1]]) * axis[:, self.idx[381:, 2]] / (torch.norm((joi[:, self.idx[381:, 0]] - joi[:, self.idx[381:, 1]]), dim=2, p=2, keepdim=True) + 1e-8), dim=-1))
            ind3 = torch.sum((joi[1:, [0, 0, 0]] - joi[:-1, [0, 0, 0]]) * axis[:-1, :3], dim=-1)
            ind3 = torch.cat([ind3, ind3[-1:]])
            indicators = torch.cat([ind1, ind2, ind3], axis=1)
            indicators[1:, :115] = torch.diff(indicators[:, :115], axis=0)
            indicators[:1, :115] = indicators[1:2, :115]
            indicators[1:, 381:389]  = torch.diff(indicators[:, 381:389], axis=0)
            indicators[:1, 381:389]  = indicators[1:2, 381:389]
            indicators[torch.abs(indicators) < 1e-3] = 0
            indicators = torch.sign(indicators)
            return indicators
        else:
            if index < 381:
                indicator = torch.sum((joi[:, self.idx[index, 0]] - joi[:, self.idx[index, 1]]) * axis[:, self.idx[index, 2]], axis=-1)
            elif index < 389:
                x1 = joi[:, self.idx[index, 0]] - joi[:, self.idx[index, 1]]
                x2 = axis[:, self.idx[index, 2]]
                cos = torch.clip(torch.nn.functional.cosine_similarity(x1, x2, dim=1, eps=1e-8), -1, 1)
                indicator = torch.arccos(cos)
            else:
                indicator = torch.sum((joi[1:, 0] - joi[:-1, 0]) * axis[:-1, index - 389], dim=-1)
            if index < 115 or 389 > index > 381:
                indicator = torch.diff(indicator)
            indicator[torch.abs(indicator) < 1e-3] = 0
            indicator = torch.sign(indicator)
            return indicator

if __name__ == '__main__':
    body_model = smplx.create('models', model_type='smplx', gender='neutral', use_face_contour=True, num_betas=16, num_expression_coeffs=10, ext='npz', use_pca=False, create_global_orient=False, create_body_pose=False, create_left_hand_pose=False, create_right_hand_pose=False, create_jaw_pose=False, create_leye_pose=False, create_reye_pose=False, create_betas=False, create_expression=False, create_transl=False,).double()
    
    data  = joblib.load('sample/motion.pkl')
    nf    = len(data['poses'])
    betas = np.concatenate([data['betas'], np.zeros(16 - len(data['betas']))])
    betas      = torch.from_numpy(betas).expand(nf, -1).double()
    expression = torch.zeros(nf, 10).double()
    pose  = torch.from_numpy(data['poses']).double()
    trans = torch.from_numpy(data['trans']).double()
    trans[1:] = trans[1:] - trans[:-1]
    trans[0]  = 0
    pose, trans = canonicalize_smplx(pose.reshape(1, nf, 55, 3), 'aa', trans[None], 'aa')
    pose  = pose[0].flatten(1)
    trans = trans[0].cumsum(0)
    smplx_data = body_model(betas=betas, expression=expression, transl=trans,
                    global_orient=pose[..., :3], body_pose=pose[..., 3:66], jaw_pose=pose[..., 66:69], 
                    leye_pose=pose[..., 69:72], reye_pose=pose[..., 72:75], 
                    left_hand_pose=pose[..., 75:120], right_hand_pose=pose[..., 120:165], 
                    return_verts=False, return_shaped=False, dense_verts=False)
    joints = smplx_data.joints.detach().cpu().squeeze()
    kp = JOI2KP()
    print(kp(joints).shape)