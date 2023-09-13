import os
import cv2
import numpy as np
import torch
from flow_colors import flow_to_color
from geometry import *
from ctypes import *
import ctypes
import random
import math

lib = cdll.LoadLibrary("external/forward_warping/libwarping.so")
warp = lib.forward_warping


def moveing_object_with_mask(depth_path, disp, rgb, K, inv_K, instance_mask, i):

    # Cast I0 and D0 to pytorch tensors
    h, w = rgb.shape[:2]
    rgb = torch.from_numpy(np.expand_dims(rgb, 0)).float().cuda()
    # depth = torch.from_numpy(np.expand_dims(depth, 0)).float().cuda()
    
    # debug
    # depth = cv2.imread(depth_path, -1) / (2**16-1)
    # if depth.shape[0] != h or depth.shape[1] != w:
    #     depth = cv2.resize(depth, (w, h))

    # Get depth map and normalize
    depth = 1.0 / (disp[0] + 0.005)
    depth[depth > 100] = 100
    # depth = torch.from_numpy(np.expand_dims(depth, 0)).float().cuda()
    
    instance_mask = instance_mask[0]
    instance_mask = torch.stack([instance_mask, instance_mask], -1)

    # Create objects in charge of 3D projection
    backproject_depth = BackprojectDepth(1, h, w).cuda()
    project_3d = Project3D(1, h, w).cuda()

    # Prepare p0 coordinates
    meshgrid = np.meshgrid(range(w), range(h), indexing="xy")
    p0 = np.stack(meshgrid, axis=-1).astype(np.float32)

    # Initiate masks dictionary
    masks = {}
    axisangle = torch.from_numpy(np.array([[[0, 0, 0]]], dtype=np.float32)).cuda()
    translation = torch.from_numpy(np.array([[0, 0, 0]])).cuda()

    # Compute (R|t)
    T1 = transformation_from_parameters(axisangle, translation)
    
    temp = torch.zeros((1, 4, 4)).cuda()
    temp[0, -1, -1] = 1.
    temp[:, :3, :3] = K
    K = temp
    
    temp = torch.zeros((1, 4, 4)).cuda()
    temp[0, -1, -1] = 1.
    temp[:, :3, :3] = inv_K
    inv_K = temp

    # Back-projection
    cam_points = backproject_depth(depth, inv_K)

    # Apply transformation T_{0->1}
    p1, z1 = project_3d(cam_points, K, T1)
    z1 = z1.reshape(1, h, w)

   # Simulate objects moving independently
    if True:

        sign = -1

        # Random t (scalars and signs). Zeros and small motions are avoided as before
        # cix = (random.random()*0.05+0.05) * \
        #     (sign*(-1)**random.randrange(2))
        # ciy = (random.random()*0.05+0.05) * \
        #     (sign*(-1)**random.randrange(2))
        # ciz = (random.random()*0.05+0.05) * \
        #     (sign*(-1)**random.randrange(2))
        cix = (random.random()*0.05+0.05) 
        ciy = -1*(random.random()*0.05+0.05) 
        ciz = (random.random()*0.05+0.05) 
        camerai_mot = [cix, ciy, ciz]

        # Random Euler angles (scalars and signs). Zeros and small rotations are avoided as before
        aix = (random.random()*math.pi / 72.0 + math.pi /
               72.0) * (sign*(-1)**random.randrange(2))
        aiy = (random.random()*math.pi / 72.0 + math.pi /
               72.0) * (sign*(-1)**random.randrange(2))
        aiz = (random.random()*math.pi / 72.0 + math.pi /
               72.0) * (sign*(-1)**random.randrange(2))
        camerai_ang = [aix, aiy, aiz]
        camerai_ang = [0, 0, 0]

        ai = torch.from_numpy(
            np.array([[camerai_ang]], dtype=np.float32)).cuda()
        tri = torch.from_numpy(np.array([[camerai_mot]])).cuda()

        # Compute (R|t)
        Ti = transformation_from_parameters(
            axisangle + ai, translation + tri)

        # Apply transformation T_{0->\pi_i}
        pi, zi = project_3d(cam_points, K, Ti)

        # If a pixel belongs to object label l, replace coordinates in I1...
        p1[instance_mask > 0] = pi[instance_mask > 0]

        # ... and its depth
        zi = zi.reshape(1, h, w)
        z1[instance_mask[:, :, :, 0] > 0] = zi[instance_mask[:, :, :, 0] > 0]

    # Bring p1 coordinates in [0,W-1]x[0,H-1] format
    p1 = (p1 + 1) / 2
    p1[:, :, :, 0] *= w - 1
    p1[:, :, :, 1] *= h - 1

    # Create auxiliary data for warping
    dlut = torch.ones(1, h, w).float().cuda() * 1000
    safe_y = np.maximum(np.minimum(p1[:, :, :, 1].cpu().long(), h - 1), 0)
    safe_x = np.maximum(np.minimum(p1[:, :, :, 0].cpu().long(), w - 1), 0)
    warped_arr = np.zeros(h*w*5).astype(np.uint8)
    img = rgb.reshape(-1).to(torch.uint8)

    # Call forward warping routine (C code)
    warp(c_void_p(img.cpu().numpy().ctypes.data), c_void_p(safe_x[0].cpu().numpy().ctypes.data),
         c_void_p(safe_y[0].cpu().numpy().ctypes.data), c_void_p(z1.reshape(-1).cpu().numpy().ctypes.data), 
         c_void_p(warped_arr.ctypes.data), c_int(h), c_int(w))
    warped_arr = warped_arr.reshape(1, h, w, 5).astype(np.uint8)

    # Warped image
    im1_raw = warped_arr[0, :, :, 0:3]

    # Validity mask H
    masks["H"] = warped_arr[0, :, :, 3:4]

    # Collision mask M
    masks["M"] = warped_arr[0, :, :, 4:5]
    # Keep all pixels that are invalid (H) or collide (M)
    masks["M"] = 1-(masks["M"] == masks["H"]).astype(np.uint8)

    # Dilated collision mask M'
    kernel = np.ones((3, 3), np.uint8)
    masks["M'"] = cv2.dilate(masks["M"], kernel, iterations=1)
    masks["P"] = (np.expand_dims(masks["M'"], -1)
                  == masks["M"]).astype(np.uint8)

    # Final mask P
    masks["H'"] = masks["H"]*masks["P"]

    # Compute flow as p1-p0
    flow_01 = p1.cpu().numpy() - p0
    im1 = rgb[0].cpu().numpy().copy()
    # mask_idx = np.logical_and(
    #     flow_01[0, :, :, 0] > 1,
    #     flow_01[0, :, :, 1] > 1
    # )
    mask_idx = np.where(instance_mask[0, :, :, 0].cpu().numpy())
    # mask_xp = mask_x, mask_y

    im1 = cv2.inpaint(im1_raw, 1 - masks["H"], 3, cv2.INPAINT_TELEA)
    flow_color = flow_to_color(flow_01[0], convert_to_bgr=True)
    mask = cv2.merge([masks["H"]*255, masks["H"]*255, masks["H"]*255])
    res = np.vstack(
        [rgb[0].cpu().numpy(), im1, im1_raw, mask, flow_color]
    )
    cv2.imwrite('temp/res-{:06d}.png'.format(i), res)