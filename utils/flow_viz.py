# Flow visualization code used from https://github.com/tomrunia/OpticalFlow_Visualization


# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-08-03

import numpy as np
from utils.arrow import arrowon
import cv2

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)

def viz_batch_mask_np(imgs, flos, fusions=None, masks=None, arrow_step=32, save_path='tmp.jpg'):
    '''
    input: imgs = [image1, image2]
    fusions = [1->2, 2->1]
    flos = [1->2, 2->1]
    masks = [1->2, 2->1]
    flow_past image2_arrow_past fusion1 mask_past
    flow_future image1_arrow_future fusion2 mask_future
    '''
    image1 = np.array(imgs[0])
    image2 = np.array(imgs[1])

    show_img_past = []
    #show_img_past.append(image1)
    img2_past = arrowon(image1, flos[0], arrow_step)
    show_img_past.append(img2_past)
    show_img_past.append(flow_to_image(flos[0]))
    if fusions is not None:
        show_img_past.append(fusions[0])
    if masks is not None:
        show_img_past.append(np.tile(masks[0]*255, (1, 1, 3)))

    show_img_future = []
    #show_img_future.append(image2)
    img2_past = arrowon(image2, flos[1], arrow_step)
    show_img_future.append(img2_past)
    show_img_future.append(flow_to_image(flos[1]))
    if fusions is not None:
        show_img_future.append(fusions[1])
    if masks is not None:
        show_img_future.append(np.tile(masks[1]*255, (1, 1, 3)))
    
    #img_flo = np.concatenate(show_img, axis=1)
    show_past = np.concatenate(show_img_past, axis=1)
    show_future = np.concatenate(show_img_future, axis=1)
    show_img = np.concatenate([show_past, show_future], axis=0)

    cv2.imwrite(save_path, show_img[:, :, [2,1,0]])

def viz_batch_mask(imgs, fusions, flos, masks, save_path):
    '''
    input: imgs = [image1, image2, image3]
    fusions = [1->2, 3->2]
    flos = [flow_past, flow_future]
    masks = [mask_past, mask_future]
    image1 image2_arrow_past fusion1  flow_past mask_past
    image3 image2_arrow_future fusion2 flow_future mask_future
    '''
    image2 = imgs[1][0].permute(1,2,0).cpu().numpy()

    show_img_past = []
    show_img_past.append(imgs[0][0].permute(1,2,0).cpu().numpy())
    img2_past = arrowon(image2, flos[0][0], 32)
    show_img_past.append(img2_past)
    show_img_past.append(fusions[0][0].permute(1,2,0).cpu().numpy())
    show_img_past.append(flow_to_image(flos[0][0].permute(1,2,0).cpu().numpy()))
    show_img_past.append(np.tile(masks[0][0].permute(1,2,0).cpu().numpy()*255, (1, 1, 3)))

    show_img_future = []
    show_img_future.append(imgs[2][0].permute(1,2,0).cpu().numpy())
    img2_future = arrowon(image2, flos[1][0], 32)
    show_img_future.append(img2_future)
    show_img_future.append(fusions[1][0].permute(1,2,0).cpu().numpy())
    show_img_future.append(flow_to_image(flos[1][0].permute(1,2,0).cpu().numpy()))
    show_img_future.append(np.tile(masks[1][0].permute(1,2,0).cpu().numpy()*255, (1, 1, 3)))
    
    #img_flo = np.concatenate(show_img, axis=1)
    show_past = np.concatenate(show_img_past, axis=1)
    show_future = np.concatenate(show_img_future, axis=1)
    show_img = np.concatenate([show_past, show_future], axis=0)

    cv2.imwrite(save_path, show_img[:, :, [2,1,0]])

def viz_batch(imgs, flo, save_path):
    show_img = []
    for img in imgs:
        img = img[0].permute(1,2,0).cpu().numpy()
        show_img.append(img)

    flo = flo[0].permute(1,2,0).cpu().numpy()
    show_img[0] = arrowon(show_img[0], flo, 32)
    
    # map flow to rgb image
    flo = flow_to_image(flo)
    show_img.append(flo)
    img_flo = np.concatenate(show_img, axis=1)

    cv2.imwrite(save_path, img_flo[:, :, [2,1,0]])
