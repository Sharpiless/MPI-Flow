import os
import math
from PIL import Image
import cv2
from tqdm import tqdm
import torch
from torch import optim
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from bilateral_filter import sparse_bilateral_filtering

# from moviepy.editor import ImageSequenceClip
from flow_colors import flow_to_color
import random
from geometry import transformation_from_parameters
from moving_obj import moveing_object_with_mask

from utils.mpi import mpi_rendering
from utils.mpi.homography_sampler import HomographySample
from write_flow import writeFlow


def arrowon(img, flow, step=32):
    h, w = flow.shape[:2]
    img2 = img.copy()
    for i in range(step // 2, h, step):
        for j in range(step // 2, w, step):
            dstx, dsty = int(i + flow[i, j, 1]), int(j + flow[i, j, 0])
            cv2.arrowedLine(img2, (j, i), (dsty, dstx), (0, 0, 255), 2, 8, 0, 0.2)
    return img2


def image_to_tensor(img_path, unsqueeze=True):
    rgb = transforms.ToTensor()(Image.open(img_path).convert("RGB"))
    if unsqueeze:
        rgb = rgb.unsqueeze(0)
    return rgb


def disparity_to_tensor(disp_path, unsqueeze=True):
    disp = cv2.imread(disp_path, 0) / 255
    # import IPython
    # IPython.embed()
    # exit()
    # disp = np.clip(disp, 0, 0.8)
    # disp = sparse_bilateral_filtering(disp, filter_size=[5, 5], num_iter=2)
    disp = torch.from_numpy(disp)[None, ...]
    if unsqueeze:
        disp = disp.unsqueeze(0)
    return disp.float()


def gen_swing_path(num_frames=90, r_x=0.14, r_y=0.0, r_z=0.10):
    "Return a list of matrix [4, 4]"
    t = torch.arange(num_frames) / (num_frames - 1)
    poses = torch.eye(4).repeat(num_frames, 1, 1)
    poses[:, 0, 3] = r_x * torch.sin(2.0 * math.pi * t)
    poses[:, 1, 3] = r_y * torch.cos(2.0 * math.pi * t)
    poses[:, 2, 3] = r_z * (torch.cos(2.0 * math.pi * t) - 1.0)
    return poses.unbind()


def render_novel_view(
    mpi_all_rgb_src,
    mpi_all_sigma_src,
    disparity_all_src,
    G_tgt_src,
    K_src_inv,
    K_tgt,
    K_src,
    src_pose,
    homography_sampler,
    hard_flow=False,
):
    xyz_src_BS3HW = mpi_rendering.get_src_xyz_from_plane_disparity(
        homography_sampler.meshgrid, disparity_all_src, K_src_inv
    )

    xyz_tgt_BS3HW = mpi_rendering.get_tgt_xyz_from_plane_disparity(
        xyz_src_BS3HW, G_tgt_src
    )

    mpi_depth_src = torch.reciprocal(disparity_all_src)
    B, S = disparity_all_src.size()
    xyz_tgt = xyz_tgt_BS3HW.reshape(B * S, 3, -1) / mpi_depth_src[0].unsqueeze(
        1
    ).unsqueeze(2)
    # BSx3xHW torch.Size([64, 3, 98304])
    meshgrid_tgt = torch.matmul(K_tgt, xyz_tgt)
    meshgrid_src = (
        homography_sampler.meshgrid.unsqueeze(0)
        .unsqueeze(1)
        .repeat(B, S, 1, 1, 1)
        .reshape(B * S, 3, -1)
    )
    mpi_flow_src = meshgrid_src - meshgrid_tgt
    H, W = mpi_all_rgb_src.shape[-2:]
    mpi_flow_src = mpi_flow_src.reshape(B, S, 3, H, W)[:, :, :2]

    tgt_imgs_syn, tgt_depth_syn, _, flow_syn = mpi_rendering.render_tgt_rgb_depth(
        homography_sampler,
        mpi_all_rgb_src,
        mpi_all_sigma_src,
        disparity_all_src,
        xyz_tgt_BS3HW,
        xyz_src_BS3HW,
        G_tgt_src,
        K_src_inv,
        K_tgt,
        mpi_flow_src,
        use_alpha=False,
        is_bg_depth_inf=False,
        hard_flow=hard_flow,
    )

    return tgt_imgs_syn, tgt_depth_syn, flow_syn


def generate_random_pose(ext_cz, base_motions=[0.1, 0.1, 0.1]):
    scx = (-1) ** random.randrange(2)
    scy = (-1) ** random.randrange(2)
    scz = (-1) ** random.randrange(2)
    if base_motions[0] == 0.1:
        scz = -1  # most cameras move forward in kitti
    else:
        scx = scx * 0.5  # object motion
        scy = scy * 0.5
        scz = scz * 0.5

    # Random scalars excluding zeros / very small motions
    cx = (random.random() * 0.1 + base_motions[0]) * scx
    cy = (random.random() * 0.1 + base_motions[1]) * scy
    cz = (random.random() * ext_cz + base_motions[2]) * scz
    camera_mot = [cx, cy, cz]

    # generate random triplet of Euler angles
    # Random sign
    sax = (-1) ** random.randrange(2)
    say = (-1) ** random.randrange(2)
    saz = (-1) ** random.randrange(2)
    
    # Random angles in -pi/18,pi/18, excluding -pi/36,pi/36 to avoid zeros / very small rotations
    ax = (random.random() * math.pi / 36.0) * sax
    ay = (random.random() * math.pi / 36.0) * say
    az = (random.random() * math.pi / 36.0) * saz
    camera_ang = [ax * 0.4, ay * 0.4, az * 0.4]

    axisangle = (
        torch.from_numpy(np.array([[camera_ang]], dtype=np.float32)).cuda().float()
    )
    translation = torch.from_numpy(np.array([[camera_mot]])).cuda().float()

    cam_ext = transformation_from_parameters(axisangle, translation)[0]
    return cam_ext


def render_3dphoto_dynamic(
    opt,
    src_imgs,  # [b,3,h,w]
    obj_mask,
    disp,
    mpi_all_src,
    disparity_all_src,
    k_src,  # [b,3,3]
    k_tgt,  # [b,3,3]
    data_path=None,
    name=None,
    hard_flow=False,
    mask_thresh=0.99,
):
    name = name.split(".")[0]
    src_np = src_imgs[0].permute(1, 2, 0).contiguous().cpu().numpy()  # [b,h,w,3]
    src_np = np.clip(np.round(src_np * 255), a_min=0, a_max=255).astype(np.uint8)[
        :, :, [2, 1, 0]
    ]

    h, w = mpi_all_src.shape[-2:]
    swing_path_list = gen_swing_path()
    src_pose = swing_path_list[0]
    obj_mask_np = obj_mask.squeeze().cpu().numpy()
    # preprocess the predict MPI
    device = mpi_all_src.device
    homography_sampler = HomographySample(h, w, device)
    k_src_inv = torch.inverse(k_src.to(torch.float64).cpu())
    k_src_inv = k_src_inv.cuda().to(k_src.dtype)
    mpi_all_rgb_src = mpi_all_src[:, :, 0:3, :, :]  # BxSx3xHxW
    mpi_all_sigma_src = mpi_all_src[:, :, 3:, :, :]  # BxSx1xHxW
    xyz_src_BS3HW = mpi_rendering.get_src_xyz_from_plane_disparity(
        homography_sampler.meshgrid.to(k_src.dtype),
        disparity_all_src,
        k_src_inv,
    )
    _, _, blend_weights, _, _, _ = mpi_rendering.render(
        mpi_all_rgb_src,
        mpi_all_sigma_src,
        xyz_src_BS3HW,
        use_alpha=False,
        is_bg_depth_inf=False,
    )
    mpi_all_rgb_src = (
        blend_weights * src_imgs.unsqueeze(1) + (1 - blend_weights) * mpi_all_rgb_src
    )

    for idx in range(1):
        cam_ext_dynamic = generate_random_pose(opt.ext_cz)
        cam_ext = generate_random_pose(opt.ext_cz, base_motions=[0, 0, 0])

        frame, depth, flowA2B, mask = render_novel_view_dynamic(
            obj_mask,
            mpi_all_rgb_src,
            mpi_all_sigma_src,
            disparity_all_src,
            cam_ext.cuda(),
            k_src_inv,
            k_tgt,
            k_src,
            src_pose,
            homography_sampler,
            hard_flow,
        )

        frame_dync, depth_dync, flowA2B_dync, mask_dync = render_novel_view_dynamic(
            1 - obj_mask,
            mpi_all_rgb_src,
            mpi_all_sigma_src,
            disparity_all_src,
            cam_ext_dynamic.cuda(),
            k_src_inv,
            k_tgt,
            k_src,
            src_pose,
            homography_sampler,
            hard_flow,
        )
        frame_np = (
            frame[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
        )  # [b,h,w,3]
        frame_np = np.clip(np.round(frame_np * 255), a_min=0, a_max=255).astype(
            np.uint8
        )[:, :, [2, 1, 0]]

        frame_dync_np = (
            frame_dync[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
        )  # [b,h,w,3]
        frame_dync_np = np.clip(
            np.round(frame_dync_np * 255), a_min=0, a_max=255
        ).astype(np.uint8)[:, :, [2, 1, 0]]
        mask = (
            mask[0].permute(1, 2, 0).cpu().squeeze().numpy().astype(np.float32)
        )  # [b,h,w,3]
        mask_dync = (
            mask_dync[0].permute(1, 2, 0).cpu().squeeze().numpy().astype(np.float32)
        )  # [b,h,w,3]

        flow_np = (
            flowA2B[0].permute(1, 2, 0).contiguous().cpu().numpy().astype(np.float32)
        )  # [b,h,w,3]
        flow_dync_np = (
            flowA2B_dync[0]
            .permute(1, 2, 0)
            .contiguous()
            .cpu()
            .numpy()
            .astype(np.float32)
        )  # [b,h,w,3]

        # A2B 光流 mask
        flow_np[obj_mask_np < mask_thresh] = 0
        flow_dync_np[obj_mask_np >= mask_thresh] = 0

        frame_np[mask < mask_thresh] = 255
        frame_dync_np[mask_dync < mask_thresh] = 255
        frame_mix = frame_dync_np.copy()
        frame_mix[mask >= mask_thresh] = frame_np[mask >= mask_thresh]
        flow_mix = flow_dync_np.copy()
        flow_mix[obj_mask_np >= mask_thresh] = flow_np[obj_mask_np >= mask_thresh]

        fill_mask = mask_dync.copy()
        fill_mask[mask >= mask_thresh] = 1

        fill_mask = (fill_mask < mask_thresh).astype(np.int32)
        inpainted = cv2.inpaint(
            frame_mix, fill_mask.astype(np.uint8), 3, cv2.INPAINT_NS
        )
        res = None
        return flow_mix, src_np, inpainted, res


def render_novel_view_dynamic(
    obj_mask,
    mpi_all_rgb_src,
    mpi_all_sigma_src,
    disparity_all_src,
    G_tgt_src,
    K_src_inv,
    K_tgt,
    K_src,
    src_pose,
    homography_sampler,
    hard_flow=False,
):
    xyz_src_BS3HW = mpi_rendering.get_src_xyz_from_plane_disparity(
        homography_sampler.meshgrid.to(K_src.dtype), disparity_all_src, K_src_inv
    )

    xyz_tgt_BS3HW = mpi_rendering.get_tgt_xyz_from_plane_disparity(
        xyz_src_BS3HW.to(K_src.dtype), G_tgt_src.to(K_src.dtype)
    )

    mpi_depth_src = torch.reciprocal(disparity_all_src)
    B, S = disparity_all_src.size()
    xyz_tgt = xyz_tgt_BS3HW.reshape(B * S, 3, -1) / mpi_depth_src[0].unsqueeze(
        1
    ).unsqueeze(2)
    # BSx3xHW torch.Size([64, 3, 98304])
    meshgrid_tgt = torch.matmul(K_tgt, xyz_tgt)
    meshgrid_src = (
        homography_sampler.meshgrid.unsqueeze(0)
        .unsqueeze(1)
        .repeat(B, S, 1, 1, 1)
        .reshape(B * S, 3, -1)
    )
    mpi_flow_src = meshgrid_src - meshgrid_tgt
    H, W = mpi_all_rgb_src.shape[-2:]
    mpi_flow_src = mpi_flow_src.reshape(B, S, 3, H, W)[:, :, :2]
    obj_mask = obj_mask.unsqueeze(1).repeat(B, S, 1, 1, 1)

    tgt_imgs_syn, tgt_depth_syn, _, flow_syn, obj_mask = (
        mpi_rendering.render_tgt_rgb_depth(
            homography_sampler,
            mpi_all_rgb_src,
            mpi_all_sigma_src,
            disparity_all_src,
            xyz_tgt_BS3HW,
            xyz_src_BS3HW,
            G_tgt_src,
            K_src_inv,
            K_tgt,
            mpi_flow_src,
            use_alpha=False,
            is_bg_depth_inf=False,
            hard_flow=hard_flow,
            obj_mask=obj_mask,
        )
    )
    flow_syn = torch.clip(flow_syn, -200, 200)
    return tgt_imgs_syn, tgt_depth_syn, flow_syn, obj_mask