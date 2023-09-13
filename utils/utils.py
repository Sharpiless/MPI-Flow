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
    for i in range(step//2, h, step):
        for j in range(step//2, w, step):
            dstx, dsty = int(i+flow[i, j, 1]), int(j+flow[i, j, 0])
            cv2.arrowedLine(img2, (j, i), (dsty, dstx),
                            (0, 0, 255), 2, 8, 0, 0.2)
    return img2


def image_to_tensor(img_path, unsqueeze=True):
    rgb = transforms.ToTensor()(Image.open(img_path))
    if unsqueeze:
        rgb = rgb.unsqueeze(0)
    return rgb


def disparity_to_tensor(disp_path, unsqueeze=True):
    disp = cv2.imread(disp_path, -1) / (2 ** 16 - 1)
    # import IPython
    # IPython.embed()
    # exit()
    # disp = np.clip(disp, 0, 0.8)
    # disp = sparse_bilateral_filtering(disp, filter_size=[5, 5], num_iter=2)
    disp = torch.from_numpy(disp)[None, ...]
    if unsqueeze:
        disp = disp.unsqueeze(0)
    return disp.float()


def gen_swing_path(num_frames=90, r_x=0.14, r_y=0., r_z=0.10):
    "Return a list of matrix [4, 4]"
    t = torch.arange(num_frames) / (num_frames - 1)
    poses = torch.eye(4).repeat(num_frames, 1, 1)
    poses[:, 0, 3] = r_x * torch.sin(2. * math.pi * t)
    poses[:, 1, 3] = r_y * torch.cos(2. * math.pi * t)
    poses[:, 2, 3] = r_z * (torch.cos(2. * math.pi * t) - 1.)
    return poses.unbind()


def render_3dphoto(
    src_imgs,  # [b,3,h,w]
    mpi_all_src,  # [b,s,4,h,w]
    disparity_all_src,  # [b,s]
    k_src,  # [b,3,3]
    k_tgt,  # [b,3,3]
    save_path,
    out_path='outputs',
):
    h, w = mpi_all_src.shape[-2:]
    device = mpi_all_src.device
    homography_sampler = HomographySample(h, w, device)
    k_src_inv = torch.inverse(k_src.cpu())
    k_src_inv = k_src_inv.cuda()

    # preprocess the predict MPI
    xyz_src_BS3HW = mpi_rendering.get_src_xyz_from_plane_disparity(
        homography_sampler.meshgrid,
        disparity_all_src,
        k_src_inv,
    )
    mpi_all_rgb_src = mpi_all_src[:, :, 0:3, :, :]  # BxSx3xHxW
    mpi_all_sigma_src = mpi_all_src[:, :, 3:, :, :]  # BxSx1xHxW
    _, _, blend_weights, _, _, _ = mpi_rendering.render(
        mpi_all_rgb_src,
        mpi_all_sigma_src,
        xyz_src_BS3HW,
        use_alpha=False,
        is_bg_depth_inf=False,
    )
    mpi_all_rgb_src = blend_weights * \
        src_imgs.unsqueeze(1) + (1 - blend_weights) * mpi_all_rgb_src
    # mpi_all_rgb_src torch.Size([1, 64, 3, 256, 384])
    # render novel views
    swing_path_list = gen_swing_path()
    frames = []
    src_img_np = src_imgs[0].permute(
        1, 2, 0).contiguous().cpu().numpy()  # [b,h,w,3]
    src_img_np = np.clip(np.round(src_img_np * 255),
                         a_min=0, a_max=255).astype(np.uint8)
    src_pose = swing_path_list[0]
    for i, cam_ext in enumerate(tqdm(swing_path_list)):
        frame, depth, flow = render_novel_view(
            mpi_all_rgb_src,
            mpi_all_sigma_src,
            disparity_all_src,
            cam_ext.cuda(),
            k_src_inv,
            k_tgt,
            k_src,
            src_pose,
            homography_sampler,
        )
        frame_np = frame[0].permute(
            1, 2, 0).contiguous().cpu().numpy()  # [b,h,w,3]
        flow_np = flow[0].permute(
            1, 2, 0).contiguous().cpu().numpy()  # [b,h,w,3]
        flow_color = flow_to_color(flow_np, convert_to_bgr=True)
        # import IPython
        # IPython.embed()
        # exit()
        frame_np = np.clip(np.round(frame_np * 255),
                           a_min=0, a_max=255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_path, 'image',
                    '{:05d}.png'.format(i)), frame_np[:, :, [2, 1, 0]])
        cv2.imwrite(os.path.join(out_path, 'flow',
                    '{:05d}.png'.format(i)), flow_color)
        np.save(os.path.join(out_path, 'flow_npy',
                '{:05d}.npy'.format(i)), flow_np)
        frame_np = np.hstack([src_img_np, frame_np, flow_color])
        frames.append(frame_np)
    cv2.imwrite(os.path.join(out_path, 'src_image.png'.format(i)),
                src_img_np[:, :, [2, 1, 0]])
    rgb_clip = ImageSequenceClip(frames, fps=10)
    rgb_clip.write_videofile(save_path, verbose=False,
                             codec='mpeg4', logger=None, bitrate='2000k')


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
    hard_flow=False
):
    xyz_src_BS3HW = mpi_rendering.get_src_xyz_from_plane_disparity(
        homography_sampler.meshgrid,
        disparity_all_src,
        K_src_inv
    )

    xyz_tgt_BS3HW = mpi_rendering.get_tgt_xyz_from_plane_disparity(
        xyz_src_BS3HW,
        G_tgt_src
    )

    mpi_depth_src = torch.reciprocal(disparity_all_src)
    B, S = disparity_all_src.size()
    xyz_tgt = xyz_tgt_BS3HW.reshape(
        B * S, 3, -1) / mpi_depth_src[0].unsqueeze(1).unsqueeze(2)
    # BSx3xHW torch.Size([64, 3, 98304])
    meshgrid_tgt = torch.matmul(K_tgt, xyz_tgt)
    meshgrid_src = homography_sampler.meshgrid.unsqueeze(
        0).unsqueeze(1).repeat(B, S, 1, 1, 1).reshape(B * S, 3, -1)
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
        hard_flow=hard_flow
    )

    return tgt_imgs_syn, tgt_depth_syn, flow_syn


def render_3dphoto_single(
    src_imgs,  # [b,3,h,w]
    mpi_all_src,  # [b,s,4,h,w]
    disparity_all_src,  # [b,s]
    k_src,  # [b,3,3]
    k_tgt,  # [b,3,3]
    save_path,
    out_path='outputs',
    single=False,
    data_path=None,
    name=None,
    hard_flow=False
):
    h, w = mpi_all_src.shape[-2:]
    device = mpi_all_src.device
    homography_sampler = HomographySample(h, w, device)
    k_src_inv = torch.inverse(k_src.to(torch.float64).cpu())
    k_src_inv = k_src_inv.cuda().to(k_src.dtype)

    # preprocess the predict MPI
    xyz_src_BS3HW = mpi_rendering.get_src_xyz_from_plane_disparity(
        homography_sampler.meshgrid,
        disparity_all_src,
        k_src_inv,
    )
    mpi_all_rgb_src = mpi_all_src[:, :, 0:3, :, :]  # BxSx3xHxW
    mpi_all_sigma_src = mpi_all_src[:, :, 3:, :, :]  # BxSx1xHxW
    _, _, blend_weights, _, _, _ = mpi_rendering.render(
        mpi_all_rgb_src,
        mpi_all_sigma_src,
        xyz_src_BS3HW,
        use_alpha=False,
        is_bg_depth_inf=False,
    )
    mpi_all_rgb_src = blend_weights * \
        src_imgs.unsqueeze(1) + (1 - blend_weights) * mpi_all_rgb_src
    # mpi_all_rgb_src torch.Size([1, 64, 3, 256, 384])
    # render novel views
    # swing_path_list = gen_swing_path(num_frames=90, r_x=0, r_y=0, r_z=0.3)
    swing_path_list = gen_swing_path()
    frames = []
    src_img_np = src_imgs[0].permute(
        1, 2, 0).contiguous().cpu().numpy()  # [b,h,w,3]
    src_img_np = np.clip(np.round(src_img_np * 255),
                         a_min=0, a_max=255).astype(np.uint8)
    src_pose = swing_path_list[0]
    poses = torch.stack(swing_path_list).cpu().numpy()
    np.save('poses.npy', poses)
    # import IPython
    # IPython.embed()
    # exit()
    if single:
        rand_idx = np.random.randint(low=1, high=len(swing_path_list)-1)
        swing_path_list = [swing_path_list[rand_idx]]
    for i, cam_ext in enumerate(tqdm(swing_path_list)):
        frame, depth, flowA2B = render_novel_view(
            mpi_all_rgb_src,
            mpi_all_sigma_src,
            disparity_all_src,
            cam_ext.cuda(),
            k_src_inv,
            k_tgt,
            k_src,
            src_pose,
            homography_sampler,
            hard_flow
        )
        frame_np = frame[0].permute(
            1, 2, 0).contiguous().cpu().numpy()  # [b,h,w,3]
        flow_np = flowA2B[0].permute(
            1, 2, 0).contiguous().cpu().numpy()  # [b,h,w,3]
        flow_color = flow_to_color(flow_np, convert_to_bgr=True)
        frame_np = np.clip(np.round(frame_np * 255),
                           a_min=0, a_max=255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_path, 'image',
                                 '{:05d}.png'.format(i)), frame_np[:, :, [2, 1, 0]])
        cv2.imwrite(os.path.join(out_path, 'flow',
                                 '{:05d}.png'.format(i)), flow_color)
        src_img_np_ = arrowon(src_img_np.copy(), flow_np)
        frame_np = np.vstack([src_img_np_, frame_np, flow_color])
        np.save(os.path.join(out_path, 'flow_np',
                             '{:05d}.npy'.format(i)), flow_np)
        # frame_np = np.hstack([src_img_np_, frame_np, flow_color])
        frames.append(frame_np)
    if not single:
        cv2.imwrite(os.path.join(out_path, 'src_image.png'.format(i)),
                    src_img_np[:, :, [2, 1, 0]])
        rgb_clip = ImageSequenceClip(frames, fps=10)
        rgb_clip.write_videofile(save_path, verbose=False,
                                 codec='mpeg4', logger=None, bitrate='2000k')


def render_novel_view_flow_only(
    mpi_all_rgb_src,
    mpi_all_sigma_src,
    disparity_all_src,
    G_tgt_src,
    K_src_inv,
    K_tgt,
    K_src,
    src_pose,
    homography_sampler,
    hard_flow=False
):
    xyz_src_BS3HW = mpi_rendering.get_src_xyz_from_plane_disparity(
        homography_sampler.meshgrid,
        disparity_all_src,
        K_src_inv
    )

    xyz_tgt_BS3HW = mpi_rendering.get_tgt_xyz_from_plane_disparity(
        xyz_src_BS3HW,
        G_tgt_src
    )

    mpi_depth_src = torch.reciprocal(disparity_all_src)
    B, S = disparity_all_src.size()
    xyz_tgt = xyz_tgt_BS3HW.reshape(
        B * S, 3, -1) / mpi_depth_src[0].unsqueeze(1).unsqueeze(2)
    # BSx3xHW torch.Size([64, 3, 98304])
    meshgrid_tgt = torch.matmul(K_tgt, xyz_tgt)
    meshgrid_src = homography_sampler.meshgrid.unsqueeze(
        0).unsqueeze(1).repeat(B, S, 1, 1, 1).reshape(B * S, 3, -1)
    mpi_flow_src = meshgrid_tgt - meshgrid_src
    H, W = mpi_all_rgb_src.shape[-2:]
    mpi_flow_src = mpi_flow_src.reshape(B, S, 3, H, W)[:, :, :2]

    tgt_imgs_syn, tgt_depth_syn, _, flow_syn = mpi_rendering.render_tgt_rgb_depth(
        homography_sampler,
        mpi_all_rgb_src,
        mpi_all_sigma_src,
        disparity_all_src,
        xyz_src_BS3HW,
        xyz_src_BS3HW,
        src_pose,
        K_src_inv,
        K_src,
        mpi_flow_src,
        use_alpha=False,
        is_bg_depth_inf=False,
        hard_flow=hard_flow
    )
    return tgt_imgs_syn, tgt_depth_syn, flow_syn


def render_3dphoto_single_coco(
    src_imgs,  # [b,3,h,w]
    mpi_all_src,  # [b,s,4,h,w]
    disparity_all_src,  # [b,s]
    k_src,  # [b,3,3]
    k_tgt,  # [b,3,3]
    save_path,
    out_path='outputs',
    single=False,
    data_path=None,
    name=None,
    hard_flow=False,
    random_pose=False
):
    h, w = mpi_all_src.shape[-2:]
    device = mpi_all_src.device
    homography_sampler = HomographySample(h, w, device)
    k_src_inv = torch.inverse(k_src.cpu())
    k_src_inv = k_src_inv.cuda()

    # preprocess the predict MPI
    xyz_src_BS3HW = mpi_rendering.get_src_xyz_from_plane_disparity(
        homography_sampler.meshgrid,
        disparity_all_src,
        k_src_inv,
    )
    mpi_all_rgb_src = mpi_all_src[:, :, 0:3, :, :]  # BxSx3xHxW
    mpi_all_sigma_src = mpi_all_src[:, :, 3:, :, :]  # BxSx1xHxW
    _, _, blend_weights, _, _, _ = mpi_rendering.render(
        mpi_all_rgb_src,
        mpi_all_sigma_src,
        xyz_src_BS3HW,
        use_alpha=False,
        is_bg_depth_inf=False,
    )
    mpi_all_rgb_src = blend_weights * \
        src_imgs.unsqueeze(1) + (1 - blend_weights) * mpi_all_rgb_src
    # swing_path_list = gen_swing_path(num_frames=90, r_x=0.0, r_y=0., r_z=0.14)
    swing_path_list = gen_swing_path()
    src_img_np = src_imgs[0].permute(
        1, 2, 0).contiguous().cpu().numpy()  # [b,h,w,3]
    src_img_np = np.clip(np.round(src_img_np * 255),
                         a_min=0, a_max=255).astype(np.uint8)
    src_pose = swing_path_list[0]
    rand_idx = np.random.randint(low=1, high=len(swing_path_list)-1)
    cam_ext = swing_path_list[rand_idx]
    scx = ((-1)**random.randrange(2))
    scy = ((-1)**random.randrange(2))
    scz = ((-1)**random.randrange(2))
    # Random scalars in -0.2,0.2, excluding -0.1,0.1 to avoid zeros / very small motions
    cx = (random.random()*0.1+0.1) * scx
    cy = (random.random()*0.1+0.1) * scy
    cz = (random.random()*0.1+0.1) * scz
    camera_mot = [cx, cy, cz]

    # generate random triplet of Euler angles
    # Random sign
    sax = ((-1)**random.randrange(2))
    say = ((-1)**random.randrange(2))
    saz = ((-1)**random.randrange(2))
    # Random angles in -pi/18,pi/18, excluding -pi/36,pi/36 to avoid zeros / very small rotations
    ax = (random.random()*math.pi / 36.0 + math.pi / 36.0) * sax
    ay = (random.random()*math.pi / 36.0 + math.pi / 36.0) * say
    az = (random.random()*math.pi / 36.0 + math.pi / 36.0) * saz
    camera_ang = [ax, ay, az]

    axisangle = torch.from_numpy(
        np.array([[camera_ang]], dtype=np.float32)).cuda()
    translation = torch.from_numpy(np.array([[camera_mot]])).cuda()

    # Compute (R|t)
    T1 = transformation_from_parameters(axisangle, translation)[0]
    if random_pose:
        cam_ext = T1

    frame, depth, flowA2B = render_novel_view(
        mpi_all_rgb_src,
        mpi_all_sigma_src,
        disparity_all_src,
        cam_ext.cuda(),
        k_src_inv,
        k_tgt,
        k_src,
        src_pose,
        homography_sampler,
        hard_flow
    )
    frame_np = frame[0].permute(
        1, 2, 0).contiguous().cpu().numpy()  # [b,h,w,3]
    flow_np = flowA2B[0].permute(
        1, 2, 0).contiguous().cpu().numpy()  # [b,h,w,3]
    name = name.split('/')[-1][:-4]
    frame_np = np.clip(np.round(frame_np * 255),
                       a_min=0, a_max=255).astype(np.uint8)
    writeFlow(os.path.join(data_path, 'flo',
              '{}.flo'.format(name)), flow_np)
    cv2.imwrite(os.path.join(data_path, 'image_1',
                '{}.png'.format(name)), src_img_np[:, :, [2, 1, 0]])
    cv2.imwrite(os.path.join(data_path, 'image_2',
                '{}.png'.format(name)), frame_np[:, :, [2, 1, 0]])


def render_3dphoto_cumtom(
    src_imgs,  # [b,3,h,w]
    mpi_all_src,  # [b,s,4,h,w]
    disparity_all_src,  # [b,s]
    k_src,  # [b,3,3]
    k_tgt,  # [b,3,3]
    save_path,
    out_path='outputs',
    single=False,
    data_path=None,
    name=None,
    hard_flow=False,
    random_pose=False,
    repeat=1
):
    h, w = mpi_all_src.shape[-2:]
    device = mpi_all_src.device
    homography_sampler = HomographySample(h, w, device)
    k_src_inv = torch.inverse(k_src.cpu())
    k_src_inv = k_src_inv.cuda()

    # preprocess the predict MPI
    xyz_src_BS3HW = mpi_rendering.get_src_xyz_from_plane_disparity(
        homography_sampler.meshgrid,
        disparity_all_src,
        k_src_inv,
    )
    mpi_all_rgb_src = mpi_all_src[:, :, 0:3, :, :]  # BxSx3xHxW
    mpi_all_sigma_src = mpi_all_src[:, :, 3:, :, :]  # BxSx1xHxW
    _, _, blend_weights, _, _, _ = mpi_rendering.render(
        mpi_all_rgb_src,
        mpi_all_sigma_src,
        xyz_src_BS3HW,
        use_alpha=False,
        is_bg_depth_inf=False,
    )
    mpi_all_rgb_src = blend_weights * \
        src_imgs.unsqueeze(1) + (1 - blend_weights) * mpi_all_rgb_src
    # swing_path_list = gen_swing_path(num_frames=90, r_x=0.0, r_y=0., r_z=0.14)
    swing_path_list = gen_swing_path()
    src_img_np = src_imgs[0].permute(
        1, 2, 0).contiguous().cpu().numpy()  # [b,h,w,3]
    src_img_np = np.clip(np.round(src_img_np * 255),
                         a_min=0, a_max=255).astype(np.uint8)
    src_pose = swing_path_list[0]
    name = name.split('/')[-1][:-4]
    for idx in range(repeat):
        scx = ((-1)**random.randrange(2))
        scy = ((-1)**random.randrange(2))
        scz = ((-1)**random.randrange(2))
        # Random scalars in -0.2,0.2, excluding -0.1,0.1 to avoid zeros / very small motions
        cx = (random.random()*0.1+0.1) * scx
        cy = (random.random()*0.1+0.1) * scy
        cz = (random.random()*0.1+0.1) * scz
        camera_mot = [cx, cy, cz]

        # generate random triplet of Euler angles
        # Random sign
        sax = ((-1)**random.randrange(2))
        say = ((-1)**random.randrange(2))
        saz = ((-1)**random.randrange(2))
        # Random angles in -pi/18,pi/18, excluding -pi/36,pi/36 to avoid zeros / very small rotations
        ax = (random.random()*math.pi / 36.0 + math.pi / 36.0) * sax
        ay = (random.random()*math.pi / 36.0 + math.pi / 36.0) * say
        az = (random.random()*math.pi / 36.0 + math.pi / 36.0) * saz
        camera_ang = [ax, ay, az]
        camera_ang = [0, 0, 0]

        axisangle = torch.from_numpy(
            np.array([[camera_ang]], dtype=np.float32)).cuda()
        translation = torch.from_numpy(np.array([[camera_mot]])).cuda()

        # Compute (R|t)
        cam_ext = transformation_from_parameters(axisangle, translation)[0]

        frame, depth, flowB2A = render_novel_view(
            mpi_all_rgb_src,
            mpi_all_sigma_src,
            disparity_all_src,
            cam_ext.cuda(),
            k_src_inv,
            k_tgt,
            k_src,
            src_pose,
            homography_sampler,
            hard_flow
        )
        frame_np = frame[0].permute(
            1, 2, 0).contiguous().cpu().numpy()  # [b,h,w,3]
        flow_np = flowB2A[0].permute(
            1, 2, 0).contiguous().cpu().numpy()  # [b,h,w,3]
        frame_np = np.clip(np.round(frame_np * 255),
                           a_min=0, a_max=255).astype(np.uint8)
        writeFlow(os.path.join(data_path, 'flo',
                               '{}_{:04d}.flo'.format(name, idx)), flow_np)
        cv2.imwrite(os.path.join(data_path, 'image_2',
                    '{}_{:04d}.png'.format(name, idx)), src_img_np[:, :, [2, 1, 0]])
        cv2.imwrite(os.path.join(data_path, 'image_1',
                    '{}_{:04d}.png'.format(name, idx)), frame_np[:, :, [2, 1, 0]])
        # if idx > 0:
        #     import IPython
        #     IPython.embed()
        #     exit()


def render_3dphoto_single_kitti_pose(
    src_imgs,  # [b,3,h,w]
    mpi_all_src,  # [b,s,4,h,w]
    disparity_all_src,  # [b,s]
    k_src,  # [b,3,3]
    k_tgt,  # [b,3,3]
    save_path,
    out_path='outputs',
    single=False,
    data_path=None,
    name=None,
    hard_flow=False,
    extra=None,
    z_only=False,
    repeat=1,
    poses_data=None
):
    name = name.split('/')[-1][:-4]
    h, w = mpi_all_src.shape[-2:]
    device = mpi_all_src.device
    homography_sampler = HomographySample(h, w, device)
    k_src_inv = torch.inverse(k_src.to(torch.float64).cpu())
    k_src_inv = k_src_inv.cuda().to(k_src.dtype)

    # preprocess the predict MPI
    xyz_src_BS3HW = mpi_rendering.get_src_xyz_from_plane_disparity(
        homography_sampler.meshgrid,
        disparity_all_src,
        k_src_inv,
    )
    mpi_all_rgb_src = mpi_all_src[:, :, 0:3, :, :]  # BxSx3xHxW
    mpi_all_sigma_src = mpi_all_src[:, :, 3:, :, :]  # BxSx1xHxW
    _, _, blend_weights, _, _, _ = mpi_rendering.render(
        mpi_all_rgb_src,
        mpi_all_sigma_src,
        xyz_src_BS3HW,
        use_alpha=False,
        is_bg_depth_inf=False,
    )
    mpi_all_rgb_src = blend_weights * \
        src_imgs.unsqueeze(1) + (1 - blend_weights) * mpi_all_rgb_src
    # swing_path_list = gen_swing_path(num_frames=90, r_x=0.0, r_y=0., r_z=0.14)
    swing_path_list = gen_swing_path()
    src_img_np = src_imgs[0].permute(
        1, 2, 0).contiguous().cpu().numpy()  # [b,h,w,3]
    src_img_np = np.clip(np.round(src_img_np * 255),
                         a_min=0, a_max=255).astype(np.uint8)
    src_pose = swing_path_list[0]

    for i_ in range(repeat):
        if not poses_data is None and name in poses_data:
            bax, bcx, bcz = poses_data[name]
        else:
            print(name, 'not found in poses dict.')
            bax, bcx, bcz = math.pi / 36.0, 0.1, 0.2
        bay, baz, bcy = math.pi / 36.0 * 0.2, math.pi / 36.0 * 0.2, 0.1
        scx = ((-1)**random.randrange(2))
        scy = ((-1)**random.randrange(2))
        scz = ((-1)**random.randrange(2))
        # Random scalars in -0.2,0.2, excluding -0.1,0.1 to avoid zeros / very small motions
        cx = random.random()*0.15 * scx + bcx
        cy = random.random()*0.15 * scy + bcy
        cz = random.random()*0.15 * scz + bcz
        camera_mot = [cx, cy, cz]
        if i_ == 0 and not poses_data is None:
            camera_mot = [bcx, 0, bcz]
        # generate random triplet of Euler angles
        # Random sign
        sax = ((-1)**random.randrange(2))
        say = ((-1)**random.randrange(2))
        saz = ((-1)**random.randrange(2))
        # Random angles in -pi/18,pi/18, excluding -pi/36,pi/36 to avoid zeros / very small rotations
        ax = random.random()*math.pi / 36.0 * 0.2 * sax + bax
        ay = random.random()*math.pi / 36.0 * 0.2 * say + bay
        az = random.random()*math.pi / 36.0 * 0.2 * saz + baz
        camera_ang = [ax, ay, az]
        if i_ == 0 and not poses_data is None:
            camera_ang = [bax, 0, 0]

        camera_ang = [0, 0, 0]
        camera_mot = [0, 0, 0]

        axisangle = torch.from_numpy(
            np.array([[camera_ang]], dtype=np.float32)).cuda()
        translation = torch.from_numpy(np.array([[camera_mot]])).cuda()
        # Compute (R|t)
        cam_ext = transformation_from_parameters(axisangle, translation)[0]

        frame, depth, flowA2B = render_novel_view(
            mpi_all_rgb_src,
            mpi_all_sigma_src,
            disparity_all_src,
            cam_ext.cuda(),
            k_src_inv,
            k_tgt,
            k_src,
            src_pose,
            homography_sampler,
            hard_flow
        )
        frame_np = frame[0].permute(
            1, 2, 0).contiguous().cpu().numpy()  # [b,h,w,3]
        flow_np = flowA2B[0].permute(
            1, 2, 0).contiguous().cpu().numpy()  # [b,h,w,3]
        frame_np = np.clip(np.round(frame_np * 255),
                           a_min=0, a_max=255).astype(np.uint8)
        writeFlow(os.path.join(data_path, 'flo',
                               '{}_{}_{}.flo'.format(extra, name, i_)), flow_np)
        cv2.imwrite(os.path.join(data_path, 'image_1',
                    '{}_{}_{}.png'.format(extra, name, i_)), src_img_np[:, :, [2, 1, 0]])
        cv2.imwrite(os.path.join(data_path, 'image_2',
                    '{}_{}_{}.png'.format(extra, name, i_)), frame_np[:, :, [2, 1, 0]])
        res = np.vstack([
            src_img_np[:, :, [2, 1, 0]],
            frame_np[:, :, [2, 1, 0]]
        ])
        cv2.imwrite('res.png', res)
        import IPython
        IPython.embed()
        exit()


def render_3dphoto_video(
    src_imgs,  # [b,3,h,w]
    mpi_all_src,  # [b,s,4,h,w]
    disparity_all_src,  # [b,s]
    k_src,  # [b,3,3]
    k_tgt,  # [b,3,3]
    out_path='outputs',
    name=None,
    hard_flow=False,
    poses=None
):
    h, w = mpi_all_src.shape[-2:]
    device = mpi_all_src.device
    homography_sampler = HomographySample(h, w, device)
    k_src_inv = torch.inverse(k_src.to(torch.float64).cpu())
    k_src_inv = k_src_inv.cuda().to(k_src.dtype)

    # preprocess the predict MPI
    xyz_src_BS3HW = mpi_rendering.get_src_xyz_from_plane_disparity(
        homography_sampler.meshgrid,
        disparity_all_src,
        k_src_inv,
    )
    mpi_all_rgb_src = mpi_all_src[:, :, 0:3, :, :]  # BxSx3xHxW
    mpi_all_sigma_src = mpi_all_src[:, :, 3:, :, :]  # BxSx1xHxW
    _, _, blend_weights, _, _, _ = mpi_rendering.render(
        mpi_all_rgb_src,
        mpi_all_sigma_src,
        xyz_src_BS3HW,
        use_alpha=False,
        is_bg_depth_inf=False,
    )
    mpi_all_rgb_src = blend_weights * \
        src_imgs.unsqueeze(1) + (1 - blend_weights) * mpi_all_rgb_src
    # swing_path_list = gen_swing_path(num_frames=90, r_x=0.0, r_y=0., r_z=0.14)
    swing_path_list = gen_swing_path()
    src_img_np = src_imgs[0].permute(
        1, 2, 0).contiguous().cpu().numpy()  # [b,h,w,3]
    src_img_np = np.clip(np.round(src_img_np * 255),
                         a_min=0, a_max=255).astype(np.uint8)
    src_pose = swing_path_list[0]

    # Compute (R|t)
    untrust, R, T = poses
    R[:, :] = 0
    R[0, 0], R[1, 1], R[2, 2] = 1, 1, 1
    T[:, :] = 0
    cam_ext = np.concatenate([R, T], 1)
    ones = np.array([[0, 0, 0, 1]])
    cam_ext = np.concatenate([cam_ext, ones], 0)
    cam_ext = torch.from_numpy(cam_ext).cuda().to(k_src.dtype)
    if untrust:
        return

    frame, depth, flowA2B = render_novel_view(
        mpi_all_rgb_src,
        mpi_all_sigma_src,
        disparity_all_src,
        cam_ext.cuda(),
        k_src_inv,
        k_tgt,
        k_src,
        src_pose,
        homography_sampler,
        hard_flow
    )
    frame_np = frame[0].permute(
        1, 2, 0).contiguous().cpu().numpy()  # [b,h,w,3]
    flow_np = flowA2B[0].permute(
        1, 2, 0).contiguous().cpu().numpy()  # [b,h,w,3]
    name = name.split('/')[-1][:-4]
    frame_np = np.clip(np.round(frame_np * 255),
                       a_min=0, a_max=255).astype(np.uint8)
    flow_color = flow_to_color(flow_np, convert_to_bgr=True)
    src_img_np = arrowon(src_img_np, flow_np)
    cv2.imwrite(os.path.join(out_path, 'flow',
                '{}.png'.format(name)), flow_color)
    cv2.imwrite(os.path.join(out_path, 'image',
                '{}.png'.format(name)), frame_np[:, :, [2, 1, 0]])
    cv2.imwrite(os.path.join(out_path, 'src_image',
                '{}.png'.format(name)), src_img_np[:, :, [2, 1, 0]])
    import IPython
    IPython.embed()
    exit()


def render_3dphoto_cumtom_pose(
    src_imgs,  # [b,3,h,w]
    mpi_all_src,  # [b,s,4,h,w]
    disparity_all_src,  # [b,s]
    k_src,  # [b,3,3]
    k_tgt,  # [b,3,3]
    save_path,
    out_path='outputs',
    single=False,
    data_path=None,
    name=None,
    hard_flow=False,
    random_pose=False,
    repeat=1,
    pose=None,
):
    h, w = mpi_all_src.shape[-2:]
    device = mpi_all_src.device
    homography_sampler = HomographySample(h, w, device)
    k_src_inv = torch.inverse(k_src.to(torch.float64).cpu())
    k_src_inv = k_src_inv.cuda().to(k_src.dtype)

    # preprocess the predict MPI
    xyz_src_BS3HW = mpi_rendering.get_src_xyz_from_plane_disparity(
        homography_sampler.meshgrid,
        disparity_all_src,
        k_src_inv,
    )
    mpi_all_rgb_src = mpi_all_src[:, :, 0:3, :, :]  # BxSx3xHxW
    mpi_all_sigma_src = mpi_all_src[:, :, 3:, :, :]  # BxSx1xHxW
    _, _, blend_weights, _, _, _ = mpi_rendering.render(
        mpi_all_rgb_src,
        mpi_all_sigma_src,
        xyz_src_BS3HW,
        use_alpha=False,
        is_bg_depth_inf=False,
    )
    mpi_all_rgb_src = blend_weights * \
        src_imgs.unsqueeze(1) + (1 - blend_weights) * mpi_all_rgb_src
    # swing_path_list = gen_swing_path(num_frames=90, r_x=0.0, r_y=0., r_z=0.14)
    swing_path_list = gen_swing_path()
    src_img_np = src_imgs[0].permute(
        1, 2, 0).contiguous().cpu().numpy()  # [b,h,w,3]
    src_img_np = np.clip(np.round(src_img_np * 255),
                         a_min=0, a_max=255).astype(np.uint8)
    src_pose = swing_path_list[0]
    name = name.split('/')[-1][:-4]

    for idx in range(repeat):

        untrust, R, T = pose
        if idx > 0:
            # R_noise = (np.random.random(R.shape)*2-1)*0.1
            # maxR, minR = R.max(), R.min()
            # R = np.clip(R+R_noise, minR, maxR)

            T_noise = (np.random.random(T.shape)*2-1)*np.max(np.abs(T))
            T = T+T_noise
        cam_ext = np.concatenate([R, T], 1)

        ones = np.array([[0, 0, 0, 1]])
        cam_ext = np.concatenate([cam_ext, ones], 0)
        cam_ext = torch.from_numpy(cam_ext).cuda().to(k_src.dtype)
        if idx > 0 and np.random.rand() > 0.5:
            cam_ext = torch.inverse(cam_ext.cpu()).cuda().to(k_src.dtype)

        if idx > 0 and np.random.rand() > 0.5:
            R = torch.zeros_like(cam_ext)[:3, :3]
            R[0, 0] = 1.
            R[1, 1] = 1.
            R[2, 2] = 1.
            cam_ext[:3, :3] = R

        frame, depth, flowA2B = render_novel_view(
            mpi_all_rgb_src,
            mpi_all_sigma_src,
            disparity_all_src,
            cam_ext.cuda(),
            k_src_inv,
            k_tgt,
            k_src,
            src_pose,
            homography_sampler,
            hard_flow
        )
        frame_np = frame[0].permute(
            1, 2, 0).contiguous().cpu().numpy()  # [b,h,w,3]
        flow_np = flowA2B[0].permute(
            1, 2, 0).contiguous().cpu().numpy()  # [b,h,w,3]
        frame_np = np.clip(np.round(frame_np * 255),
                           a_min=0, a_max=255).astype(np.uint8)
        # flow_color = flow_to_color(flow_np, convert_to_bgr=True)
        # src_img_np_ = arrowon(src_img_np.copy(), flow_np)

        # cv2.imwrite('results/{}.png'.format(idx), np.hstack(
        #     [src_img_np_[:, :, [2, 1, 0]], frame_np[:, :, [2, 1, 0]], flow_color]))
        # continue
        writeFlow(os.path.join(data_path, 'flo',
                               '{}_{:04d}.flo'.format(name, idx)), flow_np)
        cv2.imwrite(os.path.join(data_path, 'image_1',
                    '{}_{:04d}.png'.format(name, idx)), src_img_np[:, :, [2, 1, 0]])
        cv2.imwrite(os.path.join(data_path, 'image_2',
                    '{}_{:04d}.png'.format(name, idx)), frame_np[:, :, [2, 1, 0]])
    # import IPython
    # IPython.embed()
    # exit()


def render_3dphoto_single_flow(
    src_imgs,  # [b,3,h,w]
    flo,
    mpi_all_src,  # [b,s,4,h,w]
    disparity_all_src,  # [b,s]
    k_src,  # [b,3,3]
    out_path='outputs',
    hard_flow=False
):
    h, w = mpi_all_src.shape[-2:]
    device = mpi_all_src.device
    homography_sampler = HomographySample(h, w, device)
    k_src_inv = torch.inverse(k_src.to(torch.float64).cpu())
    k_src_inv = k_src_inv.cuda().to(k_src.dtype)

    # preprocess the predict MPI
    xyz_src_BS3HW = mpi_rendering.get_src_xyz_from_plane_disparity(
        homography_sampler.meshgrid,
        disparity_all_src,
        k_src_inv,
    )
    mpi_all_rgb_src = mpi_all_src[:, :, 0:3, :, :]  # BxSx3xHxW
    mpi_all_sigma_src = mpi_all_src[:, :, 3:, :, :]  # BxSx1xHxW
    _, _, blend_weights, _, _, _ = mpi_rendering.render(
        mpi_all_rgb_src,
        mpi_all_sigma_src,
        xyz_src_BS3HW,
        use_alpha=False,
        is_bg_depth_inf=False,
    )
    mpi_all_rgb_src = blend_weights * \
        src_imgs.unsqueeze(1) + (1 - blend_weights) * mpi_all_rgb_src
    swing_path_list = gen_swing_path()
    src_img_np = src_imgs[0].permute(
        1, 2, 0).contiguous().cpu().numpy()  # [b,h,w,3]
    src_img_np = np.clip(np.round(src_img_np * 255),
                         a_min=0, a_max=255).astype(np.uint8)

    B, S = mpi_all_rgb_src.shape[:2]
    meshgrid_tgt_homo = homography_sampler.meshgrid.unsqueeze(0).expand(S, 3, h, w)[
        :, :2]
    flo = flo.expand(S, 2, h, w)
    meshgrid_src = (meshgrid_tgt_homo + flo).permute(0, 2, 3, 1)
    src_BCHW = torch.cat(
        (mpi_all_rgb_src, mpi_all_sigma_src), dim=2)  # BxSx(3+1+3)xHxW

    meshgrid_src[:, :, :, 0] = (meshgrid_src[:, :, :, 0]+0.5) / (w * 0.5) - 1
    meshgrid_src[:, :, :, 1] = (meshgrid_src[:, :, :, 1]+0.5) / (h * 0.5) - 1
    tgt_mpi_xyz_BsCHW = torch.nn.functional.grid_sample(src_BCHW.view(B*S, 4, h, w),
                                                        grid=meshgrid_src, padding_mode='border',
                                                        align_corners=False)
    tgt_mpi_xyz = tgt_mpi_xyz_BsCHW.view(B, S, 4, h, w)
    tgt_rgb_BS3HW = tgt_mpi_xyz[:, :, 0:3, :, :]
    tgt_sigma_BS1HW = tgt_mpi_xyz[:, :, 3:4, :, :]
    tgt_xyz_BS3HW = xyz_src_BS3HW

    # Bx3xHxW, Bx1xHxW, Bx1xHxW
    tgt_z_BS1HW = tgt_xyz_BS3HW[:, :, -1:]
    tgt_sigma_BS1HW = torch.where(tgt_z_BS1HW >= 0,
                                  tgt_sigma_BS1HW,
                                  torch.zeros_like(tgt_sigma_BS1HW, device=tgt_sigma_BS1HW.device))
    frame, depth, _, _, _, _ = mpi_rendering.render(tgt_rgb_BS3HW,
                                                    tgt_sigma_BS1HW,
                                                    tgt_xyz_BS3HW,
                                                    use_alpha=False,
                                                    is_bg_depth_inf=False,
                                                    hard_flow=hard_flow)

    frame_np = frame[0].permute(
        1, 2, 0).contiguous().cpu().numpy()  # [b,h,w,3]
    # flow_np = flowA2B[0].permute(
    #     1, 2, 0).contiguous().cpu().numpy()  # [b,h,w,3]
    # flow_color = flow_to_color(flow_np, convert_to_bgr=True)
    frame_np = np.clip(np.round(frame_np * 255),
                       a_min=0, a_max=255).astype(np.uint8)
    cv2.imwrite('a-dst.png', frame_np[:, :, [2, 1, 0]])
    # cv2.imwrite(os.path.join(out_path, 'flow',
    #                          '{:05d}.png'.format(i)), flow_color)
    # src_img_np_ = arrowon(src_img_np.copy(), flow_np)
    # frame_np = np.vstack([src_img_np_, frame_np, flow_color])
    # import IPython
    # IPython.embed()
    # exit()


def render_novel_view_with_flow(
    mpi_all_rgb_src,
    mpi_all_sigma_src,
    disparity_all_src,
    G_tgt_src,
    K_src_inv,
    K_tgt,
    homography_sampler,
    hard_flow=False
):
    xyz_src_BS3HW = mpi_rendering.get_src_xyz_from_plane_disparity(
        homography_sampler.meshgrid,
        disparity_all_src,
        K_src_inv
    )

    xyz_tgt_BS3HW = mpi_rendering.get_tgt_xyz_from_plane_disparity(
        xyz_src_BS3HW,
        G_tgt_src
    )

    mpi_depth_src = torch.reciprocal(disparity_all_src)
    B, S = disparity_all_src.size()
    xyz_tgt = xyz_tgt_BS3HW.reshape(
        B * S, 3, -1) / mpi_depth_src[0].unsqueeze(1).unsqueeze(2)
    # BSx3xHW torch.Size([64, 3, 98304])
    meshgrid_tgt = torch.matmul(K_tgt, xyz_tgt)
    meshgrid_src = homography_sampler.meshgrid.unsqueeze(
        0).unsqueeze(1).repeat(B, S, 1, 1, 1).reshape(B * S, 3, -1)
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
        hard_flow=hard_flow
    )

    return tgt_imgs_syn, tgt_depth_syn, flow_syn


def render_3dphoto_single_with_mask(
    src_imgs,  # [b,3,h,w]
    mask,
    disp,
    depth_path,
    mpi_all_src,  # [b,s,4,h,w]
    disparity_all_src,  # [b,s]
    k_src,  # [b,3,3]
    k_tgt,  # [b,3,3]
    save_path,
    out_path='outputs',
    single=False,
    data_path=None,
    name=None,
    hard_flow=False,
):
    h, w = mpi_all_src.shape[-2:]
    device = mpi_all_src.device
    homography_sampler = HomographySample(h, w, device)
    k_src_inv = torch.inverse(k_src.to(torch.float64).cpu())
    k_src_inv = k_src_inv.cuda().to(k_src.dtype)

    # preprocess the predict MPI
    xyz_src_BS3HW = mpi_rendering.get_src_xyz_from_plane_disparity(
        homography_sampler.meshgrid,
        disparity_all_src,
        k_src_inv,
    )
    mpi_all_rgb_src = mpi_all_src[:, :, 0:3, :, :]  # BxSx3xHxW
    mpi_all_sigma_src = mpi_all_src[:, :, 3:, :, :]  # BxSx1xHxW
    _, _, blend_weights, _, _, _ = mpi_rendering.render(
        mpi_all_rgb_src,
        mpi_all_sigma_src,
        xyz_src_BS3HW,
        use_alpha=False,
        is_bg_depth_inf=False,
    )
    mpi_all_rgb_src = blend_weights * \
        src_imgs.unsqueeze(1) + (1 - blend_weights) * mpi_all_rgb_src
    # mpi_all_rgb_src torch.Size([1, 64, 3, 256, 384])
    # render novel views
    # swing_path_list = gen_swing_path(num_frames=90, r_x=0, r_y=0, r_z=0.3)
    swing_path_list = gen_swing_path()
    frames = []
    src_img_np = src_imgs[0].permute(
        1, 2, 0).contiguous().cpu().numpy()  # [b,h,w,3]
    src_img_np = np.clip(np.round(src_img_np * 255),
                         a_min=0, a_max=255).astype(np.uint8)
    src_pose = swing_path_list[0]
    poses = torch.stack(swing_path_list).cpu().numpy()
    np.save('poses.npy', poses)
    # import IPython
    # IPython.embed()
    # exit()
    if single:
        rand_idx = np.random.randint(low=1, high=len(swing_path_list)-1)
        swing_path_list = [swing_path_list[rand_idx]]
    for i, cam_ext in enumerate(tqdm(swing_path_list)):
        frame, depth, flowA2B = render_novel_view(
            mpi_all_rgb_src,
            mpi_all_sigma_src,
            disparity_all_src,
            cam_ext.cuda(),
            k_src_inv,
            k_tgt,
            k_src,
            src_pose,
            homography_sampler,
            hard_flow
        )
        frame_np = frame[0].permute(
            1, 2, 0).contiguous().cpu().numpy()  # [b,h,w,3]
        flow_np = flowA2B[0].permute(
            1, 2, 0).contiguous().cpu().numpy()  # [b,h,w,3]
        depth = depth[0, 0].contiguous().cpu().numpy()  # [b,h,w,3]
        flow_color = flow_to_color(flow_np, convert_to_bgr=True)
        frame_np = np.clip(np.round(frame_np * 255),
                           a_min=0, a_max=255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_path, 'image',
                                 'm{:05d}.png'.format(i)), frame_np[:, :, [2, 1, 0]])
        cv2.imwrite(os.path.join(out_path, 'flow',
                                 'm{:05d}.png'.format(i)), flow_color)
        cv2.imwrite(os.path.join(out_path, 'depth',
                                 'm{:05d}.png'.format(i)), depth*10)
        src_img_np_ = arrowon(src_img_np.copy(), flow_np)
        frame_np = np.vstack([src_img_np_, frame_np, flow_color])
        moveing_object_with_mask(
            depth_path, disp, src_img_np[:, :, [2, 1, 0]], k_src, k_src_inv, mask, i)
        # frame_np = np.hstack([src_img_np_, frame_np, flow_color])
        frames.append(frame_np)
    cv2.imwrite(os.path.join(out_path, 'src_image.png'.format(i)),
                src_img_np[:, :, [2, 1, 0]])
    rgb_clip = ImageSequenceClip(frames, fps=10)
    rgb_clip.write_videofile(save_path, verbose=False,
                             codec='mpeg4', logger=None, bitrate='2000k')


# def render_3dphoto_SS(
#     src_imgs,  # [b,3,h,w]
#     dst_imgs,
#     disp,
#     model,
#     k_src,  # [b,3,3]
#     k_tgt,  # [b,3,3]
#     save_path,
#     out_path='outputs',
#     single=False,
#     data_path=None,
#     name=None,
#     hard_flow=False,
#     extra=None,
#     z_only=False,
#     repeat=1
# ):
#     swing_path_list = gen_swing_path()
#     src_img_np = src_imgs[0].permute(
#         1, 2, 0).contiguous().cpu().numpy()  # [b,h,w,3]
#     src_img_np = np.clip(np.round(src_img_np * 255),
#                          a_min=0, a_max=255).astype(np.uint8)
#     src_pose = swing_path_list[0]
#     axisangle = torch.from_numpy(
#         np.array([[[0, 0, 0]]], dtype=np.float32)).cuda().float()
#     translation = torch.from_numpy(np.array([[[0, 0, -0.1]]])).cuda().float()
#     axisangle.requires_grad = True
#     translation.requires_grad = True
#     optimizer_camera = optim.SGD(
#         [disp, axisangle, translation], lr=0.01, momentum=0.9)
#     optimizer_model = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#     frame_np_last = None
#     min_loss = 10000000
#     for i_ in range(repeat):
#         # with torch.no_grad():
#         mpi_all_src, disparity_all_src = model(src_imgs, disp)  # [b,s,4,h,w]
#         h, w = mpi_all_src.shape[-2:]
#         device = mpi_all_src.device
#         homography_sampler = HomographySample(h, w, device)
#         k_src_inv = torch.inverse(k_src.to(torch.float64).cpu())
#         k_src_inv = k_src_inv.cuda().to(k_src.dtype)

#         # preprocess the predict MPI
#         xyz_src_BS3HW = mpi_rendering.get_src_xyz_from_plane_disparity(
#             homography_sampler.meshgrid,
#             disparity_all_src,
#             k_src_inv,
#         )
#         mpi_all_rgb_src = mpi_all_src[:, :, 0:3, :, :]  # BxSx3xHxW
#         mpi_all_sigma_src = mpi_all_src[:, :, 3:, :, :]  # BxSx1xHxW
#         _, _, blend_weights, _, _ = mpi_rendering.render(
#             mpi_all_rgb_src,
#             mpi_all_sigma_src,
#             xyz_src_BS3HW,
#             use_alpha=False,
#             is_bg_depth_inf=False,
#         )
#         mpi_all_rgb_src = blend_weights * \
#             src_imgs.unsqueeze(1) + (1 - blend_weights) * mpi_all_rgb_src
#         # swing_path_list = gen_swing_path(num_frames=90, r_x=0.0, r_y=0., r_z=0.14)
#         optimizer_camera.zero_grad()
#         optimizer_model.zero_grad()
#         # Compute (R|t)
#         cam_ext = transformation_from_parameters(axisangle, translation)[0]

#         frame, depth, flowA2B = render_novel_view(
#             mpi_all_rgb_src,
#             mpi_all_sigma_src,
#             disparity_all_src,
#             cam_ext.cuda(),
#             k_src_inv,
#             k_tgt,
#             k_src,
#             src_pose,
#             homography_sampler,
#             hard_flow
#         )
#         loss = F.mse_loss(frame, dst_imgs)
#         loss.backward()
#         optimizer_camera.step()
#         optimizer_model.step()
def render_3dphoto_SS(
    src_imgs,  # [b,3,h,w]
    dst_imgs,
    disp,
    model,
    k_src,  # [b,3,3]
    k_tgt,  # [b,3,3]
    save_path,
    out_path='outputs',
    single=False,
    data_path=None,
    name=None,
    hard_flow=False,
    extra=None,
    z_only=False,
    repeat=1
):
    name = name.split('/')[-1].split('.')[0]
    src_np = src_imgs[0].permute(
        1, 2, 0).contiguous().cpu().numpy()  # [b,h,w,3]
    src_np = np.clip(np.round(src_np * 255),
                     a_min=0, a_max=255).astype(np.uint8)[:, :, [2, 1, 0]]
    dst_np = dst_imgs[0].permute(
        1, 2, 0).contiguous().cpu().numpy()  # [b,h,w,3]
    dst_np = np.clip(np.round(dst_np * 255),
                     a_min=0, a_max=255).astype(np.uint8)[:, :, [2, 1, 0]]
    mpi_all_src, disparity_all_src = model(src_imgs, disp)  # [b,s,4,h,w]
    h, w = mpi_all_src.shape[-2:]
    swing_path_list = gen_swing_path()
    src_pose = swing_path_list[0]
    min_loss = 10000000
    # preprocess the predict MPI
    device = mpi_all_src.device
    homography_sampler = HomographySample(h, w, device)
    k_src_inv = torch.inverse(k_src.to(torch.float64).cpu())
    k_src_inv = k_src_inv.cuda().to(k_src.dtype)
    mpi_all_rgb_src = mpi_all_src[:, :, 0:3, :, :]  # BxSx3xHxW
    mpi_all_sigma_src = mpi_all_src[:, :, 3:, :, :]  # BxSx1xHxW
    xyz_src_BS3HW = mpi_rendering.get_src_xyz_from_plane_disparity(
        homography_sampler.meshgrid,
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
    mpi_all_rgb_src = blend_weights * \
        src_imgs.unsqueeze(1) + (1 - blend_weights) * mpi_all_rgb_src
    count = 0
    if True:
        ax = 0.0
        cx = 0.020000000000000018
        cy = 0.020000000000000018
        cz = -0.18
    # for ax in np.arange(-0.02, 0.01, 0.01):
    #     for cx in np.arange(-0.1, 0.1, 0.02):
    #         for cy in np.arange(-0.1, 0.1, 0.02):
    #             for cz in np.arange(-0.5, 0.0, 0.01):
        axisangle = torch.from_numpy(
            np.array([[[ax, 0, 0]]], dtype=np.float32)).cuda().float()
        translation = torch.from_numpy(
            np.array([[[cx, cy, cz]]])).cuda().float()
        cam_ext = transformation_from_parameters(
            axisangle, translation)[0]

        frame, depth, flowA2B = render_novel_view(
            mpi_all_rgb_src,
            mpi_all_sigma_src,
            disparity_all_src,
            cam_ext.cuda(),
            k_src_inv,
            k_tgt,
            k_src,
            src_pose,
            homography_sampler,
            hard_flow
        )
        # import IPython
        # IPython.embed()
        # exit()
        loss = F.mse_loss(frame, dst_imgs)
        # loss = 0
        # loss += F.mse_loss(frame[:, :, int(0.7*h):, ], dst_imgs[:, :, int(0.7*h):, ])
        # loss += F.mse_loss(frame[:, :, :, :int(0.2*w)], dst_imgs[:, :, :, :int(0.2*w)])
        # loss += F.mse_loss(frame[:, :, :, -int(0.2*w):], dst_imgs[:, :, :, -int(0.2*w):])
        count += 1
        if loss < min_loss:
            print(
                '------------------------------------------------------------------------------')
            print('[Min Loss Updated]:', loss, 'step:', count)
            frame_np = frame[0].permute(
                1, 2, 0).contiguous().cpu().numpy()  # [b,h,w,3]
            frame_np = np.clip(np.round(frame_np * 255),
                               a_min=0, a_max=255).astype(np.uint8)[:, :, [2, 1, 0]]
            cv2.imwrite('temp/{}-{}.png'.format(extra, name), np.vstack(
                [src_np, dst_np, frame_np, np.abs(src_np - frame_np)]))
            min_loss = loss
            best = 'name={}, ax={}, cx={}, cy={}, cz={}\n'.format(
                name, ax, cx, cy, cz)
            print(best)
            flow_np = flowA2B[0].permute(
                1, 2, 0).contiguous().cpu().numpy()  # [b,h,w,3]
            writeFlow(os.path.join('temp/flow.flo'), flow_np)

    with open('poses-temp-{}.txt'.format(extra), 'a') as f:
        f.write(best)


def generate_random_pose(base_motions=[0.05, 0.05, 0.05]):
    scx = ((-1)**random.randrange(2))
    scy = ((-1)**random.randrange(2))
    scz = ((-1)**random.randrange(2))
    if base_motions[0] == 0.05:
        scz = -1
    else:
        scx = scx * 0.5
        scy = scy * 0.5
        scz = scz * 0.5
    # Random scalars in -0.2,0.2, excluding -0.1,0.1 to avoid zeros / very small motions
    cx = (random.random()*0.1+base_motions[0]) * scx
    cy = (random.random()*0.1+base_motions[1]) * scy
    cz = (random.random()*0.15+base_motions[2]) * scz
    camera_mot = [cx*0.5, cy*0.5, cz]

    # generate random triplet of Euler angles
    # Random sign
    sax = ((-1)**random.randrange(2))
    say = ((-1)**random.randrange(2))
    saz = ((-1)**random.randrange(2))
    if not base_motions[0] == 0.05:
        sax = sax * 0.5
        say = say * 0.5
        saz = saz * 0.5
    # Random angles in -pi/18,pi/18, excluding -pi/36,pi/36 to avoid zeros / very small rotations
    ax = (random.random()*math.pi / 36.0) * sax
    ay = (random.random()*math.pi / 36.0) * say
    az = (random.random()*math.pi / 36.0) * saz
    camera_ang = [ax*0.2, ay*0.2, az*0.2]
    
    if base_motions[0] == 0.05: # 背景
        camera_mot = [0, 0, -0.2]
        camera_ang = [0, 0, 0]
    else: 
        camera_mot = [0, 0, 0]
        camera_ang = [0, 0, 0]
    # if base_motions[0] == 0.0:
    #     camera_mot = [-0.03093760607365926, -0.041940814732665345, -0.12992744814254797]
    #     camera_ang = [-0.0044906984183519915, -0.013408794846845089, -0.00315106100918317]
    # [0.00761108834638096, 0.024679498132674824, -0.054049355577216594]
    # [-0.004240358947823318, 0.008457528065660668, 0.0029465591412749566]
    # ---------------------------------------
    # tensor([[[ 0.0162,  0.0065, -0.0201]]], device='cuda:0') tensor([[[-0.0084,  0.0276,  0.0084]]], device='cuda:0')
    # [-0.03093760607365926, -0.041940814732665345, -0.12992744814254797]
    # [-0.0044906984183519915, -0.013408794846845089, -0.00315106100918317]

    axisangle = torch.from_numpy(
        np.array([[camera_ang]], dtype=np.float32)).cuda().float()
    translation = torch.from_numpy(
        np.array([[camera_mot]])).cuda().float()
    
    if base_motions[0] == 0:
        c = 0.7
        translation = translation * c + (1-c) * torch.from_numpy(np.array([[[0.019908294054150358, -0.0013590704254421004, -0.0054824534806479225]]])).cuda().float()
        axisangle = axisangle * c + (1-c) * torch.from_numpy(np.array([[[-0.010197231871060683, 0.03575381384635365, 0.010695149122950507]]], dtype=np.float32)).cuda().float()
    
    print(translation, axisangle)
    
    cam_ext = transformation_from_parameters(
        axisangle, translation)[0]
    return cam_ext


def render_3dphoto_dynamic(
    src_imgs,  # [b,3,h,w]
    obj_mask,
    disp,
    model,
    k_src,  # [b,3,3]
    k_tgt,  # [b,3,3]
    save_path,
    out_path='outputs',
    single=False,
    data_path=None,
    name=None,
    hard_flow=False,
    extra=None,
    z_only=False,
    repeat=1,
    mask_thresh=0.99
):
    name = name.split('.')[0]
    src_np = src_imgs[0].permute(
        1, 2, 0).contiguous().cpu().numpy()  # [b,h,w,3]
    src_np = np.clip(np.round(src_np * 255),
                     a_min=0, a_max=255).astype(np.uint8)[:, :, [2, 1, 0]]
    with torch.no_grad():
        mpi_all_src, disparity_all_src = model(src_imgs, disp)  # [b,s,4,h,w]

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
    mpi_all_rgb_src = blend_weights * \
        src_imgs.unsqueeze(1) + (1 - blend_weights) * mpi_all_rgb_src

    for idx in range(repeat):
        cam_ext = generate_random_pose(base_motions=[0, 0, 0])
        cam_ext_dynamic = generate_random_pose()

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
            hard_flow
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
            hard_flow
        )
        frame_np = frame[0].permute(
            1, 2, 0).cpu().numpy().astype(np.float32)   # [b,h,w,3]
        frame_np = np.clip(np.round(frame_np * 255),
                           a_min=0, a_max=255).astype(np.uint8)[:, :, [2, 1, 0]]

        frame_dync_np = frame_dync[0].permute(
            1, 2, 0).cpu().numpy().astype(np.float32)   # [b,h,w,3]
        frame_dync_np = np.clip(np.round(frame_dync_np * 255),
                                a_min=0, a_max=255).astype(np.uint8)[:, :, [2, 1, 0]]
        mask = mask[0].permute(
            1, 2, 0).cpu().squeeze().numpy().astype(np.float32)   # [b,h,w,3]
        mask_dync = mask_dync[0].permute(
            1, 2, 0).cpu().squeeze().numpy().astype(np.float32)   # [b,h,w,3]

        flow_np = flowA2B[0].permute(
            1, 2, 0).contiguous().cpu().numpy().astype(np.float32)   # [b,h,w,3]
        flow_dync_np = flowA2B_dync[0].permute(
            1, 2, 0).contiguous().cpu().numpy().astype(np.float32)  # [b,h,w,3]

        # A2B 光流 mask
        flow_np[obj_mask_np < mask_thresh] = 0
        flow_dync_np[obj_mask_np >= mask_thresh] = 0

        flow_color = flow_to_color(flow_np, convert_to_bgr=True)
        flow_dync_color = flow_to_color(flow_dync_np, convert_to_bgr=True)
        frame_np[mask < mask_thresh] = 255
        frame_dync_np[mask_dync < mask_thresh] = 255
        frame_mix = frame_dync_np.copy()
        frame_mix[mask >= mask_thresh] = frame_np[mask >= mask_thresh]
        flow_mix = flow_dync_np.copy()
        flow_mix[obj_mask_np >= mask_thresh] = flow_np[obj_mask_np >= mask_thresh]
        flow_mix_color = flow_to_color(flow_mix, convert_to_bgr=True)
        flow_mix_color = flow_dync_color.copy()
        flow_mix_color[obj_mask_np >= mask_thresh] = flow_color[obj_mask_np >= mask_thresh]

        fill_mask = mask_dync.copy()
        fill_mask[mask >= mask_thresh] = 1

        fill_mask = (fill_mask < mask_thresh).astype(np.int32)

        mix_mask = np.logical_and(mask, mask_dync).astype(np.uint8)
        # depth
        scale = 2
        depth = depth.squeeze().cpu().numpy()
        depth_dync = depth_dync.squeeze().cpu().numpy()

        depth_mask = np.logical_and(depth > depth_dync, mix_mask)
        frame_mix_depth = frame_mix.copy()
        frame_mix_depth[depth_mask] = frame_dync_np[depth_mask]

        depth_res = np.vstack(
            [depth*scale,
             depth_dync*scale, depth_mask * 255
             ]).astype(np.uint8)

        inpainted = cv2.inpaint(
            frame_mix, fill_mask.astype(np.uint8), 3, cv2.INPAINT_NS)
        fill_mask_depth = fill_mask.copy()
        frame_mix_depth_inpainted = cv2.inpaint(
            frame_mix_depth, fill_mask_depth.astype(np.uint8), 3, cv2.INPAINT_NS)
        # visualization
        res1 = np.vstack(
            [src_np,
             frame_np,
             frame_dync_np,
             frame_mix])
        res2 = np.vstack(
            [inpainted,
             flow_color,
             flow_dync_color,
             flow_mix_color])

        mask_res = np.vstack([mix_mask, mask, mask_dync, fill_mask]) * 255
        depth_res = cv2.merge([depth_res, depth_res, depth_res])
        depth_res = np.vstack([frame_mix_depth_inpainted, depth_res])
        mask_res = cv2.merge([mask_res, mask_res, mask_res])
        res = np.hstack([res1, res2, mask_res, depth_res])
        from imutils import resize
        H, W = res.shape[:2]
        cv2.imwrite('temp/image.png', resize(res,
                    width=int(W/2), height=int(H/2)))
        writeFlow(os.path.join(data_path, 'flo',
                               '{}_{}_{}.flo'.format(extra, name, idx)), flow_mix)
        cv2.imwrite(os.path.join(data_path, 'image_1',
                    '{}_{}_{}.png'.format(extra, name, idx)), src_np)
        cv2.imwrite(os.path.join(data_path, 'image_2',
                    '{}_{}_{}.png'.format(extra, name, idx)), inpainted)


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
    hard_flow=False
):
    xyz_src_BS3HW = mpi_rendering.get_src_xyz_from_plane_disparity(
        homography_sampler.meshgrid.to(K_src.dtype),
        disparity_all_src,
        K_src_inv
    )

    xyz_tgt_BS3HW = mpi_rendering.get_tgt_xyz_from_plane_disparity(
        xyz_src_BS3HW.to(K_src.dtype),
        G_tgt_src.to(K_src.dtype)
    )

    mpi_depth_src = torch.reciprocal(disparity_all_src)
    B, S = disparity_all_src.size()
    xyz_tgt = xyz_tgt_BS3HW.reshape(
        B * S, 3, -1) / mpi_depth_src[0].unsqueeze(1).unsqueeze(2)
    # BSx3xHW torch.Size([64, 3, 98304])
    meshgrid_tgt = torch.matmul(K_tgt, xyz_tgt)
    meshgrid_src = homography_sampler.meshgrid.unsqueeze(
        0).unsqueeze(1).repeat(B, S, 1, 1, 1).reshape(B * S, 3, -1)
    mpi_flow_src = meshgrid_src - meshgrid_tgt
    H, W = mpi_all_rgb_src.shape[-2:]
    mpi_flow_src = mpi_flow_src.reshape(B, S, 3, H, W)[:, :, :2]
    obj_mask = obj_mask.unsqueeze(1).repeat(B, S, 1, 1, 1)

    tgt_imgs_syn, tgt_depth_syn, _, flow_syn, obj_mask = mpi_rendering.render_tgt_rgb_depth(
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
        obj_mask=obj_mask
    )
    flow_syn = torch.clip(flow_syn, -512, 512)
    return tgt_imgs_syn, tgt_depth_syn, flow_syn, obj_mask


def render_3dphoto_dynamic_objects(
    src_imgs,  # [b,3,h,w]
    obj_masks,
    disp,
    model,
    k_src,  # [b,3,3]
    k_tgt,  # [b,3,3]
    save_path,
    out_path='outputs',
    single=False,
    data_path=None,
    name=None,
    hard_flow=False,
    extra=None,
    z_only=False,
    repeat=1,
    mask_thresh=0.99
):
    name = name.split('.')[0]
    src_np = src_imgs[0].permute(
        1, 2, 0).contiguous().cpu().numpy()  # [b,h,w,3]
    src_np = np.clip(np.round(src_np * 255),
                     a_min=0, a_max=255).astype(np.uint8)[:, :, [2, 1, 0]]
    with torch.no_grad():
        mpi_all_src, disparity_all_src = model(src_imgs, disp)  # [b,s,4,h,w]

    h, w = mpi_all_src.shape[-2:]
    swing_path_list = gen_swing_path()
    src_pose = swing_path_list[0]
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
    mpi_all_rgb_src = blend_weights * \
        src_imgs.unsqueeze(1) + (1 - blend_weights) * mpi_all_rgb_src

    for idx in range(repeat):
        cam_ext = generate_random_pose(base_motions=[0, 0, 0])
        cam_ext_dynamic = generate_random_pose()
        obj_mask = obj_masks[idx % len(obj_masks)]
        obj_mask_np = obj_mask.squeeze().cpu().numpy()
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
            hard_flow
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
            hard_flow
        )
        frame_np = frame[0].permute(
            1, 2, 0).cpu().numpy().astype(np.float32)   # [b,h,w,3]
        frame_np = np.clip(np.round(frame_np * 255),
                           a_min=0, a_max=255).astype(np.uint8)[:, :, [2, 1, 0]]

        frame_dync_np = frame_dync[0].permute(
            1, 2, 0).cpu().numpy().astype(np.float32)   # [b,h,w,3]
        frame_dync_np = np.clip(np.round(frame_dync_np * 255),
                                a_min=0, a_max=255).astype(np.uint8)[:, :, [2, 1, 0]]
        mask = mask[0].permute(
            1, 2, 0).cpu().squeeze().numpy().astype(np.float32)   # [b,h,w,3]
        mask_dync = mask_dync[0].permute(
            1, 2, 0).cpu().squeeze().numpy().astype(np.float32)   # [b,h,w,3]

        flow_np = flowA2B[0].permute(
            1, 2, 0).contiguous().cpu().numpy().astype(np.float32)   # [b,h,w,3]
        flow_dync_np = flowA2B_dync[0].permute(
            1, 2, 0).contiguous().cpu().numpy().astype(np.float32)  # [b,h,w,3]

        # A2B 光流 mask
        flow_color = flow_to_color(flow_np, convert_to_bgr=True)
        flow_dync_color = flow_to_color(flow_dync_np, convert_to_bgr=True)
        flow_np[obj_mask_np < mask_thresh] = 0
        flow_dync_np[obj_mask_np >= mask_thresh] = 0
        # flow_color[obj_mask_np < mask_thresh] = 255
        # flow_dync_color[obj_mask_np >= mask_thresh] = 255

        # frame_np[mask < mask_thresh] = 255
        # frame_dync_np[mask_dync < mask_thresh] = 255
        frame_mix = frame_dync_np.copy()
        frame_mix[mask >= mask_thresh] = frame_np[mask >= mask_thresh]
        flow_mix = flow_dync_np.copy()
        flow_mix[obj_mask_np >= mask_thresh] = flow_np[obj_mask_np >= mask_thresh]
        flow_mix_color = flow_to_color(flow_mix, convert_to_bgr=True)
        # flow_mix_color = flow_dync_color.copy()
        # flow_mix_color[obj_mask_np >= mask_thresh] = flow_color[obj_mask_np >= mask_thresh]

        fill_mask = mask_dync.copy()
        fill_mask[mask >= mask_thresh] = 1

        fill_mask = (fill_mask < mask_thresh).astype(np.int32)

        mix_mask = np.logical_and(mask, mask_dync).astype(np.uint8)
        # depth
        scale = 2
        depth = depth.squeeze().cpu().numpy()
        depth_dync = depth_dync.squeeze().cpu().numpy()

        depth_mask = np.logical_and(depth > depth_dync, mix_mask)
        frame_mix_depth = frame_mix.copy()
        frame_mix_depth[depth_mask] = frame_dync_np[depth_mask]

        inpainted = cv2.inpaint(
            frame_mix, fill_mask.astype(np.uint8), 3, cv2.INPAINT_NS)
        fill_mask_depth = np.logical_and(mask_dync < mask_thresh, depth_mask)
        fill_mask_depth = np.logical_or(fill_mask_depth, fill_mask)
        frame_mix_depth_inpainted = cv2.inpaint(
            frame_mix_depth, fill_mask_depth.astype(np.uint8), 3, cv2.INPAINT_NS)
        fill_mask_depth = fill_mask_depth.astype(np.uint8)
        fill_mask = fill_mask.astype(np.uint8)
        # visualization
        res1 = np.vstack(
            [src_np,
             frame_np,
             frame_dync_np,
             frame_mix])
        res2 = np.vstack(
            [inpainted,
             flow_color,
             flow_dync_color,
             flow_mix_color])

        def to255(img):
            max_, min_ = img.max(), img.min()
            img = (img - min_) / (max_ - min_) * 255
            return img
        
        depth = to255(1/depth)
        depth_dync = to255(1/depth_dync)
        
        depth_res = np.vstack(
            [depth,
             depth_dync, depth_mask * 255
             ]).astype(np.uint8)
        mask_res = np.vstack([mix_mask, mask, mask_dync, fill_mask]) * 255
        depth_res = cv2.merge([depth_res, depth_res, depth_res])
        depth_res = np.vstack([frame_mix_depth_inpainted, depth_res])
        mask_res = cv2.merge([mask_res, mask_res, mask_res])
        res = np.hstack([res1, res2, mask_res, depth_res])
        from imutils import resize
        H, W = res.shape[:2]
        cv2.imwrite('temp/image-raw.png', res)
        cv2.imwrite('temp/image.png', resize(res,
                    width=int(W/2), height=int(H/2)))
        # cv2.imwrite('temp/image.png', frame_dync_np)
        # data_path = 'temp'
        # writeFlow(os.path.join(data_path, 'flo',
        #                        '{}_{}_{}.flo'.format(extra, name, idx)), flow_mix)
        # cv2.imwrite(os.path.join(data_path, 'image_1',
        #             '{}_{}_{}.png'.format(extra, name, idx)), src_np)
        # cv2.imwrite(os.path.join(data_path, 'image_2',
        #             '{}_{}_{}.png'.format(extra, name, idx)), inpainted)
        input()
    raise


def render_3dphoto_dynamic_objects_vis(
    flo,
    src_imgs,  # [b,3,h,w]
    tgt_imgs,
    obj_masks,
    disp,
    model,
    k_src,  # [b,3,3]
    k_tgt,  # [b,3,3]
    save_path,
    out_path='outputs',
    single=False,
    data_path=None,
    name=None,
    hard_flow=False,
    extra=None,
    z_only=False,
    repeat=1,
    mask_thresh=0.99
):
    flo, valid = flo
    floc = flow_to_color(flo, convert_to_bgr=True)
    valid = valid.astype(np.int32)
    in_valid = (1 - valid).astype(np.int32)
    floc[in_valid > 0] = 0
    # import IPython
    # IPython.embed()
    # exit()

    name = name.split('.')[0]
    src_np = src_imgs[0].permute(
        1, 2, 0).contiguous().cpu().numpy()  # [b,h,w,3]
    src_np = np.clip(np.round(src_np * 255),
                     a_min=0, a_max=255).astype(np.uint8)[:, :, [2, 1, 0]]
    tgt_np = tgt_imgs[0].permute(
        1, 2, 0).contiguous().cpu().numpy()  # [b,h,w,3]
    tgt_np = np.clip(np.round(tgt_np * 255),
                     a_min=0, a_max=255).astype(np.uint8)[:, :, [2, 1, 0]]
    with torch.no_grad():
        mpi_all_src, disparity_all_src = model(src_imgs, disp)  # [b,s,4,h,w]

    h, w = mpi_all_src.shape[-2:]
    swing_path_list = gen_swing_path()
    src_pose = swing_path_list[0]
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
    mpi_all_rgb_src = blend_weights * \
        src_imgs.unsqueeze(1) + (1 - blend_weights) * mpi_all_rgb_src
    min_epe = 100
    for idx in tqdm(range(repeat)):
        cam_ext, camera_mot, camera_ang = generate_random_pose2(base_motions=[0, 0, 0])
        cam_ext_dynamic, camera_mot_d, camera_ang_d = generate_random_pose2()
        obj_mask = obj_masks[idx % len(obj_masks)]
        obj_mask_np = obj_mask.squeeze().cpu().numpy()
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
            hard_flow
        )
        flowA2B = flowA2B.to(torch.float32)
        oH, oW = flowA2B.shape[2:]
        H, W = valid.shape[:2]
        flowA2B[:, 0] = flowA2B[:, 0] * W/oW
        flowA2B[:, 1] = flowA2B[:, 1] * H/oH
        
        frame = F.interpolate(frame, size=(H, W),
                            mode='bilinear', align_corners=True)
        flowA2B = F.interpolate(flowA2B, size=(H, W),
                            mode='bilinear', align_corners=True)
        mask = F.interpolate(mask, size=(H, W),
                            mode='bilinear', align_corners=True)
        depth = F.interpolate(depth, size=(H, W),
                            mode='bilinear', align_corners=True)

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
            hard_flow
        )
        
        flowA2B_dync = flowA2B_dync.to(torch.float32)
        flowA2B_dync[:, 0] = flowA2B_dync[:, 0] * W/oW
        flowA2B_dync[:, 1] = flowA2B_dync[:, 1] * H/oH
        
        frame_dync = F.interpolate(frame_dync, size=(H, W),
                            mode='bilinear', align_corners=True)
        flowA2B_dync = F.interpolate(flowA2B_dync, size=(H, W),
                            mode='bilinear', align_corners=True)
        mask_dync = F.interpolate(mask_dync, size=(H, W),
                            mode='bilinear', align_corners=True)
        depth_dync = F.interpolate(depth_dync, size=(H, W),
                            mode='bilinear', align_corners=True)
        
        frame_np = frame[0].permute(
            1, 2, 0).cpu().numpy().astype(np.float32)   # [b,h,w,3]
        frame_np = np.clip(np.round(frame_np * 255),
                           a_min=0, a_max=255).astype(np.uint8)[:, :, [2, 1, 0]]

        frame_dync_np = frame_dync[0].permute(
            1, 2, 0).cpu().numpy().astype(np.float32)   # [b,h,w,3]
        frame_dync_np = np.clip(np.round(frame_dync_np * 255),
                                a_min=0, a_max=255).astype(np.uint8)[:, :, [2, 1, 0]]
        mask = mask[0].permute(
            1, 2, 0).cpu().squeeze().numpy().astype(np.float32)   # [b,h,w,3]
        mask_dync = mask_dync[0].permute(
            1, 2, 0).cpu().squeeze().numpy().astype(np.float32)   # [b,h,w,3]

        flow_np = flowA2B[0].permute(
            1, 2, 0).contiguous().cpu().numpy().astype(np.float32)   # [b,h,w,3]
        flow_dync_np = flowA2B_dync[0].permute(
            1, 2, 0).contiguous().cpu().numpy().astype(np.float32)  # [b,h,w,3]

        # A2B 光流 mask
        # import IPython
        # IPython.embed()
        # exit()
        obj_mask_np = cv2.resize(obj_mask_np.astype(np.uint8), (W, H))
        # flow_np[obj_mask_np < mask_thresh] = 0
        # flow_dync_np[obj_mask_np >= mask_thresh] = 0
        flow_color = flow_to_color(flow_np, convert_to_bgr=True)
        flow_dync_color = flow_to_color(flow_dync_np, convert_to_bgr=True)
        frame_np[mask < mask_thresh] = 255
        frame_dync_np_copy = frame_dync_np.copy()
        frame_dync_np[mask_dync < mask_thresh] = 255
        frame_mix = frame_dync_np.copy()
        frame_mix[mask >= mask_thresh] = frame_np[mask >= mask_thresh]
        flow_mix = flow_dync_np.copy()
        flow_mix[obj_mask_np >= mask_thresh] = flow_np[obj_mask_np >= mask_thresh]
        flow_mix_color = flow_to_color(flow_mix, convert_to_bgr=True)

        fill_mask = mask_dync.copy()
        fill_mask[mask >= mask_thresh] = 1

        fill_mask = (fill_mask < mask_thresh).astype(np.int32)

        mix_mask = np.logical_and(mask, mask_dync).astype(np.uint8)
        # depth
        depth = depth.squeeze().cpu().numpy()
        # depth[mask < mask_thresh] = 255
        depth_dync = depth_dync.squeeze().cpu().numpy()
        # depth_dync[mask < mask_thresh] = 255

        depth_mask = np.logical_and(depth > depth_dync, mix_mask)
        frame_mix_depth = frame_mix.copy()
        frame_mix_depth[depth_mask] = frame_dync_np[depth_mask]

        inpainted = cv2.inpaint(
            frame_mix, fill_mask.astype(np.uint8), 3, cv2.INPAINT_NS)
        fill_mask_depth = np.logical_and(mask_dync < mask_thresh, depth_mask)
        fill_mask_depth = np.logical_or(fill_mask_depth, fill_mask)
        # fill_mask_depth[depth_mask] = 0
        # fill_mask_depth = fill_mask
        frame_mix_depth_inpainted = cv2.inpaint(
            frame_mix_depth, fill_mask_depth.astype(np.uint8), 3, cv2.INPAINT_NS)

        obj_mask_np = obj_mask.squeeze().cpu().numpy()
        obj_mask_np = obj_mask_np.astype(np.uint8) * 255
        # import IPython
        # IPython.embed()
        # exit()
        obj_mask_np = cv2.merge([obj_mask_np, obj_mask_np, obj_mask_np])
        tgt_np = cv2.resize(tgt_np, (W, H))
        obj_mask_np = cv2.resize(obj_mask_np, (W, H))
        src_np = cv2.resize(src_np, (W, H))
        gt = np.vstack([src_np, tgt_np, frame_dync_np_copy])
        res = np.vstack([flow_dync_color, inpainted, np.abs(inpainted.astype(np.float32) - tgt_np.astype(np.float32))]).copy()
        flow_mix_color_raw = flow_mix_color.copy()
        flow_mix_color[in_valid > 0] = 0
        error = np.abs(floc.astype(np.float32)-flow_mix_color.astype(np.float32))
        # error[in_valid > 0] = 0

        epe_b = np.sqrt(np.mean((flo[valid > 0] - flow_mix[valid > 0])**2))
        if epe_b < min_epe:
            min_epe = epe_b
            print('[update epe]', epe_b)
            print(camera_mot_d)
            print(camera_ang_d)
            print(camera_mot)
            print(camera_ang)
            print('---------------------------------------')
            cv2.putText(error, 'EPE: {:.2f}'.format(epe_b), (50, 80),
                        cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 5)
            flo_v = np.vstack([flow_mix_color_raw, floc, error*2])
            import imutils
            res = np.hstack([gt, res, flo_v])
            H, W = res.shape[:2]
            # cv2.imwrite('temp/a.png', imutils.resize(res,
            #             width=int(W/2), height=int(H/2)))
            cv2.imwrite('temp/a.png', res)
            data_path = 'temp'
            writeFlow(os.path.join(data_path, 'flo',
                                '{}_{}_{}.flo'.format(extra, name, idx)), flow_mix)
            cv2.imwrite(os.path.join(data_path, 'image_1',
                        '{}_{}_{}.png'.format(extra, name, idx)), src_np)
            cv2.imwrite(os.path.join(data_path, 'image_2',
                        '{}_{}_{}.png'.format(extra, name, idx)), inpainted)
    raise


def generate_random_pose2(base_motions=[0.05, 0.05, 0.05]):
    scx = ((-1)**random.randrange(2))
    scy = ((-1)**random.randrange(2))
    scz = ((-1)**random.randrange(2))
    if base_motions[0] == 0.05:
        c1, c2, c3 = [0.019908294054150358, -0.0013590704254421004, -0.0054824534806479225]
        cx = (random.random()*0.02) * scx + c1
        cy = (random.random()*0.02) * scy + c2
        cz = (random.random()*0.02) * scz + c3

        camera_mot = [cx, cy, cz]
        camera_mot = [c1, c2, c3]
        
        sax = ((-1)**random.randrange(2))
        say = ((-1)**random.randrange(2))
        saz = ((-1)**random.randrange(2))
        # Random angles in -pi/18,pi/18, excluding -pi/36,pi/36 to avoid zeros / very small rotations
        a1, a2, a3 = [-0.010197231871060683, 0.03575381384635365, 0.010695149122950507]
        ax = (random.random()*math.pi / 36.0*0.2) * sax + a1
        ay = (random.random()*math.pi / 36.0*0.2) * say + a2
        az = (random.random()*math.pi / 36.0*0.2) * saz + a3
        camera_ang = [ax, ay, az]
        camera_ang = [a1, a2, a3]
    else:
        # scz = 1
        # Random scalars in -0.2,0.2, excluding -0.1,0.1 to avoid zeros / very small motions
        c1, c2, c3 = [-0.025708628162462034, 0.05177329034524801, -0.029273890982408526]
        cx = (random.random()*0.03) * scx + c1
        cy = (random.random()*0.03) * scy + c2
        cz = (random.random()*0.03) * scz + c3
        
        # cx = (random.random()*0.2) * scx
        # cy = (random.random()*0.2) * scy
        # cz = (random.random()*0.2) * scz
        
        camera_mot = [cx, cy, cz]
        camera_mot = [c1, c2, c3]
        # generate random triplet of Euler angles
        # Random sign
        sax = ((-1)**random.randrange(2))
        say = ((-1)**random.randrange(2))
        saz = ((-1)**random.randrange(2))
        # Random angles in -pi/18,pi/18, excluding -pi/36,pi/36 to avoid zeros / very small rotations
        a1, a2, a3 = [-0.029484299732337824, 0.030514200304787017, 0.005960222614354038]
        ax = (random.random()*math.pi / 36.0*0.2) * sax + a1
        ay = (random.random()*math.pi / 36.0*0.2) * say + a2
        az = (random.random()*math.pi / 36.0*0.2) * saz + a3
        
        # ax = (random.random()*math.pi / 36.0) * sax
        # ay = (random.random()*math.pi / 36.0) * say
        # az = (random.random()*math.pi / 36.0) * saz
        camera_ang = [ax, ay, az]
        camera_ang = [a1, a2, a3]
        # camera_mot = [0, 0, 0]
        # camera_ang = [0, 0, 0]

    axisangle = torch.from_numpy(
        np.array([[camera_ang]], dtype=np.float32)).cuda().float()
    translation = torch.from_numpy(
        np.array([[camera_mot]])).cuda().float()
    cam_ext = transformation_from_parameters(
        axisangle, translation)[0]
    return cam_ext, camera_mot, camera_ang
