import argparse
import torch
import torch.nn.functional as F
import os

from utils.utils import (
    image_to_tensor,
    disparity_to_tensor,
    render_3dphoto_dynamic,
)
from model.AdaMPI import MPIPredictor
from random import seed
import numpy as np

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--image_path', type=str, default='images_kitti/000043_11.png')
parser.add_argument('--depth_path', type=str, default='images_kitti/depth/000043_11.png')
parser.add_argument('--mask_path', type=str, default='images_kitti/mask/000043_11_mask_00.png')
parser.add_argument('--width', type=int, default=1280)
parser.add_argument('--height', type=int, default=384)
parser.add_argument('--seed', type=int, default=114514)
parser.add_argument('--ckpt_path', type=str,
                    default='adampiweight/adampi_64p.pth')

opt, _ = parser.parse_known_args()

seed(opt.seed)
np.random.seed(opt.seed)

# render 3D photo
K = torch.tensor([
    [0.58, 0, 0.5],
    [0, 0.58, 0.5],
    [0, 0, 1]
]).cuda().half()
K[0, :] *= opt.width
K[1, :] *= opt.height
K = K.unsqueeze(0)

# load pretrained model
ckpt = torch.load(opt.ckpt_path)
model = MPIPredictor(
    width=opt.width,
    height=opt.height,
    num_planes=ckpt['num_planes'],
)
model.load_state_dict(ckpt['weight'])
model = model.cuda().half()
model = model.eval()

image = image_to_tensor(opt.image_path).cuda().half()  # [1,3,h,w]
obj_mask = image_to_tensor(opt.mask_path).cuda().half()  # [1,3,h,w]
disp = disparity_to_tensor(opt.depth_path).cuda().half() # [1,1,h,w]
image = F.interpolate(image, size=(opt.height, opt.width),
                      mode='bilinear', align_corners=True)
obj_mask = F.interpolate(obj_mask, size=(opt.height, opt.width),
                      mode='bilinear', align_corners=True)
disp = F.interpolate(disp, size=(opt.height, opt.width),
                     mode='bilinear', align_corners=True)
# disp.requires_grad = True
if image.shape[1] < 3:
    image = torch.cat([image, image, image], 1)
    
if not os.path.exists('outputs'):
    os.mkdir('outputs')
    
# predict MPI planes
render_3dphoto_dynamic(
    image,
    obj_mask,
    disp,
    model,
    K,
    K,
    data_path='outputs',
    name='demo'
)
