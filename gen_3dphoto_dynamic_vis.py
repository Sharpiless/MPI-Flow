import argparse
import torch
import torch.nn.functional as F
import os

from utils.utils import (
    image_to_tensor,
    disparity_to_tensor,
    render_3dphoto_dynamic_objects_vis,
)
from model.AdaMPI import MPIPredictor
from tqdm import tqdm
import numpy as np
import cv2

def readFlowKITTI(filename, size=None):
    flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)

    flow = flow[:, :, ::-1].astype(np.float32)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2**15) / 64.0
    if not size is None:
        h, w = flow.shape[:2]
        print(h, w)
        scale_x = size[0] / w
        scale_y = size[1] / h
        flow = cv2.resize(flow, size, interpolation=1)
        valid = cv2.resize(valid, size, interpolation=1)
        print(scale_x, scale_y)
        flow[:, :, 0] *= scale_x
        flow[:, :, 1] *= scale_y
    return flow, valid

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, default="training-kitti")
# parser.add_argument('--width', type=int, default=384)
# parser.add_argument('--height', type=int, default=128)
parser.add_argument('--width', type=int, default=1280)
parser.add_argument('--height', type=int, default=384)
parser.add_argument('--ckpt_path', type=str,
                    default="adampiweight/adampi_64p.pth")
parser.add_argument("--hard_flow", dest="no_sharp",
                    action="store_true", help="Disable depth sharpening")

opt, _ = parser.parse_known_args()

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
    num_planes=ckpt["num_planes"],
)
model.load_state_dict(ckpt["weight"])
model = model.cuda().half()
model = model.eval()
images = sorted(os.listdir(os.path.join(opt.data_path, 'image_1')))
    
for i, img in enumerate(tqdm(images)):
    # load flow
    flo_path = os.path.join(opt.data_path, 'flo', img[:-4]+'.png')
    flo = readFlowKITTI(flo_path, (opt.width, opt.height))
    # load input
    img_path = os.path.join(opt.data_path, 'image_1', img)
    image = image_to_tensor(img_path).cuda().half()  # [1,3,h,w]
    image = F.interpolate(image, size=(opt.height, opt.width),
                          mode='bilinear', align_corners=True)
    # load target
    tgt_img_path = os.path.join(opt.data_path, 'image_2', img.replace('_10.png', '_11.png'))
    tgt_image = image_to_tensor(tgt_img_path).cuda().half()  # [1,3,h,w]
    tgt_image = F.interpolate(tgt_image, size=(opt.height, opt.width),
                          mode='bilinear', align_corners=True)
    obj_masks = []
    for i in range(1, 2):
        dst_img_path = os.path.join(opt.data_path, 'mask', img[:-4]+'_mask_{:02d}.png'.format(i))
        obj_mask = image_to_tensor(dst_img_path).cuda().half()  # [1,3,h,w]
        obj_mask = F.interpolate(obj_mask, size=(opt.height, opt.width),
                            mode='bilinear', align_corners=True)
        obj_masks.append(obj_mask)
    disp_path = os.path.join(opt.data_path, 'depth', img[:-4]+'.png')
    disp = disparity_to_tensor(disp_path).cuda().half() # [1,1,h,w]
    disp = F.interpolate(disp, size=(opt.height, opt.width),
                         mode='bilinear', align_corners=True)
    # disp.requires_grad = True
    if image.shape[1] < 3:
        image = torch.cat([image, image, image], 1)
    # predict MPI planes
    render_3dphoto_dynamic_objects_vis(
        flo,
        image,
        tgt_image,
        obj_masks,
        disp,
        model,
        K,
        K,
        None,
        single=True,
        data_path='temp',
        name=img,
        extra='image_2',
        z_only=False,
        repeat=500
    )

# [0.0045183388442011805, 0.012812617443831414, 0.16312378879060746] [0.025205111978071227, -0.024918800828944553, 0.03034400746481415]
# [0.054235124477263336, 0.07090040112944174, -0.20309710564985056] [0.02965591324458749, 0.02158969690574429, -0.019346799124299586]