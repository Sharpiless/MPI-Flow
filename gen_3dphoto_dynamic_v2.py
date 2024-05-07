import argparse
import torch
import torch.nn.functional as F
import os
import cv2
from tqdm import tqdm
from torchvision.utils import save_image
from write_flow import writeFlow

from utils.utils import (
    image_to_tensor,
    disparity_to_tensor,
    render_3dphoto_dynamic,
)
from model.AdaMPI import MPIPredictor
from random import seed
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--width', type=int, default=1280)
parser.add_argument('--height', type=int, default=384)
parser.add_argument('--seed', type=int, default=114514)
parser.add_argument('--ext_cz', type=float, default=0.15)
parser.add_argument('--ckpt_path', type=str,
                    default='adampiweight/adampi_64p.pth')
parser.add_argument('--repeat', type=int, default=5)
parser.add_argument('--base', type=str,
                    default='', required=True)
parser.add_argument('--out', type=str,
                    default='', required=True)

opt, _ = parser.parse_known_args()

print(opt)

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
model.eval()
# model = torch.jit.script(model)

out = opt.out
base = opt.base

if not os.path.exists(out):
    os.mkdir(out)
    os.mkdir(f"{out}/src_images")
    os.mkdir(f"{out}/dst_images")
    os.mkdir(f"{out}/flows")
    os.mkdir(f"{out}/obj_mask")


img_base = os.path.join(base, "images")
disp_base = os.path.join(base, "disps")
mask_base = os.path.join(base, "masks")

for img in tqdm(sorted(os.listdir(img_base))):
    
    name = img.split(".")[0]
    
    image = image_to_tensor(os.path.join(img_base, img)).cuda().half()  # [1,3,h,w]
    obj_mask_np = np.array(Image.open(os.path.join(mask_base, img)).convert("L"))
    disp = disparity_to_tensor(os.path.join(disp_base, img)).cuda().half() # [1,1,h,w]
    
    image = F.interpolate(image, size=(opt.height, opt.width),
                        mode='bilinear', align_corners=True)
    disp = F.interpolate(disp, size=(opt.height, opt.width),
                        mode='bilinear', align_corners=True)
    
    # disp.requires_grad = True
    with torch.no_grad():
        mpi_all_src, disparity_all_src = model(image, disp)  # [b,s,4,h,w]
        
    # import IPython
    # IPython.embed()
    # exit()
        
    for r in range(opt.repeat):
        # predict MPI planes
        obj_index = np.random.randint(obj_mask_np.max()) + 1
        # print(obj_mask_np.max(), obj_index)
        obj_mask = torch.FloatTensor(obj_mask_np == obj_index).cuda().half().unsqueeze(0).unsqueeze(0)  # [1,3,h,w]
        obj_mask = F.interpolate(obj_mask, size=(opt.height, opt.width),
                            mode='bilinear', align_corners=True)
        
        flow_mix, src_np, inpainted, res = render_3dphoto_dynamic(
            opt,
            image,
            obj_mask,
            disp,
            mpi_all_src,
            disparity_all_src,
            K,
            K,
            data_path='outputs',
            name='demo'
        )

        writeFlow(os.path.join(out, "flows", f'{name}_{r}.flo'), flow_mix)
        cv2.imwrite(os.path.join(out, "dst_images", f'{name}_{r}.png'), inpainted)
        cv2.imwrite(os.path.join(out, "src_images", f'{name}_{r}.png'), src_np)