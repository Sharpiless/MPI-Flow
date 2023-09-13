import argparse
import torch
import torch.nn.functional as F
import os

from utils.utils import (
    image_to_tensor,
    disparity_to_tensor,
    render_3dphoto_SS,
)
from model.AdaMPI import MPIPredictor
from tqdm import tqdm


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, default="inputs_kitti/demo")
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
]).cuda()
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
model = model.cuda()
model = model.eval()
images = sorted(os.listdir(os.path.join(opt.data_path, 'image')))
for name, parameter in model.named_parameters():
    parameter.requires_grad = False
    
for i, img in enumerate(tqdm(images[:-1])):
    # load input
    img_path = os.path.join(opt.data_path, 'image', img.split('/')[-1])
    dst_img_path = os.path.join(opt.data_path, 'image', images[i+1].split('/')[-1])
    disp_path = os.path.join(opt.data_path, 'depth',
                             img.split('/')[-1][:-4]+'.png')
    image = image_to_tensor(img_path).cuda()  # [1,3,h,w]
    dst_image = image_to_tensor(dst_img_path).cuda()  # [1,3,h,w]
    disp = disparity_to_tensor(disp_path).cuda() # [1,1,h,w]
    image = F.interpolate(image, size=(opt.height, opt.width),
                          mode='bilinear', align_corners=True)
    dst_image = F.interpolate(dst_image, size=(opt.height, opt.width),
                          mode='bilinear', align_corners=True)
    disp = F.interpolate(disp, size=(opt.height, opt.width),
                         mode='bilinear', align_corners=True)
    # disp.requires_grad = True
    if image.shape[1] < 3:
        image = torch.cat([image, image, image], 1)
    # predict MPI planes
    render_3dphoto_SS(
        image,
        dst_image,
        disp,
        model,
        K,
        K,
        None,
        single=True,
        data_path='/Extra/guowx/data/kitti-flow-15/testing_sync/training/',
        name=img,
        extra='image_2',
        z_only=False,
        repeat=2000
    )
