import argparse
import torch
import torch.nn.functional as F
import os

from utils.utils import (
    image_to_tensor,
    disparity_to_tensor,
    render_3dphoto_dynamic_objects,
)
from model.AdaMPI import MPIPredictor
from tqdm import tqdm


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, default="../dataset/DAVIS/")
# parser.add_argument('--width', type=int, default=384)
# parser.add_argument('--height', type=int, default=128)
parser.add_argument('--width', type=int, default=1152)
parser.add_argument('--height', type=int, default=640)
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
images = sorted(os.listdir(os.path.join(opt.data_path, 'images')))
print(opt.data_path)
for i, img in enumerate(tqdm(images[15:])):
    # load input
    print(img)
    img_path = os.path.join(opt.data_path, 'images', img)
    disp_path = os.path.join(opt.data_path, 'depth', img[:-4]+'.png')
    image = image_to_tensor(img_path).cuda().half()  # [1,3,h,w]
    obj_masks = []

    dst_img_path = os.path.join(opt.data_path, 'masks', img[:-4]+'.png'.format(i))
    # obj_mask = image_to_tensor(dst_img_path).cuda().half()  # [1,3,h,w]
    # obj_mask[:, :, :] = 0
    disp = disparity_to_tensor(disp_path).cuda().half() # [1,1,h,w]
    obj_mask = torch.zeros_like(disp)
    obj_mask = F.interpolate(obj_mask, size=(opt.height, opt.width),
                            mode='bilinear', align_corners=True)
    obj_masks.append(obj_mask)
    image = F.interpolate(image, size=(opt.height, opt.width),
                          mode='bilinear', align_corners=True)
    disp = F.interpolate(disp, size=(opt.height, opt.width),
                         mode='bilinear', align_corners=True)
    # disp.requires_grad = True
    if image.shape[1] < 3:
        image = torch.cat([image, image, image], 1)
    # predict MPI planes
    render_3dphoto_dynamic_objects(
        image,
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
        repeat=20
    )