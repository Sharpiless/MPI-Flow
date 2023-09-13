import numpy as np
import cv2
import glob
from utils import flow_viz
import torch
from torch.nn import functional as F

def gen_random_perspective():
    '''
    generate a random 3x3 perspective matrix
    '''
    init_M = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    noise = np.random.normal(0, 0.0001, 9)
    noise = np.reshape(noise, [3, 3])
    noise[2, 2] = 0
    return init_M + noise

def get_flow(img, M):
    '''
    use img shape and M  to calculate flow
    return flow
    '''
    ## calculate flow
    x = np.linspace(0, img.shape[1]-1, img.shape[1])
    y = np.linspace(0, img.shape[0]-1, img.shape[0])
    xx, yy = np.meshgrid(x, y)
    coords = np.stack([xx, yy, np.ones_like(xx)], axis=0)
    #new_coords = np.einsum('ij,jkl->ikl', M, np.transpose(coords, (2, 0, 1)))
    new_coords = np.einsum('ij,jkl->ikl', M, coords)
    xx2 = new_coords[0, :, :] / new_coords[2, :, :]
    yy2 = new_coords[1, :, :] / new_coords[2, :, :]
    #xx2 = xx2 * img.shape[1]
    #yy2 = yy2 * img.shape[0]
    #import pdb; pdb.set_trace()
    xx2 = xx2.astype(np.float32)
    yy2 = yy2.astype(np.float32)
    flow_x = xx2-xx #* img.shape[1]
    flow_y = yy2-yy #* img.shape[0]
    flow_x = flow_x.astype(np.float32)
    flow_y = flow_y.astype(np.float32)
    return np.stack([flow_x, flow_y], axis=2)

def transform(img, flow):
    '''
    remap image according to the M.
    return warped img and flow
    '''

    flow = flow.astype(np.float32)
    flow_x = flow[:, :, 0]
    flow_y = flow[:, :, 1]

    ## warp img by flow
    fh, fw = flow_x.shape
    add = np.mgrid[0:fh,0:fw].astype(np.float32);  

    img_flow = cv2.remap(img, flow_y+add[1,:,:], flow_x+add[0,:,:], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return img_flow

def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow

    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1 ,-1).repeat(H ,1)
    yy = torch.arange(0, H).view(-1 ,1).repeat(1 ,W)
    xx = xx.view(1 ,1 ,H ,W).repeat(B ,1 ,1 ,1)
    yy = yy.view(1 ,1 ,H ,W).repeat(B ,1 ,1 ,1)
    grid = torch.cat((xx ,yy) ,1).float()
    if x.is_cuda:
        grid = grid.cuda()
    vgrid = torch.autograd.Variable(grid).detach() + flo

    # scale grid to [-1,1]
    vgrid[: ,0 ,: ,:] = 2.0 *vgrid[: ,0 ,: ,:].clone() / max( W -1 ,1 ) -1.0
    vgrid[: ,1 ,: ,:] = 2.0 *vgrid[: ,1 ,: ,:].clone() / max( H -1 ,1 ) -1.0

    vgrid = vgrid.permute(0 ,2 ,3 ,1)
    flo = flo.permute(0 ,2 ,3 ,1)
    output = F.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = F.grid_sample(mask, vgrid).detach()

    #mask[mask <0.9999] = 0
    #mask[mask >0] = 1

    return output, mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backwarp_tenGrid = {}

def warp_rife(tenInput, tenFlow):
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device).view(
            1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device).view(
            1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[k] = torch.cat(
            [tenHorizontal, tenVertical], 1).to(device)

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)

if __name__ == "__main__":

    import sys
    #img_path = '/share/boyuan/Data/snow_good_imgs/01e1fbc1a563c467018370037ebf6cd3ae_258/000006.png'
    img_path = '/share/boyuan/Data/snow_good_imgs/skiing_0825/000006.png'
    seg_prefix = '/share/boyuan/Projects/mmdetection/output/skiing_0825/000006'
    img = cv2.imread(img_path)

    #all_mask = np.zeros((img.shape[0], img.shape[1]))
    res_flow = np.zeros((img.shape[0], img.shape[1], 2))
    for m_path in glob.glob(seg_prefix+"-*"):
        sub_mask = cv2.imread(m_path)[:, :, 0]
        sub_mask = sub_mask / 255
        sub_mask = sub_mask.astype(np.uint8)
        M = gen_random_perspective()
        sub_flow = get_flow(img, M)
        res_flow[np.where(sub_mask==1)] = sub_flow[np.where(sub_mask==1)]

    res_img = transform(img, res_flow)

    img_flo = flow_viz.flow_to_image(res_flow)
    out = np.concatenate([img, img_flo, res_img], axis=1)
    print(np.max(res_flow))
    cv2.imwrite('tmp/warp.jpg', out)
    cv2.imwrite('/share/boyuan/Projects/RAFT/tmp/skiing_per/000007.png', res_img)



