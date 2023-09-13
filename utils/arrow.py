import argparse, os, time, random, torch, cv2
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import pdb
import sys
sys.path.append('core')
import zipfile
import pickle


def pack_img_into_bytes(img,kind = "jpg",ratio=95,png_ratio = 1 ):
    if kind == "jpg":
        params  = [cv2.IMWRITE_JPEG_QUALITY, ratio]
        msg     = cv2.imencode(".jpg", img, params)[1]
    elif kind == "png":
        params  = [cv2.IMWRITE_PNG_COMPRESSION, png_ratio]
        msg     = cv2.imencode(".png", img, params)[1]

    byte_msg = (np.array(msg)).tobytes()
    return byte_msg

def load_img_from_bytes(byte_msg):
    img = cv2.imdecode(np.frombuffer(byte_msg, np.uint8), cv2.IMREAD_COLOR)
    return img

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
def raft_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--raft_model', default='/share/zhouzhengguang/RealFlow/pre-trained/raft-RFDAVIS.pth', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--outpath', type=str, default='flow')
    parser.add_argument('--resize', action='store_true', help='resize image')
    parser.add_argument('--ratio', type=int, default=1)
    parser.add_argument('--iter', type=int, default=3)
    return parser.parse_args()

def load_raft_model(args, d=0):
    model = torch.nn.DataParallel(RAFT(args))
    if args.small:
        model.load_state_dict(torch.load('/share/zhouzhengguang/code/vfi/RAFT/raft-small.pth'))
    else:
        model.load_state_dict(torch.load('/share/zhouzhengguang/code/vfi/RAFT/raft-things.pth'))
    model = model.module
    model.to(torch.device('cuda:{}'.format(d)))
    model.eval()
    return model 

def raft_infer_return_flow(model, I0, I1, h, w, shape=320, iters=5, return_tensor=False):
    st=time.time();  starter.record()
    _, flow2 = model(I0, I1, iters=iters, test_mode=True)
    flow2 = torch.nn.functional.interpolate(input=flow2, size=(h, w), mode='bilinear', align_corners=False)
    flow2 = flow2[0].cpu().numpy().transpose(1, 2, 0)
    ender.record();  torch.cuda.synchronize(); curr_time = starter.elapsed_time(ender)
    print('raft flow', abs(flow2).sum(2).max(), time.time()-st, curr_time/1000, I0.shape, flow2.shape)
    return flow2

def arrowon(img, flow, step=32):
    h, w = flow.shape[:2]
    img2 = img.copy()
    for i in range(step//2, h, step):
        for j in range(step//2, w, step):
            dstx, dsty = int(i+flow[i,j,1]), int(j+flow[i,j,0])
            cv2.arrowedLine(img2,(j,i), (dsty,dstx), (0,0,255),2,8,0,0.2)
    return img2

def read_zip(args, vname=None, zipname=None, prefix='', flen=None):
    if vname is None:
        vname = args.video.split('/')[-1]
    if zipname is None:
        zipname = '/share/zhouzhengguang/VFI/raftflow_zip/{}.zip'.format(vname)
    else:
        zipname = '/share/zhouzhengguang/VFI/raftflow_zip/{}'.format(zipname)
    flow_list = []
    with zipfile.ZipFile(zipname) as myzip:
        nlist = myzip.namelist(); # pdb.set_trace()
        assert prefix+'{}_minmax.txt'.format(vname) in nlist 
        flen = len(nlist)-1 if flen is None else flen
        for i in range(flen):
            name = prefix+vname+'_{:06d}.jpg'.format(i)
            with myzip.open(name) as myfile:
                img = load_img_from_bytes(myfile.read())
                print(img.shape)
                flow_list.append(img)
        name = prefix+vname+'_minmax.txt'
        with myzip.open(name) as myfile:
            minmax_list = [l.decode("utf-8").strip() for l in myfile.readlines()]
    print(vname, len(flow_list), len(minmax_list));  #pdb.set_trace()
    return flow_list, minmax_list

def video_to_flow(args):
    flow_list, minmax_list = read_zip(args)
    t0=time.time()
    videogen = cv2.VideoCapture(args.video)
    fps = videogen.get(cv2.CAP_PROP_FPS)
    tot_frame = videogen.get(cv2.CAP_PROP_FRAME_COUNT);  print('video tot frames', tot_frame)
    ret, lastframe = videogen.read();  h, w = lastframe.shape[:2];  print(h, w);  
    print(h, w)
    # lastframe = cv2.resize(lastframe, (w,h))
    fh, fw = flow_list[0].shape[:2]
    tnum=0; 
    vname = args.video[:-4].split('/')[-1]+'_zip'
    flow_path = os.path.join(args.outpath, vname); os.makedirs(flow_path, exist_ok=True)
    forward_flow_txt = open(flow_path+'_flow_minmax.txt', 'w')
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    vid_out = cv2.VideoWriter(flow_path+'_arrow.mp4', fourcc, 30, (w*2, h))
    step = 48//args.ratio
    while True:
        ret, frame = videogen.read()
        if not ret:
            break
        flow1 = flow_list[tnum];  flow1 = cv2.resize(flow1, (w, h))
        flow1 = np.float32(flow1[:,:,:2]); 
        flow_minmax = [eval(x) for x in minmax_list[tnum].split(' ')];  tnum+=1
        flow1 = np.float32(flow1)/255.*(flow_minmax[1]-flow_minmax[0])+flow_minmax[0]
        flow1[:,:,0]*=w/fw;  flow1[:,:,1]*=h/fh
        img2 = arrowon(lastframe, flow1, step)
        # cv2.imwrite('{}/{:05d}_arrow.png'.format(flow_path, tnum), img2)
        vid_out.write(np.hstack([lastframe, img2]))
        # z.writestr("{}.png".format(tnum), pack_img_into_bytes(flow1, "jpg", ratio=90))
        # if tnum>=90: break
        lastframe = frame 
        print(tnum)
        
    # z.close()
    vid_out.release();  print(tnum, time.time()-t0)
    print(flow_path+'_arrow.mp4')

def compress_jpg(img):
    # img = cv2.resize(img, (360, 640))
    params  = [cv2.IMWRITE_JPEG_QUALITY, 90]
    msg     = cv2.imencode(".jpg", img, params)[1]
    byte_msg = (np.array(msg)).tobytes()
    img = cv2.imdecode(np.frombuffer(byte_msg, np.uint8), cv2.IMREAD_COLOR)
    return img 

def diff_jpg_flow(args):
    # flow_list, minmax_list = read_zip(args)
    # flow_list2, minmax_list2 = read_zip(args, vname='1659786838550_4.mov', zipname='bae5.zip')
    flow_list, minmax_list = read_zip(args, vname='1659957120133_2.mov', zipname='99d1dd48f2c549e5b10bb0379b5dec31.zip', prefix='99d1dd48f2c549e5b10bb0379b5dec31/', flen=215)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device) 
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)
    raft_model = load_raft_model(args)
    t0=time.time()
    videogen = cv2.VideoCapture(args.video)
    fps = videogen.get(cv2.CAP_PROP_FPS)
    tot_frame = videogen.get(cv2.CAP_PROP_FRAME_COUNT)
    ret, lastframe = videogen.read();  h, w = lastframe.shape[:2];  print(h, w); 
    ratio = max(240/h, 240/w)   #  short side 240
    h2, w2 = int(h*ratio), int(w*ratio)
    tmp = 8
    h2 = ((h2 - 1) // tmp + 1) * tmp
    w2 = ((w2 - 1) // tmp + 1) * tmp
    print(h2, w2)
    lastframe2 = cv2.resize(lastframe, (w2,h2))
    I1 = torch.from_numpy(np.transpose(lastframe2, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() 
    tnum=0;  fh, fw = flow_list[0].shape[:2];  flow3 = np.zeros((h, w, 1))
    
    vname = args.video[:-4].split('/')[-1]+'_diff_zip'
    flow_path = os.path.join(args.outpath, vname); os.makedirs(flow_path, exist_ok=True); print(flow_path)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    vid_out = cv2.VideoWriter(flow_path+'_arrow.mp4', fourcc, 30, (w*2, h))
    step = 48//args.ratio
    while True:
        ret, frame = videogen.read()
        if not ret or tnum>=len(flow_list)-1:
            break
        frame = compress_jpg(frame)
        frame2 = cv2.resize(frame, (w2, h2))
        flow1 = flow_list[tnum];  flow1 = cv2.resize(flow1, (w, h));  flow2 = flow1.copy()
        flow1 = np.float32(flow1[:,:,:2]); 
        flow_minmax = [eval(x) for x in minmax_list[tnum].split(' ')]; 
        flow1 = np.float32(flow1)/255.*(flow_minmax[1]-flow_minmax[0])+flow_minmax[0]
        flow1[:,:,0]*=w/fw;  flow1[:,:,1]*=h/fh; print(tnum, 'flow1', flow1.min(), flow1.max())
        img2 = arrowon(lastframe.copy(), flow1, step)

        tnum+=1
        # I0 = I1
        # I1 = torch.from_numpy(np.transpose(frame2, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() 
        # flow1 = raft_infer_return_flow(raft_model, I0, I1, h, w, shape=320, iters=3); print(tnum, flow_minmax, flow1.min(), flow1.max())
        # flow1[:,:,0]*=w/I0.shape[3];  flow1[:,:,1]*=h/I0.shape[2]; print(tnum,  'flow2', flow1.min(), flow1.max())
        # img1 = arrowon(lastframe.copy(), flow1, step)
        # if tnum<=300:
        #     if not os.path.exists('{}/{:05d}_flow.png'.format(flow_path, tnum)):
        #         flow1_max, flow1_min = flow1.max(), flow1.min(); 
        #         flow1 = (flow1-flow1_min)/(flow1_max-flow1_min)*255
        #         flow1 = (np.concatenate((flow1, flow3), axis=2)).astype(np.uint8)
        #         cv2.imwrite('{}/{:05d}_flow.png'.format(flow_path, tnum), np.hstack([lastframe, flow1, flow2]))
        #         cv2.imwrite('{}/{:05d}_arrow.png'.format(flow_path, tnum), np.hstack([img1, img2]))
        # else: break
        # vid_out.write(np.hstack([img1, img2]))
        vid_out.write(np.hstack([img2, flow2]))
        lastframe = frame
        
    vid_out.release();  print(tnum, time.time()-t0)
    print(flow_path+'_arrow.mp4')


def video_to_flow_lh(args):
    # flow_list2, minmax_list2 = read_zip(args, vname='1659698231974_2.mov', prefix='36fe604473724552ac5b4ccea95bfe1f/', flen=299)
    flow_list2, minmax_list2 = read_zip(args, zipname='skiing_1_360p.mov_raftjpg3.zip')
    flow_list, minmax_list = read_zip(args, zipname='skiing_1_360p.mov_jpg.zip')
    # flow_list, minmax_list = read_zip(args)
    
    t0=time.time()
    videogen = cv2.VideoCapture(args.video)
    fps = videogen.get(cv2.CAP_PROP_FPS)
    tot_frame = videogen.get(cv2.CAP_PROP_FRAME_COUNT);  print('video tot frames', tot_frame)
    ret, lastframe = videogen.read();  h, w = lastframe.shape[:2];  print(h, w);  
    print(h, w)
    fh, fw = flow_list[0].shape[:2]
    tnum=0; 
    vname = args.video[:-4].split('/')[-1]+'_zip_raftjpg2_360'
    flow_path = os.path.join(args.outpath, vname); os.makedirs(flow_path, exist_ok=True)
    forward_flow_txt = open(flow_path+'_flow_minmax.txt', 'w')
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    vid_out = cv2.VideoWriter(flow_path+'_arrow_lh.mp4', fourcc, 30, (w*2, h))
    step = 48//args.ratio
    while True:
        ret, frame = videogen.read()
        if not ret:
            break
        flow1 = flow_list[tnum];  flow1 = cv2.resize(flow1, (w, h))
        flow1 = np.float32(flow1[:,:,:2]); 
        flow_minmax = [eval(x) for x in minmax_list[tnum].split(' ')]; 
        flow1 = np.float32(flow1)/255.*(flow_minmax[1]-flow_minmax[0])+flow_minmax[0]
        flow1[:,:,0]*=w/fw;  flow1[:,:,1]*=h/fh
        img2 = arrowon(lastframe, flow1, step)

        flow1 = flow_list2[tnum];  flow1 = cv2.resize(flow1, (w, h))
        flow1 = np.float32(flow1[:,:,:2]); 
        flow_minmax = [eval(x) for x in minmax_list2[tnum].split(' ')];  tnum+=1
        flow1 = np.float32(flow1)/255.*(flow_minmax[1]-flow_minmax[0])+flow_minmax[0]
        flow1[:,:,0]*=w/fw;  flow1[:,:,1]*=h/fh
        img1 = arrowon(lastframe, flow1, step)
        cv2.imwrite('{}/{:05d}_arrow.png'.format(flow_path, tnum), np.hstack([img1, img2]))
        vid_out.write(np.hstack([img2, img1]))
        
        lastframe = frame 
        print(tnum)
        
    # z.close()
    vid_out.release();  print(tnum, time.time()-t0)
    print(flow_path+'_arrow.mp4')


def resize_diff(args):
    t0=time.time()
    videogen = cv2.VideoCapture(args.video)
    videogen2 = cv2.VideoCapture('/share/zhouzhengguang/VFI/raftflow_bmk/skiing_1_360p.mov')
    fps = videogen.get(cv2.CAP_PROP_FPS)
    tot_frame = videogen.get(cv2.CAP_PROP_FRAME_COUNT);  print('video tot frames', tot_frame)
    
    vname = args.video[:-4].split('/')[-1]+'_resize_diff'
    flow_path = os.path.join(args.outpath, vname); os.makedirs(flow_path, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # vid_out = cv2.VideoWriter(flow_path+'_arrow_lh.mp4', fourcc, 30, (w*2, h))
    tnum=0
    while True:
        ret, frame = videogen.read();  ret2, frame2 = videogen2.read()
        if not ret2:
            break
        frame1 = cv2.resize(frame, (640, 360))
        cv2.imwrite('{}/{:05d}_arrow.png'.format(flow_path, tnum), np.hstack([frame1, frame2]))
        tnum+=1; print(tnum)

def zip_flow_to_pkl(args):
    import pickle
    # flow_list2, minmax_list2 = read_zip(args, vname='1659753665777_2.mov', zipname='629d5069fbb24dd5a06d392bf6d6eab6.zip', prefix='629d5069fbb24dd5a06d392bf6d6eab6/', flen=299)
    flow_list2, minmax_list2 = read_zip(args, vname='skiing_1_360p.mov', zipname = 'skiing_1_360p.mov_jpg.zip')
    t0=time.time()
    pk_list = []
    for i in range(len(flow_list2)):
        flow1 = flow_list2[i];# print(flow1[20,20])
        flow1 = np.float32(flow1[:,:,:2]); 
        flow_minmax = [eval(x) for x in minmax_list2[i].split(' ')]; 
        flow1 = np.float32(flow1)/255.*(flow_minmax[1]-flow_minmax[0])+flow_minmax[0]
        pk_list.append(flow1)
        # mag = np.sum(np.abs(flow1[:,:,:2]), axis=2).reshape(-1)
        # mag_sort = np.sort(mag)
        # mmag = np.mean(mag_sort[-len(mag_sort)//4:])
        # pdb.set_trace()
    with open('flow/skiing_1_360p_3.pk', 'wb') as f:
        pickle.dump(pk_list, f)
        
        
def for_lh(args):
    jpg_path = '/share/zhouzhengguang/VFI/raftflow_zip/d97/d97476872e9045c285959a671723b315'
    with open(os.path.join(jpg_path, '1659786499453_2.mov_minmax.txt'), 'r') as f:
        minmax_list = [l.strip() for l in f.readlines()];  print(len(minmax_list))
    t0=time.time()
    videogen = cv2.VideoCapture(args.video)
    fps = videogen.get(cv2.CAP_PROP_FPS)
    tot_frame = videogen.get(cv2.CAP_PROP_FRAME_COUNT);  print('video tot frames', tot_frame)
    ret, lastframe = videogen.read();  h, w = lastframe.shape[:2];  print(h, w);  
    print(h, w)
    tmp = cv2.imread(os.path.join(jpg_path, '1659786499453_2.mov_{:06d}.jpg'.format(0)))
    fh, fw = tmp.shape[:2]
    tnum=0; 
    vname = args.video[:-4].split('/')[-1]+'_zip'
    flow_path = os.path.join(args.outpath, vname); os.makedirs(flow_path, exist_ok=True)
    forward_flow_txt = open(flow_path+'_flow_minmax.txt', 'w')
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    vid_out = cv2.VideoWriter(flow_path+'_arrow.mp4', fourcc, 30, (w, h))
    step = 48//args.ratio
    while True:
        ret, frame = videogen.read()
        if not ret:
            break
        flow1 = cv2.imread(os.path.join(jpg_path, '1659786499453_2.mov_{:06d}.jpg'.format(tnum)))
        flow1 = cv2.resize(flow1, (w, h))
        flow1 = np.float32(flow1[:,:,:2]); 
        flow_minmax = [eval(x) for x in minmax_list[tnum].split(' ')];  tnum+=1
        flow1 = np.float32(flow1)/255.*(flow_minmax[1]-flow_minmax[0])+flow_minmax[0]
        flow1[:,:,0]*=w/fw;  flow1[:,:,1]*=h/fh
        img2 = arrowon(lastframe, flow1, step)
        cv2.imwrite('{}/{:05d}_arrow.png'.format(flow_path, tnum), img2)
        # vid_out.write(np.hstack([lastframe, img2]))
        vid_out.write(img2)
        # z.writestr("{}.png".format(tnum), pack_img_into_bytes(flow1, "jpg", ratio=90))
        # if tnum>=90: break
        lastframe = frame 
        print(tnum)
        
    # z.close()
    vid_out.release();  print(tnum, time.time()-t0)
    print(flow_path+'_arrow.mp4')


def diff_kbs(args):
    device = torch.device('cuda:0')
    torch.cuda.set_device(device) 
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)
    raft_model = load_raft_model(args)
    t0=time.time()
    videogen = cv2.VideoCapture(args.video)
    fps = videogen.get(cv2.CAP_PROP_FPS)
    tot_frame = videogen.get(cv2.CAP_PROP_FRAME_COUNT)
    ret, lastframe = videogen.read();  h, w = lastframe.shape[:2];  print(h, w); 
    ratio = max(240/h, 240/w)   #  short side 240
    h2, w2 = int(h*ratio), int(w*ratio)
    tmp = 8
    h2 = ((h2 - 1) // tmp + 1) * tmp
    w2 = ((w2 - 1) // tmp + 1) * tmp
    print(h2, w2)
    lastframe2 = cv2.resize(lastframe, (w2,h2))
    I1 = torch.from_numpy(np.transpose(lastframe2, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() 
    tnum=0;  flow3 = np.zeros((h, w, 1))
    vname = args.video[:-4].split('/')[-1]+'_diff_kbs'
    flow_path = os.path.join(args.outpath, vname); os.makedirs(flow_path, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    vid_out = cv2.VideoWriter(flow_path+'_arrow.mp4', fourcc, 30, (w*2, h))
    step = 48//args.ratio
    while True:
        ret, frame = videogen.read()
        if not ret:
            break
        frame2 = cv2.resize(frame, (w2, h2))
        I0 = I1
        I1 = torch.from_numpy(np.transpose(frame2, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() 
        flow1 = raft_infer_return_flow(raft_model, I0, I1, h, w, shape=320, iters=3);
        flow1[:,:,0]*=w/I0.shape[3];  flow1[:,:,1]*=h/I0.shape[2]; print(tnum,  'flow2', flow1.min(), flow1.max())
        img1 = arrowon(lastframe.copy(), flow1, step)
        
        flow1_max, flow1_min = flow1.max(), flow1.min(); 
        flow1 = (flow1-flow1_min)/(flow1_max-flow1_min)*255
        flow1 = (np.concatenate((flow1, flow3), axis=2)).astype(np.uint8)

        vid_out.write(np.hstack([img1, flow1]))
        lastframe = frame
        tnum+=1
        
    vid_out.release();  print(tnum, time.time()-t0)
    print(flow_path+'_arrow.mp4')


def gen_flow_videos(args):
    device = torch.device('cuda:0')
    torch.cuda.set_device(device) 
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)
    raft_model = load_raft_model(args)
    t0=time.time()
    videogen = cv2.VideoCapture(args.video)
    fps = videogen.get(cv2.CAP_PROP_FPS)
    tot_frame = videogen.get(cv2.CAP_PROP_FRAME_COUNT)
    ret, lastframe = videogen.read();  h, w = lastframe.shape[:2];  print(h, w); 
    ssd = 360;  iters=3
    ratio = max(ssd/h, ssd/w)   #  short side 240
    h2, w2 = int(h*ratio), int(w*ratio)
    tmp = 8
    h2 = ((h2 - 1) // tmp + 1) * tmp
    w2 = ((w2 - 1) // tmp + 1) * tmp
    print(h2, w2)
    lastframe2 = cv2.resize(lastframe, (w2,h2))
    I1 = torch.from_numpy(np.transpose(lastframe2, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() 
    tnum=0;  flow3 = np.zeros((h2, w2, 1))
    vname = args.video[:-4].split('/')[-1]
    flow_name = os.path.join(args.outpath, vname)+'_raftflow_{}_iter{}'.format(ssd, iters)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # vid_out = cv2.VideoWriter(flow_name+'.mp4', fourcc, 30, (w2, h2))
    fw = open(flow_name+'.txt', 'w')
    pbar = tqdm(total=tot_frame-1); dlflow = []
    while True:
        ret, frame = videogen.read()
        if not ret:
            break
        frame2 = cv2.resize(frame, (w2, h2))
        I0 = I1
        I1 = torch.from_numpy(np.transpose(frame2, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() 
        _, flow2 = raft_model(I0, I1, iters=iters, test_mode=True)
        # flow2 = torch.nn.functional.interpolate(input=flow2, size=(h, w), mode='bilinear', align_corners=False)
        flow1 = flow2[0].cpu().numpy().transpose(1, 2, 0)
        dlflow.append(flow1)
        # flow1_max, flow1_min = flow1.max(), flow1.min(); 
        # flow1 = (flow1-flow1_min)/(flow1_max-flow1_min)*255
        # flow1 = (np.concatenate((flow1, flow3), axis=2)).astype(np.uint8)
        # fw.write(' '.join([str(x) for x in [flow1_min, flow1_max]])+'\n')
        # vid_out.write(flow1)
        # lastframe = frame
        # tnum+=1
        pbar.update(1)
        
    # vid_out.release();  print(tnum, )
    # print(flow_name)
    fw.close();  pbar.close()
    with open(flow_name+'.pk', 'wb') as f:
        pickle.dump(dlflow, f)
    print(flow_name, time.time()-t0)

def gen_smooth_weight():
    lx = 8
    x=['{:02d}*1'.format(i) for i in range(1, lx+1)]
    for i in range(5):
        y=[x[0]]
        for j in range(1,lx-1):
            if i>0:
                # pdb.set_trace()
                tmp1 = {a.split('*')[0]:eval(a.split('*')[-1]) for a in x[j-1].split('+')}
                tmp2 = {a.split('*')[0]:eval(a.split('*')[-1]) for a in x[j].split('+')}
                tmp3 = {a.split('*')[0]:eval(a.split('*')[-1]) for a in x[j+1].split('+')}
                klist = list(set(list(tmp1.keys())+list(tmp2.keys())+list(tmp3.keys())))
                new = {}
                for key in klist:
                    new[key] = 0.25*tmp1.get(key, 0)+0.5*tmp2.get(key, 0)+0.25*tmp3.get(key, 0)
                out = '+'.join([key+'*'+str(new[key]) for key in sorted(list(new.keys()))])
                # pdb.set_trace()
            else:
                out = '+'.join([x[j-1].split('*')[0]+'*'+str(0.25), x[j].split('*')[0]+'*'+str(0.5), x[j+1].split('*')[0]+'*'+str(0.25)])
            y.append(out)
        y.append(x[-1])
        x=y
        # print(i, x)
    print('='*50)
    for a in x:
        print(a)
        # print([eval(a1.split('*')[-1]) for a1 in a.split('+')])

flow_smooth_dict={
    1:[0.548828125, 0.12890625, 0.1611328125, 0.107421875, 0.04296875, 0.009765625, 0.0009765625],
    2:[0.2265625, 0.1611328125, 0.236328125, 0.2041015625, 0.1171875, 0.0439453125, 0.009765625, 0.0009765625],
    3:[0.0654296875, 0.107421875, 0.2041015625, 0.24609375, 0.205078125, 0.1171875, 0.0439453125, 0.009765625, 0.0009765625],
    4:[0.01171875, 0.04296875, 0.1171875, 0.205078125, 0.24609375, 0.205078125, 0.1171875, 0.0439453125, 0.009765625, 0.0009765625],
    5:[0.0009765625, 0.009765625, 0.0439453125, 0.1171875, 0.205078125, 0.24609375, 0.205078125, 0.1171875, 0.0439453125, 0.009765625, 0.0009765625]
    # -5:[0.0009765625, 0.009765625, 0.0439453125, 0.1171875, 0.205078125, 0.24609375, 0.205078125, 0.1171875, 0.04296875, 0.01171875]
    # -4:[0.0009765625, 0.009765625, 0.0439453125, 0.1171875, 0.205078125, 0.24609375, 0.2041015625, 0.107421875, 0.0654296875]
    # -3:[0.0009765625, 0.009765625, 0.0439453125, 0.1171875, 0.2041015625, 0.236328125, 0.1611328125, 0.2265625]
    # -2[0.0009765625, 0.009765625, 0.04296875, 0.107421875, 0.1611328125, 0.12890625, 0.548828125]
}

def smooth_diff():  #  ok
    kd = [0.0009765625, 0.009765625, 0.0439453125, 0.1171875, 0.205078125, 0.24609375, 0.205078125, 0.1171875, 0.0439453125, 0.009765625, 0.0009765625]
    x = list(np.random.randn(50));  x2 = x[:]
    for i in range(5):
        y=[x[0]]
        for j in range(1, 49):
            y.append(0.25*x[j-1]+0.5*x[j]+0.25*x[j+1])
        y.append(x[-1])
        x=y 

    for i in range(1):
        y2=[x2[0]]
        for j in range(1, 49):
            if j<5:
                y2.append(sum([x2[t]*flow_smooth_dict[j][t] for t in range(len(flow_smooth_dict[j]))]))
            elif j>=50-5:
                k = 50-j-1; # print(j, k, flow_smooth_dict[k]); #pdb.set_trace()
                y2.append(sum([x2[50-t-1]*flow_smooth_dict[k][t] for t in range(len(flow_smooth_dict[k]))]))
            else:
                # print(j)
                y2.append(sum([x2[j-5+t]*flow_smooth_dict[5][t] for t in range(11)]))
        y2.append(x2[-1])
        x2=y2
    print(y)
    print(y2)
    print(sum(np.array(y)-np.array(y2)))
    pdb.set_trace()

if __name__ == '__main__':
    args = raft_args()
    # video_to_flow(args)   #  python infer.py --video /share/zhouzhengguang/VFI/rife/demo_server/zm7.mp4
    # diff_jpg_flow(args)
    # video_to_flow_lh(args)
    # resize_diff(args)
    # zip_flow_to_pkl(args)
    # for_lh(args)
    # diff_kbs(args)
    # gen_smooth_weight()
    # smooth_diff()
    # gen_flow_videos(args)
    vpath = '/share/zhouzhengguang/VFI/raftflow_bmk/giveout'
    for name in os.listdir(vpath):
        args.video = os.path.join(vpath, name)
        gen_flow_videos(args)
