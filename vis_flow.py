import os
import cv2
import numpy as np

def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))

base = "dataset/debug"

if not os.path.exists(os.path.join(base, "vis")):
    os.mkdir(os.path.join(base, "vis"))

for img in os.listdir(os.path.join(base, "src_images")):
    for r in range(4):
        image1 = cv2.imread(os.path.join(base, "src_images", img))
        image2 = cv2.imread(os.path.join(base, "dst_images", img.replace(".png", f"_{r}.png")))
        flow= readFlow(os.path.join(base, "flows", img.replace(".png", f"_{r}.flo")))
        print(flow.max(), flow.min(), flow.shape)

        H, W = image1.shape[:2]
        res = np.vstack([image1, image2])

        for _ in range(30):
            
            x1 = np.random.randint(W)
            y1 = np.random.randint(H)
            x2 = x1 + int(flow[y1, x1, 0])
            y2 = y1 + int(flow[y1, x1, 1]) + H
            
            cv2.line(res, (x1, y1), (x2, y2), (0,255,0), 2)
            
        cv2.imwrite(os.path.join(base, "vis", img.replace(".png", f"_{r}.png")), res)