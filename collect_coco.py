import os
from tqdm import tqdm
from shutil import copy
import cv2

base = '/Extra/share/COCO/images/'
output = '/Extra/guowx/data/dCOCO-mpi/coco/training/image'

with open('dCOCO_file_list.txt', 'r') as f:
    data = f.read().splitlines()
    
maxH, maxW = 0, 0
    
for img in tqdm(data):
    src = os.path.join(base, img)
    dst = os.path.join(output, img.split('/')[-1])
    H, W = cv2.imread(src).shape[:2]
    maxH = max(maxH, H)
    maxW = max(maxW, W)
    # copy(src, dst)
print(maxH, maxH)