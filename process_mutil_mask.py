import os
import cv2

base = '../kitti15/training/mask_2_multi/'
images = [v for v in os.listdir(base) if not 'mask' in v]

for img in images:
    dst = os.path.join(base, dst)
    for i in range(4):
        dst = 