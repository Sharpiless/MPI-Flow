import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from tqdm import tqdm
import sys
import cv2
from core.utils import flow_viz

def readFlowKITTI(filename):
    flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)
    flow = flow[:,:,::-1].astype(np.float32)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2**15) / 64.0
    return flow, valid

def flow_uv_to_colors(n, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((n.shape[0], 3))

    colorwheel = flow_viz.make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    # rad = np.sqrt(np.square(u) + np.square(v))
    # a = np.arctan2(-v, -u)/np.pi   # 角度
    # H*W, 范围(-1, 1)
    a = n / np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1

        # idx = (rad <= 1)
        # col[idx]  = 1 - rad[idx] * (1-col[idx])
        # col[~idx] = col[~idx] * 0.75   # out of range

        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        # flow_image[:,ch_idx] = np.floor(255 * col)
        flow_image[:, ch_idx] = col
    return flow_image


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


def histogram_plot(data, name):
    # 归一化
    # Min = np.min(data)
    # Max = np.max(data)
    # print(Min, Max)
    # data = (data - Min) / (Max - Min)
    # data.sort()
    # data = np.clip(data, 0, 1)

    # n, bins, patches = plt.hist(x=data, bins='auto', color='#0504aa',
    #                             alpha=0.7, rwidth=0.85)
    plt.hist(x=data, bins=64, color='#0504aa',
             alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(name)
    # maxfreq = n.max()
    # # 设置y轴的上限
    # plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    # plt.ylim([0,1500])
    # plt.xlim([0, 128])

    plt.savefig('temp.png')


def flow_count(flo):
    # flo = flo[0].permute(1, 2, 0).cpu().numpy()
    u = flo[:, :, 0]
    v = flo[:, :, 1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    # a = np.arctan2(-v, -u) / np.pi
    a = np.arctan2(-v, -u)

    a = np.array(a).ravel()
    return a


def circular_hist(ax, x, bins=128, density=True, offset=0, gaps=True):
    """
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    # Wrap angles to [-pi, pi)
    # x *= 180
    # x = (x + np.pi) % (2 * np.pi) - np.pi

    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins + 1)

    # Bin data and record counts
    # n, bins = np.histogram(x, bins=bins)
    n, bins = np.histogram(x, bins=bins, range=(-np.pi, np.pi))

    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        # radius = (area / np.pi) ** .5
        radius = (area / np.pi)
        # radius = (area)
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax
    # color_list = np.random.rand(len(bins[:-1]), 3)
    color_list = flow_uv_to_colors(np.array(bins[:-1]))
    # patches = ax.bar(bins[:-1], radius, zorder=2, align='edge', width=widths, color='C0',
    #                  edgecolor='C0', fill=True, linewidth=0.1)
    patches = ax.bar(bins[:-1], radius, zorder=2, align='edge', width=widths, color=color_list,
                     edgecolor=color_list, fill=True, linewidth=0.1)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    return n, bins, patches


def demo(base):
    count_list = []
    flow_list = [v for v in os.listdir(base) if v.endswith('.png')]
    for flow_up in tqdm(flow_list):
        flow_up, _ = readFlowKITTI(os.path.join(base, flow_up))
        # flow_up[:, :, 0] *= 854 / 1280
        # flow_up[:, :, 1] *= 480 / 384
        # flow_up = readFlow(os.path.join(base, flow_up))
        count = flow_count(flow_up)
        count = np.random.choice(count, size=1000, replace=False)
        count = np.clip(count, -5, 5)
        count_list.append(count)
        # n, bins = np.histogram(count, bins=bins, range=(-np.pi, np.pi))
        # count = len(n[n >= 20])
        # count_list.append(count)

    count_list = np.array(count_list).ravel()
    count_list = np.load('temp2.npy')
    np.save('temp2.npy', count_list)
    histogram_plot(count_list, 'angle')
    plt.cla()
    # fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection='polar'))
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(
        projection='polar'), figsize=(3, 3), dpi=200)
    # Visualise by area of bins
    # plt.arrow(lw = 5)
    # ax.grid(linewidth=3)
    ax.xaxis.grid(linewidth=3)

    SMALL_SIZE = 15
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 25
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    circular_hist(ax, count_list, bins=512)  # 原来的bin 512
    plt.savefig('temp2.png')
    return count_list


if __name__ == '__main__':

    demo('depthstillation-davis')
    # demo('../dataset/Sintel/flow/')
    # demo('RealFlow-data/flo')
    # demo('MF-kitti-train')
    # demo('MF-davis')
    # demo('/share/zhouzhengguang/backup/data/flyingthings3d')