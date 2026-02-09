"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = np.clip(np.transpose(image_numpy, (1, 2, 0)), 0, 1) * 255.0
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

import torch.nn.functional as F
def rgb_to_grayscale(tensor):
    """将一个 RGB 图像张量转换为灰度张量。"""
    # 使用标准的亮度转换公式
    return 0.299 * tensor[:, 0:1, :, :] + 0.587 * tensor[:, 1:2, :, :] + 0.114 * tensor[:, 2:3, :, :]


def calculate_ssim(img1, img2, window_size=11, size_average=True):
    """计算两张图像之间的 SSIM。"""
    # 将图像转换为灰度图
    img1 = rgb_to_grayscale(img1)
    img2 = rgb_to_grayscale(img2)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    window = torch.ones(1, 1, window_size, window_size, device=img1.device) / (window_size * window_size)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=1)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=1)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=1) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=1) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

from skimage.metrics import structural_similarity

# def calculate_ssim(img1, img2):
#     """
#     使用 scikit-image 库计算两张 PyTorch 张量图像之间的 SSIM。
#     """
#     # 首先，使用项目中已有的 tensor2im 函数将 PyTorch 张量转换为 NumPy 数组。
#     # tensor2im 的输出格式是 (H, W, C)，数据类型是 uint8，范围 [0, 255]，这正是 scikit-image 所需要的。
#     img1_np = tensor2im(img1)
#     img2_np = tensor2im(img2)
#
#     # scikit-image 的 structural_similarity 函数可以直接处理多通道（彩色）图像。
#     # 我们需要设置 channel_axis=-1 (对于新版本) 或 multichannel=True (对于旧版本)。
#     # data_range 设为 255，因为我们的图像数据范围是 0-255。
#     # 该函数直接返回一个 float 类型的 SSIM 值。
#     try:
#         # 优先使用新版本的 channel_axis 参数，它更明确。
#         ssim_score = structural_similarity(img1_np, img2_np, channel_axis=-1, data_range=img1_np.max() - img1_np.min())
#     except TypeError:
#         # 如果上述调用失败（例如因为版本较旧不支持 channel_axis），则回退到使用 multichannel 参数。
#         ssim_score = structural_similarity(img1_np, img2_np, multichannel=True, data_range=img1_np.max() - img1_np.min())
#
#     return ssim_score
# --- 修改结束 ---