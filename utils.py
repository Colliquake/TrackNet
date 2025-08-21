import torch
from matplotlib import pyplot as plt
import cv2
import numpy as np

def display_image(inp_tensor, img_pos=0):
    """
    img_pos of 0 will get the middle image. img_pos -1 will get the img BEFORE the middle img, 1 will get the img AFTER the middle img, and so on.
    inp_tensor will be of shape [c, h, w]
    """
    num_channels = inp_tensor.shape[0]
    num_imgs = num_channels // 3
    middle_idx = num_imgs - 1 - (num_imgs // 2)
    selected_idx = middle_idx + img_pos

    img = inp_tensor[3*selected_idx:3*selected_idx+3]
    img = torch.permute(img, (1, 2, 0))
    # img = img[:, :, [2, 1, 0]]
    plt.figure()
    # plt.imshow(img)
    plt.imshow(img.cpu().numpy())
    plt.show()

def get_ball_xy(depth_indices, threshold=128):
    ret = []
    for indices in depth_indices:
        heatmap = torch.where(indices >= threshold, torch.tensor(255), torch.tensor(0))
        heatmap = torch.reshape(heatmap, (360, 640))
        heatmap = heatmap.detach().cpu().numpy().astype(np.uint8)
        blurred_heatmap = cv2.GaussianBlur(heatmap, (7, 7), 3)
        blurred_heatmap = blurred_heatmap.astype(np.uint8)
        circles = cv2.HoughCircles(blurred_heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=30, param2=6, minRadius=2, maxRadius=8)
        x, y = None, None
        
        if circles is not None:
            if circles.shape[1] != 1:
                ret.append((None, None))
                continue
            x = circles[0][0][0]
            y = circles[0][0][1]
        ret.append((x, y))

    return ret