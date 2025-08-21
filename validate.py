import torch
import torch.nn as nn
import numpy as np
from utils import get_ball_xy

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = []
    dist = []
    tp = [0, 0, 0, 0]
    fp = [0, 0, 0, 0]
    tn = [0, 0, 0, 0]
    fn = [0, 0, 0, 0]

    total_visibility = [0, 0, 0, 0]     # for debugging
    
    with torch.no_grad():
        for iter_id, (imgs, gt_output, label) in enumerate(val_loader):
            imgs = imgs.to(device)
            gt_output = gt_output.to(device)
            label = label.to(device)
            
            outputs = model(imgs)
            gt_output = gt_output.long()
            loss = criterion(outputs, gt_output)
            running_loss.append(loss.item())
            
            softmax_layers = nn.Softmax(dim=1)(outputs)
            depth_indices = torch.argmax(softmax_layers, dim=1)

            circles = get_ball_xy(depth_indices, threshold=128)
            for idx, (x, y) in enumerate(circles):
                visibility = int(label[idx][0].item())
                total_visibility[visibility] += 1
                x_coord = label[idx][1].item()
                y_coord = label[idx][2].item()
                status = int(label[idx][3].item())

                if not x or not y:
                    if visibility == 0:
                        tn[visibility] += 1
                    else:
                        fn[visibility] += 1
                else:
                    x_dist = (x_coord - x)**2
                    y_dist = (y_coord - y)**2
                    distance = np.sqrt(x_dist + y_dist)
                    
                    if visibility == 0 or distance >= 5.0:
                        fp[visibility] += 1
                    else:
                        tp[visibility] += 1
                        x_dist = (x_coord - x)**2
                        y_dist = (y_coord - y)**2
                        dist.append(np.sqrt(x_dist + y_dist))
            
    return np.mean(running_loss), np.mean(dist), tp, fp, tn, fn, total_visibility
