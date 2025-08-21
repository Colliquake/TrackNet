import math
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
import os
from natsort import natsorted
import pandas as pd
from PIL import Image

class TennisDataset(Dataset):
    def pixels_away_gaussian(self, sigma):
        """
            given sigma, determine how many orthogonal pixels away floor(G(x, y)) is greater than or equal to 128
        """
        p = math.sqrt(-2 * sigma * sigma * math.log(128 / 255))
        return math.floor(p)
    
    def create_ball_patch(self, sigma):
        p = self.pixels_away_gaussian(sigma)
            
        # create new patch
        one_row = np.zeros(p*2 + 1)
        for i in range(len(one_row)):
            one_row[i] = math.floor(((1/(2 * math.pi * sigma * sigma)) * math.exp(-((p - i) ** 2 ) / (2 * sigma * sigma))) * (2 * math.pi * sigma * sigma * 255)) / 255.0

        arr_rows = np.reshape(one_row, (len(one_row), 1))
        arr_cols = np.reshape(one_row, (1, len(one_row)))
        patch = np.multiply(arr_rows, arr_cols)
        patch = np.where(patch >= 0.5, patch, 0.0)
        patch = patch * 255.0
        patch = np.floor(patch)

        return patch

    def __init__(self, base_path, frames=1, resize=(360, 640), transform=None):
        self.base_path = base_path
        self.frames = frames
        self.resize = resize
        self.transform = v2.Compose([
            v2.Resize((resize[0], resize[1])),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])
        self.ball_patch = self.create_ball_patch(sigma=3)

        frames_ahead = frames // 2
        frames_behind = frames - 1 - frames_ahead

        data = []
        for game_name in natsorted(os.listdir(base_path)):
            game_path = os.path.join(base_path, game_name)
            if not os.path.isdir(game_path):
                continue

            for clip_name in natsorted(os.listdir(game_path)):
                clip_path = os.path.join(game_path, clip_name)
                if not os.path.isdir(clip_path):
                    continue

                # get labels, format their filenames, and combine them into one
                label_file_path = os.path.join(clip_path, 'Label.csv')
                label = pd.read_csv(label_file_path)
                label = label[frames_behind:-frames_ahead][:]
                label = label.fillna(-1)
                label = label.rename(columns={"file name": "filename"})
                label['filename'] = game_name + '_' + clip_name + '_' + label['filename']
                data.extend(label.values.tolist())

        self.labels = pd.DataFrame(data, columns=['filename', 'visibility', 'x-coordinate', 'y-coordinate', 'status'])
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        imgs = []
        label = self.labels.iloc[idx]
        
        filename = label.iloc[0]
        first_sep = filename.index("_")
        second_sep = filename.index("_", first_sep + 1)
        game_name = filename[:first_sep]
        clip_name = filename[first_sep+1:second_sep]
        img_num = int(filename[second_sep+1:filename.index(".")])

        frames_behind = self.frames - 1 - (self.frames // 2)
        start = img_num - frames_behind
        for i in range(start, start + self.frames):
            img_name = str(i).zfill(4) + ".jpg"
            img_path = os.path.join(self.base_path, game_name, clip_name, img_name)
            img = Image.open(img_path).convert('RGB')
            orig_w, orig_h = img.size
            if self.transform:
                img = self.transform(img)
            imgs.append(img)

        imgs = torch.cat(imgs, dim=0)

        # label consists of 'visibility', 'x-coordinate', 'y-coordinate', and 'status'
        label_vis = label.iloc[1]
        label_x = label.iloc[2]
        label_y = label.iloc[3]
        orig_x = label.iloc[2]
        orig_y = label.iloc[3]
        label_status = label.iloc[4]
        if label_x != -1:
            label_x = label_x * (self.resize[1] / orig_w)
        if label_y != -1:
            label_y = label_y * (self.resize[0] / orig_h)
        
        label = torch.tensor([label_vis, label_x, label_y, label_status], dtype=torch.float32)

        img_name = str(img_num).zfill(4) + ".jpg"
        path_gt = os.path.join(self.base_path, game_name, clip_name, img_name)
        gt_output = self.get_gt(path_gt, (label_x, label_y))

        return imgs, gt_output, label
    
    def get_gt(self, path_gt, ball_coord):
        img_x = self.resize[1]
        img_y = self.resize[0]
        (x, y) = ball_coord
        x = int(x)
        y = int(y)

        gt = np.zeros((self.resize[1], self.resize[0]), dtype=np.float32)
        if x == -1 or y == -1:
            return np.reshape(gt, self.resize[1] * self.resize[0])
        
        p = self.ball_patch.shape[0] // 2

        x1 = max(0, x - p)
        x2 = min(img_x, x + p + 1)
        y1 = max(0, y - p)
        y2 = min(img_y, y + p + 1)

        p1 = max(0, p - min(x, p))
        p2 = p1 + (x2 - x1)
        q1 = max(0, p - min(y, p))
        q2 = q1 + (y2 - y1)

        gt[x1:x2, y1:y2] = self.ball_patch[p1:p2, q1:q2]
        gt = np.reshape(gt, (640, 360))
        gt = np.rot90(gt, axes=(0, 1))
        gt = np.flip(gt, axis=0)
        
        return np.reshape(gt, self.resize[1] * self.resize[0])