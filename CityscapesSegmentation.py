import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
import re
from utils.cityscapes import CityscapesSegmentation

class Augm(data.Dataset):
    BASE_DIR = 'cityscapes'
    NUM_CLASS = 19

    def __init__(self, people_file, road_dataset):
        self.road_dataset = road_dataset

    def __getitem__(self, index):
        img, mask = self.road_dataset[index]

        # TODO
        return img, mask

    def __len__(self):
        return len(self.road_dataset)

nemodrive_road = CityscapesSegmentation()
nemodrive_aug = Augm("path", nemodrive_road)

img, mask = test[0]

def _get_cityscapes_pairs(folder, split='train'):
    def get_path_pairs(folder, split_f):
        img_paths = []
        mask_paths = []
        with open(split_f, 'r') as lines:
            for line in tqdm(lines):
                ll_str = re.split('\t', line)
                imgpath = os.path.join(folder, ll_str[0].rstrip())
                maskpath = os.path.join(folder, ll_str[1].rstrip())
                if os.path.isfile(maskpath):
                    img_paths.append(imgpath)
                    mask_paths.append(maskpath)
                else:
                    print('cannot find the mask:', maskpath)
        return img_paths, mask_paths

    if split == 'train':
        split_f = os.path.join(folder, 'train_fine.txt')
        img_paths, mask_paths = get_path_pairs(folder, split_f)
    elif split == 'val':
        split_f = os.path.join(folder, 'val_fine.txt')
        img_paths, mask_paths = get_path_pairs(folder, split_f)
    elif split == 'test':
        split_f = os.path.join(folder, 'test.txt')
        img_paths, mask_paths = get_path_pairs(folder, split_f)
    else:
        split_f = os.path.join(folder, 'trainval_fine.txt')
        img_paths, mask_paths = get_path_pairs(folder, split_f)

    return img_paths, mask_paths