import torch.utils.data as data
from road import overlay_people_on_road
import numpy as np
from PIL import Image


class AugmentationDataset(data.Dataset):
    def __init__(self, people_file_path, trainset):
        self.people_file_path = people_file_path
        self.trainset = trainset

    def __getitem__(self, item):
        img, mask = self.trainset[item]
        img = overlay_people_on_road(self.people_file_path, np.array(img), np.array(mask))
        img = Image.fromarray(img, 'RGB')
        return img, mask

    def __len__(self):
        return len(self.trainset)

