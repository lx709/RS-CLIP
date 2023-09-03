import sys
import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms


def read_txt(path, sep=' '):
    ims, labels = [], []
    with open(path, 'r') as f:
        for line in f.readlines():
            im, label = line.strip().split(sep)[:2]
            ims.append(im)
            labels.append(int(label))
    return ims, labels

# novel_class = [17, 14, 2, 3, 5, 0]
# novel_class = ['beach', 'golfcourse, 'mobilehomepark', 'river', 'sparseresidential', tenniscourt']

class RSDataset(Dataset):
    def __init__(self, data_root='./UCMerced_LandUse/', txt_path='train.txt', transform=None, test=False, novel_class=None):
        self.data_root = data_root
        self.ims, self.labels = read_txt(os.path.join(data_root, txt_path))
        label_list = read_txt(os.path.join(data_root, 'label_list.txt'), sep=',')
        self.transform = transform
        self.test = test
        
        if novel_class is not None:
            cls2id = dict(zip(label_list[0], label_list[1]))
            novel_class_ids = [cls2id[cls] for cls in novel_class]
            sel_idx = np.where([c in novel_class_ids for c in self.labels])[0]
            self.ims = [self.ims[i] for i in sel_idx]
            self.labels = [novel_class_ids.index(self.labels[i]) for i in sel_idx]
            self.classes = novel_class
        else:
            self.classes = label_list[0]

    def __getitem__(self, index):
        im_path0 = self.ims[index]
        label = self.labels[index]
        im_path = os.path.join(self.data_root, 'Images', im_path0)
        im = Image.open(im_path)
        if self.transform is not None:
            im = self.transform(im)

        return im, label, im_path0

    def __len__(self):
        return len(self.ims)

if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor()])
    dst_train = RSDataset(data_root='./UCMerced_LandUse/', txt_path='train.txt', transform=transform)
    dataloader_train = DataLoader(dst_train, shuffle=True, batch_size=1, num_workers=0)
    print('classes', dst_train.classes)
    for data in dataloader_train:
        print(data[0].shape, data[1])
        #print loc, cls
