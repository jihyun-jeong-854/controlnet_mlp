import json # We need to use the JSON package to load the data, since the data is stored in JSON format
import cv2
import numpy as np

from torch.utils.data import Dataset

import os
from annotator.util import HWC3, resize_image, resize_image_square

ROOT = "C:/Users/kim/Desktop/controlnet1/"
# ROOT = "./"

class Fill50kDataset(Dataset):
    def __init__(self):
        self.data = []
        with open(os.path.join(ROOT, 'training/fill50k/prompt.json'), 'rt') as f:
            for line in f:
                # d = json.loads(line)
                # if d['source'].split('/')[1].split('.')[0].startswith('200'):
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread(os.path.join(ROOT, 'training/fill50k/', source_filename))
        target = cv2.imread(os.path.join(ROOT, 'training/fill50k/', target_filename))

        source = resize_image(source, 256)
        target = resize_image(target, 256)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)


class ValDataset(Dataset):
    def __init__(self, dataset_name):
        self.data = []
        if dataset_name == 'fill50k':
            names = ['fill50k']
        else:
            names = ['things', 'laion-art', 'CC3M']
        for ds in names:
            with open(os.path.join(ROOT, 'testing/prompt.json'), 'rt') as f:
                for line in f:
                    self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_path = item['source']
        target_path = item['target']
        prompt = item['prompt']
        ds_label = item['ds_label']


        source = cv2.imread(os.path.join(ROOT, 'testing/', source_path))
        target = cv2.imread(os.path.join(ROOT, 'testing/', target_path))

        source = resize_image(source, 256)
        target = resize_image(target, 256)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source, ds_label=ds_label)

