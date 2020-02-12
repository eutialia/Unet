import os
import random
import numpy as np
import random
from os.path import isdir, exists, abspath, join
from PIL import Image, ImageOps


class DataLoader():
    def __init__(self, root_dir='data', batch_size=2, test_percent=.1):
        self.batch_size = batch_size
        self.test_percent = test_percent

        self.root_dir = abspath(root_dir)
        self.data_dir = join(self.root_dir, 'scans')
        self.labels_dir = join(self.root_dir, 'labels')

        self.files = os.listdir(self.data_dir)

        self.data_files = [join(self.data_dir, f) for f in self.files]
        self.label_files = [join(self.labels_dir, f) for f in self.files]

        self.data_files = [Image.open(img) for img in self.data_files]
        self.label_files = [Image.open(label) for label in self.label_files]

        self.data_files_flip = [self.flip(img) for img in self.data_files]
        self.label_files_flip = [self.flip(label) for label in self.label_files]

        self.data_files_rot = [self.rotate(img) for img in self.data_files]
        self.label_files_rot = [self.rotate(label) for label in self.label_files]

        self.data_files_enh = [self.enhance(img) for img in self.data_files]
        self.label_files_enh = self.label_files

        self.data_files += self.data_files_enh + self.data_files_flip + self.data_files_rot
        self.label_files += self.label_files_enh + self.label_files_flip + self.label_files_rot
        random.shuffle(self.data_files)
        random.shuffle(self.label_files)

    def __iter__(self):
        n_train = self.n_train()

        if self.mode == 'train':
            current = 0
            endId = n_train
        elif self.mode == 'test':
            current = n_train
            endId = len(self.data_files)

        while current < endId:
            current += 1
            data_image = self.data_prep(self.data_files[current-1])
            label_image = self.data_prep(self.label_files[current-1])
            yield (data_image, label_image)

    def setMode(self, mode):
        self.mode = mode

    def n_train(self):
        data_length = len(self.data_files)
        return np.int_(data_length - np.floor(data_length * self.test_percent))

    def data_prep(self, input):
        image = input.resize((572, 572))
        image = np.array(image, dtype='float32')
        image /= 255.
        return image

    def label_prep(self, input):
        image = input.resize((572, 572))
        image = np.array(image, dtype='float32')
        return image

    def flip(self, input):
        vertical = ImageOps.flip(input)
        horizontal = ImageOps.mirror(input)
        return random.choice([vertical, horizontal])

    def rotate(self, input):
        angles = [90, 45, 135]
        return input.rotate(angle = random.choice(angles))

    def enhance(self, input):
        return ImageOps.autocontrast(input, cutoff=0.3)