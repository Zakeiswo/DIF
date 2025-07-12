import os
import os.path
# import cv2
import numpy as np
import PIL
import torch
from torch.utils.data import Dataset
from PIL import Image

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

class MyData(Dataset):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    def __init__(self, root, transform=True):
        super(MyData, self).__init__()
        self.root = root
        self._transform = transform

        img_root = os.path.join(self.root, 'train_images')
        gt_root = os.path.join(self.root, 'train_masks')
        dp_root = os.path.join(self.root, 'train_depth_negation')

        file_imgnames = os.listdir(img_root)
        self.img_names = []
        self.gt_names = []
        self.dp_names = []
        self.names = []
        for i, name in enumerate(file_imgnames):
            if not name.endswith('.jpg'):
                continue
            self.img_names.append(
                os.path.join(img_root, name[:-4] + '.jpg')
            )
            self.gt_names.append(
                os.path.join(gt_root, name[:-4] + '.png')
            )
            self.dp_names.append(
                os.path.join(dp_root, name[:-4] + '.png')
            )
            self.names.append(name[:-4])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        # load image

        # img
        img_file = self.img_names[index]
        img = PIL.Image.open(img_file)
        # print(img_file)
        img = img.resize((256, 256))
        img = np.array(img, dtype=np.uint8)

        # gt
        gt_file = self.gt_names[index]
        gt = PIL.Image.open(gt_file)
        gt = gt.resize((256,256))
        gt = np.array(gt, dtype=np.int32)
        # gt[gt != 0] = 1
        gt[gt <= 255/2] = 0
        gt[gt > 255/2] = 1

        # dp
        dp_file = self.dp_names[index]
        dp = PIL.Image.open(dp_file)
        dp = dp.resize((256,256))
        dp = np.array(dp, dtype=np.uint8)

        if self._transform:
            img, dp, gt = self.transform(img, dp, gt, img_file)
            return img, dp, gt, img_file, dp_file, gt_file
        else:
            return img, dp, gt, img_file, dp_file, gt_file

    def transform(self, img, dp, gt,img_file):
        img = img.astype(np.float64) / 255
        try:
            img -= self.mean
            img /= self.std
        except ValueError:
            print(img.shape)
            print(self.mean.shape)
            print(img_file)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        dp = torch.from_numpy(dp).float()
        gt = torch.from_numpy(gt).float()

        return img, dp, gt


class MyTestData(Dataset):
    """
    load images for testing
    root: director/to/images/
            structure:
            - root
                - images
                    - images (images here)
                - masks (ground truth)
    """

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    def __init__(self, root, transform=True):
        super(MyTestData, self).__init__()
        self.root = root
        self._transform = transform

        img_root = os.path.join(self.root, 'images')
        gt_root = os.path.join(self.root, 'gts')
        dp_root = os.path.join(self.root,'depths')

        # img_root = os.path.join(self.root, 'train_images')
        # gt_root = os.path.join(self.root, 'train_masks')
        # dp_root = os.path.join(self.root, 'train_depth_negation')

        file_names = os.listdir(img_root)
        self.img_names = []
        self.gt_names = []
        self.dp_names =[]
        self.names = []
        for i, name in enumerate(file_names):
            if not name.endswith('.jpg'):
                continue
            self.img_names.append(
                os.path.join(img_root, name[:-4] + '.jpg')
            )
            self.gt_names.append(
                os.path.join(gt_root,name[:-4] + '.png')
            )
            self.dp_names.append(
                os.path.join(dp_root, name[:-4] + '.png')
            )
            self.names.append(name[:-4])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        # load image
        img_file = self.img_names[index]
        img_index = img_file[-8:-4]
        img = PIL.Image.open(img_file)
        img_size = img.size
        img = img.resize((256, 256))
        img = np.array(img, dtype=np.uint8)

        gt_file = self.gt_names[index]
        gt = PIL.Image.open(gt_file)
        gt = gt.resize((256, 256))
        gt = np.array(gt, dtype=np.int32)
        # gt[gt != 0] = 1
        gt[gt <= 255/2] = 0
        gt[gt > 255/2] = 1

        depth_file = self.dp_names[index]
        dp = PIL.Image.open(depth_file)
        dp = dp.resize((256,256))
        dp = np.array(dp, dtype=np.int32)


        if self._transform:
            img,dp,gt = self.transform(img,dp,gt)
            return img,dp,gt,img_index
        else:
            return img,dp,gt,img_index

    def transform(self, img,dp,gt):
        img = img.astype(np.float64) / 255
        img -= self.mean
        img /= self.std
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

        gt = torch.from_numpy(gt).float()

        dp = torch.from_numpy(dp).float()
        return img,dp,gt