import os
import pdb
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import moco.loader
import moco.builder
import numpy as np
import torch
import pickle
import random
import math

import torch.nn.functional as F


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

class OpenImages(Dataset):
    """
    # Description:
        Dataset for retrieving FGVC Aircraft images and labels

    # Member Functions:
        __init__(self, phase, resize):  initializes a dataset
            phase:                      a string in ['train', 'val', 'test']
            resize:                     output shape/size of an image

        __getitem__(self, item):        returns an image
            item:                       the idex of image in the whole dataset

        __len__(self):                  returns the length of dataset
    """

    def __init__(self, phase='train', resize=500,full_dataset=True,rescale_crops_before=False,DATAPATH='',rescale_parameter=0,radius=0,args=''):
        # assert phase in ['train', 'val', 'test']
        self.phase = phase
        self.resize = resize
        self.radius=radius

        # self.DATAPATH = DATAPATH

        self.DATAPATH = args.DATAPATH

        self.images = []
        self.labels = []
        self.labels_to_idx = {}
        self.images_test = []
        self.labels_test = []
        self.bbox = []
        self.all_bbox = []
        self.full_dataset = full_dataset
        self.rescale_crops_before = rescale_crops_before
        self.rescale_parameter = rescale_parameter
        self.images_selected_with_all_features = np.load('images_selected_with_all_features.npy',  allow_pickle='TRUE').item()
        images_selected = np.load('images_selected_new.npy', allow_pickle='TRUE').item()

        images_selected_test = np.load('images_selected_test_40k.npy', allow_pickle='TRUE').item()
        cnt = 0
        for all_images in images_selected.keys():
            cnt_small = 0
            self.images.append(all_images + '.jpg')
            self.labels.append(images_selected[all_images])

        print(len(self.images))
        for images in images_selected_test.keys():
            self.images_test.append(images+'.jpg')
            self.labels_test.append(images_selected_test[images])

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        self.transform_moco = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])
    def is_safe(self,w1,h1,w2,h2,width_crop,height_crop):
        # print(w1,h1,w2,h2,width_crop,height_crop)
        if w1>0 and w1<width_crop and w2>0 and w2<width_crop and h1>0 and h1<height_crop and h2>0 and h2<height_crop:
            return True
        else:
            return False

    def get_params(self, img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        height, width = img.size
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(20):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h+i, w+j

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h+i, w+j
    def merge_bbox(self, bboxs):
        return [np.min(bboxs[:, 0]), np.min(bboxs[:, 1]), np.max(bboxs[:, 2]), np.max(bboxs[:, 3])]

    def __getitem__(self, item):
        # image
        img = Image.open(os.path.join(self.DATAPATH, self.images[item])).convert('RGB')  # (C, H, W)

        all_gt_bboxs, n_box = self.get_all_gt_bbox(img, item)

        while n_box < 2:
            item = torch.randint(0, len(self.images), (1,)).item()
            img = Image.open(os.path.join(self.DATAPATH, self.images[item])).convert('RGB')  # (C, H, W)
            all_gt_bboxs, n_box = self.get_all_gt_bbox(img, item)

        parent_box_id = torch.randint(low=0, high=n_box, size=[1]).item()
        parent_bbox = all_gt_bboxs[parent_box_id]
        parent_img = img.crop(parent_bbox)

        if (parent_bbox[2]-parent_bbox[0])*(parent_bbox[3]-parent_bbox[1]) < 256*256:
            base_img = self.jitter_and_resize(img, parent_bbox)
        else:
            base_img = parent_img.copy()

        child_box_num = torch.randint(low=2, high=max(min(n_box, 6), 3), size=[1]).item()
        child_box_ids = torch.randperm(n_box)[:child_box_num].numpy()
        child_box_ids = child_box_ids if parent_box_id in child_box_ids else np.append(child_box_ids, parent_box_id)
        child_box_merged = self.merge_bbox(all_gt_bboxs[child_box_ids])
        child_img = img.crop(child_box_merged)

        # print(n_box, child_box_num)
        # print(all_gt_bboxs[parent_box_id], child_box_merged)
        return self.transform_moco(base_img),self.transform_moco(base_img), self.transform(parent_img),self.transform(child_img)

    def get_all_gt_images(self, img, item):
        # list of all the boxes
        width, height = img.size
        all_gt_images = []
        all_images = self.images[item].split('.')[0]
        all_box =  self.images_selected_with_all_features[all_images]['data']
        for images in all_box:

            width1 = int(float(images[4]) * width)
            width2 = int(float(images[5]) * width)
            height1 = int(float(images[6]) * height)
            height2 = int(float(images[7]) * height)
            crop_box = (width1, height1, width2, height2)
            image_crop = img.crop((crop_box))
            all_gt_images.append(image_crop)
        return all_gt_images

    def jitter_and_resize(self, img, bbox):
        # list of all the boxes
        width, height = img.size
        width_window = max(0, 256 - (bbox[2]-bbox[0]))
        height_window = max(0, 256 - (bbox[3]-bbox[1]))
        width_min, width_max = max(0, bbox[0]-width_window), min(width, bbox[2]+width_window)
        height_min, height_max = max(0, bbox[1]-height_window), min(height, bbox[3]+width_window)
        # print(width_window, width_min, bbox[0], bbox[2], width_max)
        # print(height_window, height_min, bbox[1], bbox[3], height_max)
        width_start, width_end = torch.randint(width_min, bbox[0]+1, size=[1]).item(), torch.randint(bbox[2], width_max+1, size=[1]).item()
        height_start, height_end = torch.randint(height_min, bbox[1]+1, size=[1]).item(), torch.randint(bbox[3], height_max+1, size=[1]).item()
        # print(width_min, width_start, bbox[0], bbox[2], width_end, width_max)
        # print(height_min, height_start, bbox[1], bbox[3], height_end, height_max)
        crop_box = (width_start, height_start, width_end, height_end)
        image_crop = img.crop((crop_box))
        return image_crop

    def get_all_gt_bbox(self, img, item):
        # list of all the boxes
        width, height = img.size
        all_gt_bboxs = []
        all_images = self.images[item].split('.')[0]
        all_box =  self.images_selected_with_all_features[all_images]['data']
        for images in all_box:
            width1 = int(float(images[4]) * width)
            width2 = int(float(images[5]) * width)
            height1 = int(float(images[6]) * height)
            height2 = int(float(images[7]) * height)
            if (height2-height1)*(width2-width1) < 56*56:
                continue
            crop_box = (width1, height1, width2, height2)
            all_gt_bboxs.append(crop_box)
        return np.array(all_gt_bboxs), len(all_gt_bboxs)

    def __len__(self):
        return len(self.images)



