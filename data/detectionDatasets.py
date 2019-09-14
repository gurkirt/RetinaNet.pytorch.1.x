"""

Copyright (c) 2019 Gurkirt Singh 
 All Rights Reserved.

"""

import json
import torch
import pdb, time
import torch.utils.data as data
import pickle
from .transforms import get_image_list_resized
import torch.nn.functional as F
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image, ImageDraw


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def make_object_lists(rootpath, subsets=['train2007']):
    with open(rootpath + 'annots.json', 'r') as f:
        db = json.load(f)

    img_list = []
    names = []
    cls_list = db['classes']
    annots = db['annotations']
    idlist = []
    if 'ids' in db.keys():
        idlist = db['ids']
    ni = 0
    nb = 0.0
    print_str = ''
    for img_id in annots.keys():
        # pdb.set_trace()
        if annots[img_id]['set'] in subsets:
            names.append(img_id)
            boxes = []
            labels = []
            for anno in annots[img_id]['annos']:
                nb += 1
                boxes.append(anno['bbox'])
                labels.append(anno['label'])
            # print(labels)
            img_list.append([annots[img_id]['set'], img_id, np.asarray(boxes).astype(np.float32), np.asarray(labels).astype(np.int64), annots[img_id]['wh']])
            ni += 1

    print_str = '\n\n*Num of images {:d} num of boxes {:d} avergae {:01f}\n\n'.format(ni, int(nb), nb/ni)

    return cls_list, img_list, print_str, idlist


class DetectionDataset(data.Dataset):
    """Detection Dataset class for pytorch dataloader"""

    def __init__(self, args, train=True, image_sets=['train2017'], transform=None, anno_transform=None, full_test=False):

        self.dataset = args.dataset
        self.train = train
        self.root = args.data_root + args.dataset + '/'
        self.image_sets = image_sets
        self.transform = transform
        self.anno_transform = anno_transform
        self.ids = list()
        self.classes, self.ids, self.print_str, self.idlist = make_object_lists(self.root, image_sets)
        self.max_targets = 20
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        annot_info = self.ids[index]
        subset_str = annot_info[0]
        img_id = annot_info[1]
        boxes = annot_info[2] # boxes should be in x1 y1 x2 y2 format
        labels  =  annot_info[3]
        wh = annot_info[4]

        img_name = '{:s}{:s}.jpg'.format(self.root, img_id)
        # print(img_name)
        # t0 = time.perf_counter()
        img = Image.open(img_name).convert('RGB')
        orig_w, orig_h = img.size
        # print(img.size)
        if self.train and np.random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            w = boxes[:, 2] - boxes[:, 0]
            boxes[:, 0] = 1 - boxes[:, 2] # boxes should be in x1 y1 x2 y2 [0,1] format 
            boxes[:, 2] = boxes[:, 0] + w # boxes should be in x1 y1 x2 y2 [0,1] format
        

        # print(img.size, wh)
        img = self.transform(img)
        _, height, width = img.shape
        # print(img.shape, wh)
        wh = [width, height, orig_w, orig_h]
        # print(wh)
        boxes[:, 0] *= width # width x1
        boxes[:, 2] *= width # width x2
        boxes[:, 1] *= height # height y1
        boxes[:, 3] *= height # height y2

        targets = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return img, targets, index, wh

def custum_collate(batch):
    targets = []
    images = []
    image_ids = []
    whs = []
    # fno = []
    # rgb_images, flow_images, aug_bxsl, prior_labels, prior_gt_locations, num_mt, index
    
    for sample in batch:
        images.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
        image_ids.append(sample[2])
        whs.append(sample[3])
    
    counts = []
    max_len = -1
    for target in targets:
        max_len = max(max_len, target.shape[0])
        counts.append(target.shape[0])
    new_targets = torch.zeros(len(targets), max_len, targets[0].shape[1])
    cc = 0
    for target in targets:
        new_targets[cc,:target.shape[0]] = target
        cc += 1
    images_ = get_image_list_resized(images)
    cts = torch.LongTensor(counts)
    # print(images_.shape)
    return images_, new_targets, cts, image_ids, whs

