"""

UCF24 Dataset Classes
Author: Gurkirt Singh

Updated by Gurkirt Singh for ucf-24 , MSCOCO, VOC datasets

FOV VOC:
Original author: Francisco Massa for VOC dataset
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py
Updated by: Ellis Brown, Max deGroot for VOC dataset

Updated by: Gurkirt Singh to accpt text annotations for voc

Target is in xmin, ymin, xmax, ymax, label
coordinates are in range of [0, 1] normlised height and width


"""

import json
import torch
import pdb, time
import torch.utils.data as data
import pickle

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
            img_list.append([annots[img_id]['set'], img_id, np.asarray(boxes).astype(np.float32), np.asarray(labels).astype(np.int64)])
            ni += 1

    print_str = '\n\n*Num of images {:d} num of boxes {:d} avergae {:01f}\n\n'.format(ni, int(nb), nb/ni)

    return cls_list, img_list, print_str, idlist


class LoadImage(object):

    def __init__(self, space='BGR'):
        self.space = space

    def __call__(self, path_img):

        return cv2.imread(path_img)


class Detection(data.Dataset):
    """UCF24 Action Detection dataset class for pytorch dataloader
    """

    def __init__(self, args, train=True, image_sets=['train2017'], transform=None, anno_transform=None, full_test=False):

        self.dataset = args.dataset
        self.train = train
        self.root = args.data_root + args.dataset + '/'
        self.image_sets = image_sets
        self.transform = transform
        self.anno_transform = anno_transform
        self.ids = list()
        self.image_loader = LoadImage()
        self.classes, self.ids, self.print_str, self.idlist = make_object_lists(self.root, image_sets)
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        annot_info = self.ids[index]
        subset_str = annot_info[0]
        img_id = annot_info[1]
        boxes = annot_info[2] # boxes should be in x1 y1 x2 y2 format
        labels  =  annot_info[3]

        img_name = '{:s}{:s}.jpg'.format(self.root, img_id)
        # print(img_name)
        
        # t0 = time.perf_counter()
        img = Image.open(img_name).convert('RGB')
        # pdb.set_trace()
        width, height = img.size
        wh = [width, height]

        # print(img.size)
        if self.train and np.random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            w = boxes[:, 2] - boxes[:, 0]
            boxes[:, 0] = 1 - boxes[:, 2] # boxes should be in x1 y1 x2 y2 [0,1] format 
            boxes[:, 2] = boxes[:, 0] + w # boxes should be in x1 y1 x2 y2 [0,1] format
            
            # tp = 1.0 - boxes[:, 2].copy()
            # print(boxes)
            # print(boxes)
            # img.show()
            # draw = ImageDraw.Draw(img)
            # bb = boxes[0,:].copy()
            # bb[0] *= width
            # bb[1] *= height
            # bb[2] *= width
            # bb[3] *= height
            # draw.rectangle(bb)
            # img.show()
            # pdb.set_trace()
        
        img = self.transform(img)
        targets = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        # print(targets)
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
    images = torch.stack(images, 0)

    # images, ground_truths, _ , _, num_mt, img_indexs
    return images, targets, image_ids, whs

