import torch
from math import sqrt as sqrt
from itertools import product as product
import numpy as np



class anchorBox(object):
    """Compute anchorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, feature_areas = [32*32, 64*64, 128*128, 256*256, 512*512],
                        aspect_ratios =[0.5, 1 / 1., 1.5],
                        scale_ratios = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]),
                        dataset='all'):

        super(anchorBox, self).__init__()
        # self.type = cfg.name
        self.image_size = input_dim
        self.feature_areas = feature_areas
        self.num_anchors = 0
        self.variance = [0.1, 0.2]
        self.feature_maps = feature_size
        self.aspect_ratios = aspect_ratios
        self.scale_ratios = scale_ratios
        self.default_scale= [2.4, 2.8, 3, 3.2, 3.4]

        # self.default_scale = 2.8
        
        self.anchor_boxes = len(self.aspect_ratios)*len(self.scale_ratios)
        self.ar = self.anchor_boxes

        self.whs = self._get_whs():

    def _get_whs(self):

    anchor_wh = []
    for s1 in self.sizes:
        size_anchors = []
        for ar in self.scale_ratios:  # w/h = ar
            # print('s1', s1)
            s = s1 * s1
            h = math.sqrt(s/ar)
            w = ar * h
            for sr in scales:  # scale
                anchor_h = h*sr
                anchor_w = w*sr
                size_anchors.append([anchor_w, anchor_h])
        anchor_wh.append(np.asarray(size_anchors))
    # num_fms = len(anchor_wh)
    print(np.asarray(anchor_wh))
    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = 1 / f
                # unit center x,y
                cx = (j + 0.5) * f_k
                cy = (i + 0.5) * f_k
                whs = self.whs[k]
                for anchor_w, anchor_h in whs:
                    anchors.append([cx, cy, anchor_w/self.image_size, anchor_h/self.image_size])

        output = torch.FloatTensor(anchors).view(-1, 4)
        # output.clamp_(max=1, min=0)
        return output
