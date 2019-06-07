import torch
from math import sqrt as sqrt
from itertools import product as product
import numpy as np



class anchorBox(object):
    """Compute anchorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, sizes = [32*32, 64*64, 128*128, 256*256, 512*512],
                        ratios =[0.5, 1 / 1., 1.5],
                        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])):

        super(anchorBox, self).__init__()
        self.sizes = sizes
        self.ratios = ratios
        self.scales = scales
        self.ar = len(self.ratios)*len(self.ratios)
        self.cell_anchors = self._get_whs()

    def _get_whs(self):
        anchors = []
        for s1 in self.sizes:
            anchors.append(self._gen_generate_anchors_on_one_level(s1))

        return anchors
    
    # modified from https://github.com/fizyr/keras-retinanet/blob/master/keras_retinanet/utils/anchors.py
    # Copyright 2017-2018 Fizyr (https://fizyr.com)
    def _gen_generate_anchors_on_one_level(self, base_size=32):
        """
        Generate anchor (reference) windows by enumerating aspect ratios X
        scales w.r.t. a reference window.
        
        """

        num_anchors = len(self.ratios) * len(self.scales)

        # initialize output anchors
        anchors = np.zeros((num_anchors, 4))

        # scale base_size
        anchors[:, 2:] = base_size * np.tile(self.scales, (2, len(self.ratios))).T

        # compute areas of anchors
        areas = anchors[:, 2] * anchors[:, 3]

        # correct for ratios
        # print(areas)
        pdb.set_trace()

        anchors[:, 2] = np.sqrt(areas / np.repeat(self.ratios, len(self.scales)))
        anchors[:, 3] = anchors[:, 2] * np.repeat(self.ratios, len(self.scales))
            # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
        anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
        anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

        return anchors

    def forward(self, feature_maps):
        
        for k, fmap in enumerate(feature_maps):
            fsize = fmap.shape[-2:]
            for i, j in product(range(fsize[1]), range(fsize[0])):
                f_k = 1 / f
                cx = (j + 0.5) / fsize[1]
                cy = (i + 0.5) / fsize[0])
                whs = self.whs[k]
                for anchor_w, anchor_h in whs:
                    anchors.append([cx, cy, anchor_w, anchor_h])

        output = torch.FloatTensor(anchors).view(-1, 4)
        # output.clamp_(max=1, min=0)
        return output
