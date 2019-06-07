
"""

    Author: Gurkirt Singh
    Purpose: Check number of anchor boxes
    Please don't remove above credits and 
    give star to this repo if it has been useful to you
    Licensed under The MIT License [see LICENSE for details]
    

"""

import math, pdb
import numpy as np


def my_anchors():
    anchor_areas = [75, 38, 19, 10, 5]  # p3 -> p7
    aspect_ratios = [1/2., 1/1., 2/1.]
    scale_ratios = [1., 1.5]
    anchor_wh = []

    for sa in anchor_areas:
        s = 3.05*600/sa
        s *= s
        for ar in aspect_ratios:  # w/h = ar
            h = math.sqrt(s/ar)
            w = ar * h
            for sr in scale_ratios:  # scale
                anchor_h = h*sr
                anchor_w = w*sr
                anchor_wh.append([anchor_w, anchor_h])
    num_fms = len(anchor_areas)
    print(np.asarray(anchor_wh))



sizes   = [32,64,128,256,512]
grid_size = [75, 38, 19, 10, 5]
strides = [8,]
ratios  = np.array([0.5, 1, 2])
scales  = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])


def generate_anchors(base_size=16, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """

    if ratios is None:
        ratios = AnchorParameters.default.ratios

    if scales is None:
        scales = AnchorParameters.default.scales

    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors

def new_anchors():
    anchor_wh = []
    for s1 in sizes:
        size_anchors = []
        for ar in ratios:  # w/h = ar
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
    # return torch.Tensor(anchor_wh).view(num_fms, -1, 2)

if __name__ == '__main__':
    generate_anchors()
    # my_anchors()
    new_anchors()



#return torch.Tensor(anchor_wh).view(num_fms, -1, 2)