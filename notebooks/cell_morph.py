import sys
import os

import numpy as np
import pandas as pd
import skimage.io as sio
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_fill_holes
from skan import csr, draw

from skimage.filters import sobel, prewitt, scharr, gaussian, median, rank
from skimage.morphology import square, opening, closing, skeletonize
from skimage.morphology import star, disk, dilation, white_tophat
from skimage.morphology import remove_small_objects, reconstruction
from skimage.feature import canny
from skimage.util import pad
from skimage.measure import regionprops, label

from diff_register import im_process as imp


def binarize_image(image):

    edges = canny(image/255., sigma=0.001)
    fullim = closing(edges, selem=star(1))
    filled_im = binary_fill_holes(fullim)
    clean_im2 = remove_small_objects(filled_im, min_size=350)

    # implementation of watershed
    padw = 150
    cleanpad = pad(clean_im2, ((padw, padw),), mode='constant')
    distance = ndi.distance_transform_edt(cleanpad)
    local_maxi = peak_local_max(distance, indices=False,
                                footprint=np.ones((90, 90)),
                                labels=cleanpad)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=cleanpad)
    labels = labels[padw:-padw, padw:-padw]

    label_edge = scharr(labels)
    label_edge = label_edge < 0.0000000000001
    split_cells = np.logical_and(label_edge, clean_im2)
    split_cells = remove_small_objects(split_cells, min_size=350)

    return split_cells


folder = sys.argv[1]
all_images = [x for x in os.listdir(path=folder) if '.tif' in x]

for image_file in all_images:
    prefix = image_file.split('tif')[0]
    # Read in image and prep
    image = sio.imread(filename)
    # Extract GFAP channel
    mglia = image[1, :, :]

    clean_im = binarize_image(image)
    skel = imp.labelandskel(clean_im)
    Dfeatures = imp.mglia_features(skel, umppx=1.24)
    Dfeatures.to_csv('{}.csv'.format(prefix))
    sio.imsave('{}_bi.tif'.format(prefix), clean_im)
