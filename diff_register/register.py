"""Aligns microscopy videos and images together

Designed to align videos with large tilescan images. Requires coordinates at
which videos and tilescan images were collected. Designed based on Nikon
confocal/camera fluorescent microscope setup.

"""

import numpy as np
import xml.etree.ElementTree as et
import skimage.io as sio
from skimage.measure import ransac
from skimage.feature import match_descriptors, ORB
from skimage import transform as tf
from skimage.filters import gaussian
from dipy.viz.regtools import overlay_images
from dipy.align.imaffine import (MutualInformationMetric,
                                 AffineRegistration)


def read_xmlpoints(xmlfile, cell_image, converttopix=True, umppx=0.62,
                   offset=(17000, -1460)):
    """Read points saved from Nikon multipoints xml file.

    Parameters
    ----------
    xmlfile : string
        XML file containing locations at which trajectory videos were collected.
    cell_image : string
        TIFF file of tile scan image.
    converttopix : bool
        Specify whether coordinates should be converted from microns to pixels.
    umppx: float
        Microns per pixel. Pixel density of cell tilescan image.
    offset: list of int or float
        Coordinates (in microns) of the upper right corner of the input tilescan
        image.

    Returns
    -------
    subim : numpy.ndarray
        Tiled image extracted from cell_image at points specified by vidpoints
        and dimensions specified by dim.

    """
    tree = et.parse(xmlfile)
    root = tree.getroot()

    tilescan = cell_image
    tiled = sio.imread(tilescan)

    y = []
    x = []
    xmlpoints = []
    counter = 0

    for point in root[0]:
        if counter > 1:
            x = float(point[2].attrib['value'])
            y = float(point[3].attrib['value'])
            if converttopix:
                xmlpoints.append(((x-offset[0])/umppx, (y-offset[1])/umppx))
            else:
                xmlpoints.append((x, y))
        counter = counter + 1

    xmlmod = []
    for point in xmlpoints:
        xmlmod.append((tiled.shape[0]-point[0], point[1]))

    return xmlmod


def crop_to_videodims(cell_image, multichannel=False, vidpoint=(600, 600),
                      dim=512, save=True, fname='test.tif', correction=(0, 0),
                      dim_traj=2048):
    """Crops small images from large tilescan image

    Registration function used to crop sub-images from large tile-scanned images
    collected via confocal or fluorescent microsopy at specified points.

    Parameters
    ----------
    cell_image : numpy.ndarray
        Tiled large image to be cut.
    multichannel : bool
        Specify if image has multiple channels. Currently only works if set to
        False.
    vidpoint : list of integers or floats
        Point in image at which to perform crops. Each point is at the center of
        each image.
    dim : int
        Dimension of desired out put image e.g. 512 for a 512 by 512 pixel
        image.
    save : bool
        Specify whether user wants to save image to harddrive.
    fname : string
        Name of output file
    correction: list of int or float
        Optional xy correction coordinates if alignment doesn't appear to be
        perfect

    Returns
    -------
    subim : numpy.ndarray
        Tiled image extracted from cell_image at points specified by vidpoints
        and dimensions specified by dim.

    """

    ndim = dim

    lox = int(vidpoint[1] - ndim/2+correction[1])
    hix = int(vidpoint[1] + ndim/2+correction[1])
    loy = int(vidpoint[0] - ndim/2+correction[0])
    hiy = int(vidpoint[0] + ndim/2+correction[0])

    if not multichannel:
        subim = cell_image[lox:hix, loy: hiy]
        # subim = cell_image[int(vidpoint[1]-ndim/2):int(vidpoint[1]+ndim/2),
        # int(vidpoint[0]-ndim-20):int(vidpoint[0])-20]
        subim_new = tf.resize(subim, (dim_traj, dim_traj))
        if save:
            sio.imsave(fname, subim_new)

    return subim_new


def downsample(image, ratio=10, imread=True, imsave=True, colorcol=0):
    """Downsamples the specified image by a factor specified by ratio

    Parameters
    ----------
    image : numpy.ndarray or string
        Large image to be downsampled (or filename of said image)
    ratio : int or float
        Factor by which to downsample image
    imread : bool
        If True, reads image in as string. If false, reads in as numpy.ndarray.
    imsave : bool
        If True, saves resulting downsampled image.
    colorcol : int
        User specifies which dimension corresponds to color e.g. [c, x, y]
        would be 0 and [x, y, c] would be 2.

    Returns
    -------
    shrunk : numpy.ndarray
        Downsampled image

    Examples
    --------

    """
    if imread:
        massive = sio.imread(image)
    else:
        massive = image

    if colorcol == 0:
        shrunk = tf.resize(massive, (massive.shape[0],
                                     massive.shape[1] // ratio,
                                     massive.shape[2] // ratio))
    elif colorcol == 2:
        shrunk = tf.resize(massive, (massive.shape[0] // ratio,
                                     massive.shape[1] // ratio,
                                     massive.shape[2]))
    else:
        print('Specify correct column of color data.')

    if imsave:
        fname = '{}_s.tif'.format(image.split('.')[0])
        sio.imsave(fname, shrunk)

    return shrunk


def plot_together(image1, image2):
    """Plot a comparison of two images

    Parameters
    ----------
    image1 : numpy.ndarray
        First image to compare
    image2 : nump.ndarray
        Second image to compare

    Returns
    -------
    new_image1 : numpy.ndarray
        Modified version of image1 with same size as image2
    new_image2 : numpy.ndarray
        Modified version of image2 with same saze as image1

    """
    new_shape1 = list(image1.shape)
    new_shape2 = list(image2.shape)

    if image1.shape[0] < image2.shape[0]:
        new_shape1[0] = image2.shape[0]
    elif image1.shape[0] > image2.shape[0]:
        new_shape2[0] = image1.shape[0]

    if image1.shape[1] < image2.shape[1]:
        new_shape1[1] = image2.shape[1]
    elif image1.shape[1] > image2.shape[1]:
        new_shape2[1] = image1.shape[1]

    if new_shape1 != image1.shape:
        new_image1 = np.zeros(new_shape1, dtype=image1.dtype)
        new_image1[:image1.shape[0], :image1.shape[1]] = image1
        image1 = new_image1

    if new_shape2 != image2.shape:
        new_image2 = np.zeros(new_shape2, dtype=image2.dtype)
        new_image2[:image2.shape[0], :image2.shape[1]] = image2
        image2 = new_image2

    fig = overlay_images(image1, image2)
    fig.set_size_inches([12, 10])
    return new_image1, new_image2


def initial_alignment(static, moving, gaussian_blur=5):
    """Rough initial alignment of two images to each others.

    Uses RANSAC matched on ORB descriptors. I no longer user these custom
    alignment functions, as Nikon software has its own built-in alignment
    capabilities.

    Parameters
    ----------
    static : numpy.ndarray
        The reference image.
    moving : numpy.ndarray
        The moving image.
    gaussian_blur : int or float
        The degree of blurring to apply to the images before detecting and
        extracting ORB descriptors

    Returns
    -------
    img_warp : numpy.ndarray
        The moving image warped towards the static image
    affine : numpy.ndarray
        The affine transformation for this warping

    """

    descriptor_extractor = ORB()
    descriptor_extractor.detect_and_extract(gaussian(static, gaussian_blur))
    keypoints_left = descriptor_extractor.keypoints
    descriptors_left = descriptor_extractor.descriptors

    descriptor_extractor.detect_and_extract(gaussian(moving, gaussian_blur))
    keypoints_right = descriptor_extractor.keypoints
    descriptors_right = descriptor_extractor.descriptors

    matches = match_descriptors(descriptors_left, descriptors_right,
                                cross_check=True)

    model, inliers = ransac((keypoints_left[matches[:, 0]],
                             keypoints_right[matches[:, 1]]),
                            tf.AffineTransform,
                            min_samples=8,
                            residual_threshold=10,
                            max_trials=5000)

    img_warp = tf.warp(moving, model.inverse)

    return img_warp, model.params


def fine_alignment(static, moving, starting_affine=None):
    """Use Dipy to align two images.

    Parameters
    ----------
    static : numpy.ndarray
        The reference image.
    moving : numpy.ndarray
        The moving image.
    starting_affine : numpy.ndarray
        A proposed initial transformation

    Returns
    -------
    img_warp : numpy.ndarray
        The moving image warped towards the static image
    affine : numpy.ndarray
        The affine transformation for this warping

    """
    metric = MutualInformationMetric()
    reggy = AffineRegistration(metric=metric)
    transform = AffineTransform2D()
    affine = reggy.optimize(static, moving, transform, None,
                            starting_affine=starting_affine)
    img_warp = affine.transform(moving)
    return img_warp, affine.affine
