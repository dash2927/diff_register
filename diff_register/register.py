import numpy as np

import skimage.io as sio
from skimage.measure import ransac
from skimage.feature import match_descriptors, ORB
from skimage import transform as tf
from skimage.filters import gaussian

from dipy.viz.regtools import overlay_images
from dipy.align.imaffine import (MutualInformationMetric,
                                 AffineRegistration)

def downsample(image, ratio=10, imread = True, imsave=True, colorcol = 0):
    """
    Downsamples the specified image by a factor specified by ratio. (May just
    use tilelit function).

    Parameters
    ----------

    Returns
    -------

    Examples
    --------

    """
    if imread:
        massive = sio.imread(image)
    else:
        massive = image

    if colorcol == 0:
        shrunk = tf.resize(massive, (massive.shape[0], massive.shape[1] // ratio, massive.shape[2] // ratio))
    elif colorcol == 2:
        shrunk = tf.resize(massive, (massive.shape[0] // ratio, massive.shape[1] // ratio, massive.shape[2]))
    else:
        print('Specify correct column of color data.')
        
    if imsave:
        fname = '{}_s.tif'.format(image.split('.')[0])
        sio.imsave(fname, shrunk)
    
    return shrunk


def plot_together(image1, image2):
    """Plot a comparison of two images."""
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

    Uses RANSAC matched on ORB descriptors.

    Parameters
    ----------
    static : array
        The reference image.
    moving : array
        The moving image.
    gaussian_blur : int/float
        The degree of blurring to apply to the images before detecting and
        extracting ORB descriptors

    Returns
    -------
    img_warp : array
        The moving image warped towards the static image
    affine : array
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
    """Use mutual information to align two images.

    Parameters
    ----------
    static : array
        The reference image.
    moving : array
        The moving image.
    starting_affine : array
        A proposed initial transformation

    Returns
    -------
    img_warp : array
        The moving image warped towards the static image
    affine : array
        The affine transformation for this warping

    """
    metric = MutualInformationMetric()
    reggy = AffineRegistration(metric=metric)
    transform = AffineTransform2D()
    affine = reggy.optimize(static, moving, transform, None,
                            starting_affine=starting_affine)
    img_warp = affine.transform(moving)
    return img_warp, affine.affine


def align_DAPI_DAPI(static_DAPI, moving_DAPI):
    """
    Aligns fixed DAPI stain to fresh DAPI stain in preparation for full
    registration.

    Parameters
    ----------

    Returns
    -------

    Examples
    --------

    """

    return moved_DAPI, alignment


def apply_alignment(moving_image, alignment):
    """
    Applies alignment from previous align function to a secondary image.

    Parameters
    ----------

    Returns
    -------

    Examples
    --------

    """

def register_DAPI_DAPI(static_DAPI, moving_DAPI):
    """
    Registers fixed DAPI stain image to fresh DAPI stain image.

    Parameters
    ----------

    Returns
    -------

    Examples
    --------

    """

    return moved_DAPI, transform


def apply_transform(moving_image, transform):
    """
    Applies a transform from a previous DIPy transform to a secondary image.

    Parameters
    ----------

    Returns
    -------

    Examples
    --------

    """

    return moved_image
