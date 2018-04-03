import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import skimage.io as sio
import matplotlib.pyplot as plt
import skimage
from skimage.morphology import square, opening, closing, skeletonize
from skimage.measure import regionprops, label
from skan import csr, draw


def fuzzy_contrast(image_file, figsize=(10, 10), show=False):
    """
    Increase the contrast of input image by using fuzzy logic.

    Parameters
    ----------

    Returns
    -------

    Examples
    --------

    """

    # Build fuzzy logic system.
    dark = ctrl.Antecedent(np.linspace(0, 1, 101), 'dark')
    darker = ctrl.Consequent(np.linspace(0, 1, 101), 'darker')

    w = 90
    dark['dark'] = 1 - fuzz.sigmf(dark.universe, 0.425, w)
    dark['gray'] = fuzz.trimf(dark.universe, [0.35, 0.5, 0.65])
    dark['bright'] = fuzz.sigmf(dark.universe, 0.575, w)

    slope = 3.7
    width = 0.04
    darker['darker'] = fuzz.gbellmf(darker.universe, width, slope, 0.1)
    darker['midgray'] = fuzz.gbellmf(darker.universe, width, slope, 0.5)
    darker['brighter'] = fuzz.gbellmf(darker.universe, width, slope, 0.9)

    rule1 = ctrl.Rule(dark['dark'], darker['darker'])
    rule2 = ctrl.Rule(dark['gray'], darker['midgray'])
    rule3 = ctrl.Rule(dark['bright'], darker['brighter'])

    Fctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    F = ctrl.ControlSystemSimulation(Fctrl)

    # Apply to image
    test_image = sio.imread(image_file)
    if image_file.split('.')[1] == 'tif':
        test_image = test_image[:, :, 1] / test_image[:, :, 1].max()
    else:
        test_image = test_image / test_image.max()
    F.input['dark'] = test_image
    F.compute()
    fuzzy_image = F.output['darker']

    if show:
        fig, ax = plt.subplots(figsize=figsize)
        rf_image = (255.0 / fuzzy_image.max() * (fuzzy_image - fuzzy_image.min())).astype(np.uint8)
        ax.imshow(rf_image, cmap='gray', vmin=0, vmax=255.0)
        ax.axis('off')

    output = "fuzzy_{}.png".format(image_file.split('.')[0])
    sio.imsave(output, rf_image)

    return rf_image


def binary_image(image_file, threshold=2, figsize=(10, 10), op_image=False, close=False, show=False):
    """
    Create binary image from input image with optional opening step.

    Parameters
    ----------

    Returns
    -------

    Examples
    --------

    """

    test_image = sio.imread(image_file)
    bi_image = test_image > threshold

    if open is True:
        op_image = opening(bi_image, square(3))
    else:
        op_image = bi_image

    if close is True:
        op_image = closing(op_image, square(3))

    if show:
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(op_image, cmap='gray')
        ax.axis('off')

    op_image = op_image.astype('uint8')*255
    output = "clean_{}.png".format(image_file.split('.')[0])
    sio.imsave(output, op_image)

    return op_image


def label_image(image_file, area_thresh=50, figsize=(10, 10), show=False):
    """
    Create label image and calculate region properties.

    Parameters
    ----------

    Returns
    -------

    Examples
    --------

    """

    test_image = sio.imread(image_file)
    labels = label(test_image)
    props = regionprops(labels)

    short_image = np.zeros(labels.shape)
    counter = 0
    skip = 0
    short_props = []
    for i in range(0, len(props)):
        area = props[i]['area']
        if area < area_thresh:
            skip = skip + 1
        else:
            short_props.append(props[i])
            test_coords = props[i]['coords'].tolist()
            for coord in test_coords:
                short_image[coord[0], coord[1]] = True
            counter = counter + 1

    if show:
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(short_image, cmap='gray')
        ax.axis('off')

    short_image = short_image.astype('uint8')*255
    output = "short_{}.png".format(image_file.split('.')[0])
    sio.imsave(output, short_image)

    return short_image, short_props


def skeleton_image(image_file, threshold=50, area_thresh=50, figsize=(10, 10), show=False):
    """
    Skeletonizes the image and returns properties of each skeleton.

    Parameters
    ----------

    Returns
    -------

    Examples
    --------

    """
    # Median filtered image.
    image0 = sio.imread(image_file)
    image0 = np.ceil(255* (image0[:, :, 1] / image0[:, :, 1].max())).astype(int)
    image0 = skimage.filters.median(image0)
    filt = 'filt_{}.png'.format(image_file.split('.')[0])
    sio.imsave(filt, image0)

    #threshold the image
    binary0 = binary_image(filt, threshold=threshold, close=True, show=False)
    clean = 'clean_{}'.format(filt)

    #label image
    short_image, props = label_image(clean, area_thresh=area_thresh, show=False)
    short = 'short_{}'.format(clean)
    short_image = short_image > 1
    # Skeletonize
    skeleton0 = skeletonize(short_image)

    branch_data = csr.summarise(skeleton0)
    branch_data_short = branch_data

    #Remove small branches
    mglia = branch_data['skeleton-id'].max()
    nbranches = []

    ncount = 0
    for i in range(1, mglia+1):
        bcount = branch_data[branch_data['skeleton-id']==i]['skeleton-id'].count()
        if bcount > 0:
            ids = branch_data.index[branch_data['skeleton-id']==i].tolist()
            nbranches.append(bcount)
            for j in range(0, len(ids)):
                branch_data_short.drop([ids[j]])

            ncount = ncount + 1
    if show:
        fig, ax = plt.subplots(figsize=(10, 10))
        draw.overlay_euclidean_skeleton_2d(image0, branch_data_short,
                                           skeleton_color_source='branch-type', axes=ax)
        plt.savefig('skel_{}'.format(short))

    return skeleton0, branch_data_short, nbranches


def mglia_features(image_file):
    """
    Calculates features from input microglia image.

    Parameters
    ----------

    Returns
    -------

    Examples
    --------

    """

    X
    Y
    perimeter
    total_area
    soma_area
    eccentricity
    inertia_tensor
    label
    max_intensity
    mean_intensity
    moments
    solidity
    total_processes
    avg_p_length
    main_process

    return mglia_features
