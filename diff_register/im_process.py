import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import skimage.io as sio
import matplotlib.pyplot as plt
from skimage.morphology import square, opening, closing
from skimage.measure import regionprops, label


def fuzzy_contrast(image_file, figsize=(10, 10)):
    """
    Increase the contrast of input image by using fuzzy logic.
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

    fig, ax = plt.subplots(figsize=figsize)
    rf_image = (255.0 / fuzzy_image.max() * (fuzzy_image - fuzzy_image.min())).astype(np.uint8)
    ax.imshow(rf_image, cmap='gray', vmin=0, vmax=255.0)
    ax.axis('off')

    output = "fuzzy_{}.png".format(image_file.split('.')[0])
    sio.imsave(output, rf_image)

    return rf_image


def binary_image(image_file, threshold=2, figsize=(10, 10), op_image=False, close=False):
    """
    Create binary image from input image with optional opening step.
    """

    test_image = sio.imread(image_file)
    bi_image = test_image > threshold

    if open is True:
        op_image = opening(bi_image, square(3))
    else:
        op_image = bi_image

    if close is True:
        op_image = closing(op_image, square(3))

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(op_image, cmap='gray')
    ax.axis('off')

    op_image = op_image.astype('uint8')*255
    output = "clean_{}.png".format(image_file.split('.')[0])
    sio.imsave(output, op_image)

    return op_image


def label_image(image_file, area_thresh=50, figsize=(10, 10)):
    """
    Create label image and calculate region properties.
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

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(short_image, cmap='gray')
    ax.axis('off')

    short_image = op_image.astype('uint8')*255
    output = "short_{}.png".format(image_file.split('.')[0])
    sio.imsave(output, short_image)

    return short_image, short_props
