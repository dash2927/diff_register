"""Image processing functions designed to extract cellular features from input
fluorescently stained images.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import skimage
import skimage.io as sio
from skimage.morphology import square, opening, closing, skeletonize
from skimage.measure import regionprops, label
from skan import csr, draw


class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


def fuzzy_contrast(folder, image_file, figsize=(10, 10),
                   channel=None, show=False):
    """Increase the contrast of input image by using fuzzy logic.

    Parameters
    ----------
    folder : string
        Directory containing image_file
    image_file : string
        Filename of image to be analyzed
    figsize : tuple of int or float
        Size of output image
    show : bool
        If True, outputs image to Jupyter notebook display.
    channel : int
        Channel of image to read in for multichannel images e.g.
        testim[:, :, channel]

    Returns
    -------
    rf_image : numpy.ndarray
        Output image

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
    fname = '{}/{}'.format(folder, image_file)
    test_image = sio.imread(fname)
    if channel is None:
        test_image = test_image / test_image.max()
    else:
        test_image = test_image[:, :, channel] / test_image[:, :, channel].max()
    F.input['dark'] = test_image
    F.compute()
    fuzzy_image = F.output['darker']
    rf_image = (255.0 / fuzzy_image.max() * (fuzzy_image - fuzzy_image.min())
                ).astype(np.uint8)

    if show:
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(rf_image, cmap='gray', vmin=0, vmax=255.0)
        ax.axis('off')

    output = "fuzzy_{}.png".format(image_file.split('.')[0])
    sio.imsave(folder+'/'+output, rf_image)

    return rf_image


def binary_image(folder, image_file, threshold=2, figsize=(10, 10),
                 ajar=False, close=False, show=False,
                 channel=None, imname=None):
    """Create binary image from input image with optional opening step.

    Parameters
    ----------
    folder : string
        Directory containing image_file
    image_file : string
        Filename of image to be analyzed.
    threshold : int or float
        Intensity threshold of binary image.
    figsize : tuple of int or float
        Size of output figure
    ajar : bool
        If True, opens binary image by performing a dilation followed by
        an erosion.
    close : bool
        If True, closes binary image by performing an erosion followed by a
        dilation.
    show : bool
        If True, outputs image to Jupyter notebook display
    channel : int
        Channel of image to read in for multichannel images e.g.
        testim[:, :, channel]
    imname : string
        Desired name of output file. Defaults to 'test.png'

    Returns
    -------
    op_image : numpy.ndarray
        Output image

    Examples
    --------

    """

    fname = '{}/{}'.format(folder, image_file)
    if channel is None:
        test_image = sio.imread(fname)
    else:
        test_image = sio.imread(fname)[:, :, channel]

    bi_image = test_image > threshold

    if ajar is True:
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

    if imname is None:
        output = "clean_{}".format(image_file)
    else:
        output = imname

    sio.imsave('{}/{}'.format(folder, output), op_image)

    return op_image


def clean_image(folder, image_file, threshold=2, figsize=(10, 10),
                ajar=False, close=False, show=False,
                area_thresh=50, channel=None, imname=None):
    """Create binary image from input image with optional opening step.

    Parameters
    ----------
    folder : string
        Directory containing image_file
    image_file : string
        Filename of image to be analyzed.
    threshold : int or float
        Intensity threshold of binary image.
    figsize : tuple of int or float
        Size of output figure
    ajar : bool
        If True, opens binary image by performing a dilation followed by
        an erosion.
    close : bool
        If True, closes binary image by performing an erosion followed by a
        dilation.
    show : bool
        If True, outputs image to Jupyter notebook display
    area_thresh : int or float
        Minimum square pixels for object to be included in final image
    channel : int
        Channel of image to read in for multichannel images e.g.
        testim[:, :, channel]
    imname : string
        Desired name of output file. Defaults to 'test.png'

    Returns
    -------
    short_image : numpy.ndarray
        Output binary image. All small objects (area < area_thresh) are
        filtered out
    short_props : skimage.object
        Contains all properties of objects identified in image

    Examples
    --------

    """

    fname = '{}/{}'.format(folder, image_file)
    if channel is None:
        test_image = sio.imread(fname)
    else:
        test_image = sio.imread(fname)[:, :, channel]
    bi_image = test_image > threshold

    if ajar is True:
        op_image = opening(bi_image, square(3))
    else:
        op_image = bi_image

    if close is True:
        op_image = closing(op_image, square(3))

    op_image = op_image.astype('uint8')*255

#     if default_name:
#         output = "clean_{}.png".format(image_file.split('.')[0])
#     else:
#         output = fname

#     sio.imsave(folder+'/'+output, op_image)

    # Labelling and cleaning up image.
    test_image = op_image
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

    if imname is None:
        output = "short_{}".format(image_file)
    else:
        output = imname

    sio.imsave(folder+'/'+output, short_image)

    return short_image, short_props


def label_image(folder, image_file, area_thresh=50, figsize=(10, 10),
                show=False, imname=None):
    """Filters out small objects from binary image and finds cell features.

    Similar to clean_image, but operates on imput binary image rather than the
    raw image file. Run binary_image on raw image file before feeding into
    label_image.

    Parameters
    ----------
    folder : string
        Directory containing image_file
    image_file : string
        Filename of image to be analyzed.
    figsize : tuple of int or float
        Size of output figure
    show : bool
        If True, outputs image to Jupyter notebook display
    area_thresh : int or float
        Minimum square pixels for object to be included in final image
    channel : int
        Channel of image to read in for multichannel images e.g.
        testim[:, :, channel]
    imname : string
        Desired name of output file. Defaults to 'test.png'

    Returns
    -------
    short_image : numpy.ndarray
        Output binary image. All small objects (area < area_thresh) are
        filtered out
    short_props : skimage.object
        Contains all properties of objects identified in image

    Examples
    --------

    """

    fname = '{}/{}'.format(folder, image_file)
    test_image = sio.imread(fname)
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

    if imname is None:
        output = "short_{}".format(image_file)
    else:
        output = imname

    sio.imsave(folder+'/'+output, short_image)

    return short_image, short_props


def skeleton_image(folder, image_file, threshold=50, area_thresh=50,
                   tofilt=True, ajar=True, close=True, figsize=(10, 10),
                   show=False, channel=0, disp_binary=True, imname=None):
    """Skeletonizes the image and returns properties of each skeleton.

    Composite function of binary_image, clean_image and skeletonizing
    functionality from skan.

    Parameters
    ----------
    folder : string
        Directory containing image_file
    image_file : string
        Filename of image to be analyzed
    threshold : int or float
        Intensity threshold level for threshold.
    area_thresh : int or float
        Size cutoff level to remove small objects
    figsize : tuple of int or float
        Size out output figure
    show : bool
        If True, prints image to Jupyter notebook
    channel : int
        If multichannel is True, reads in image corresponding to this channel in
        file.
    disp_binary: bool
        If True, prints binary image instead of raw image
    imname : string
        Output filename

    Returns
    -------
    skeleton0 : numpy.ndarray
        Skeletonized version of input image_file
    branch_data_short : pandas.core.frame.DataFrame
        Data associated with each branch found in input image
    nbranches : list
        Number of branches on each cell in branch_data_short
    short_image : numpy.ndarray
        Cleaned up binary image from image_file prior to skeletonization
    props : skimage.object
        Contains all properties of objects identified in image

    Examples
    --------

    """

    # Median filtered image.
    fname = '{}/{}'.format(folder, image_file)
    image0 = sio.imread(fname)
    if channel is None:
        image0 = np.ceil(255 * (image0[:, :] / image0[:, :].max())).astype(int)
    else:
        image0 = np.ceil(255 * (image0[:, :, channel
                                       ] / image0[:, :, channel
                                                  ].max())).astype(int)

    if tofilt:
        image0 = skimage.filters.median(image0)
        image_file = 'filt_{}'.format(image_file)
        sio.imsave(folder+'/'+image_file, image0)

    # label image
    short_image, props = clean_image(folder, image_file, threshold=threshold,
                                     area_thresh=area_thresh, ajar=ajar,
                                     close=close, imname='labelim.tif',
                                     channel=None, show=False)
    short_image = short_image > 1
    # Skeletonize
    skeleton0 = skeletonize(short_image)

    branch_data = csr.summarise(skeleton0)
    branch_data_short = branch_data

    # Remove small branches
    mglia = branch_data['skeleton-id'].max()
    nbranches = []

    ncount = 0
    for i in range(1, mglia+1):
        bcount = branch_data[branch_data['skeleton-id'
                                         ] == i]['skeleton-id'].count()
        if bcount > 0:
            ids = branch_data.index[branch_data['skeleton-id'] == i].tolist()
            nbranches.append(bcount)
            for j in range(0, len(ids)):
                branch_data_short.drop([ids[j]])

            ncount = ncount + 1
    if show:
        fig, ax = plt.subplots(figsize=figsize)
        if disp_binary:
            draw.overlay_euclidean_skeleton_2d(short_image, branch_data_short,
                                               skeleton_color_source='branch-type',
                                               axes=ax)
        else:
            draw.overlay_euclidean_skeleton_2d(image0, branch_data_short,
                                               skeleton_color_source='branch-type',
                                               axes=ax)

    if imname is None:
        output = "skel_{}".format(image_file)
    else:
        output = imname
    plt.savefig('{}/{}'.format(folder, output))

    skel = Bunch(im=skeleton0, branchdat=branch_data_short, nbran=nbranches,
                 shortim=short_image, props=props)

    return skel


def mglia_features(skeleton, umppx=1):
    """Assembles feature dataset from pre-processed cellular images

    Parameters
    ----------
    props : skimage.object
        Contains raw properties from input cell image. Output from
        im_process.skeleton_image
    branch_data_short : blank
        Data from skeletonized cells. Output from im_process.skeleton_image
    convert : bool
        If True, converts from pixels to microns using umppx
    umppx : int or float
        Conversion ratio from pixels to microns

    Returns
    -------
    features : pandas.core.frames.DataFrame
        Pandas dataframe of cellular features

    Notes
    -----
    Raises error if skeleton is missing from any cell in the binary image.

    """
    #print('Successfully started')
    props = skeleton.props
    branch_data_short = skeleton.branchdat
    X = np.zeros((len(props)))
    Y = np.zeros((len(props)))
    perimeter = np.zeros((len(props)))
    areas = np.zeros((len(props)))
    eccentricity = np.zeros((len(props)))
    inertia_tensor = [0]*len(props)
    label = [0]*len(props)
    max_intensity = np.zeros((len(props)))
    mean_intensity = np.zeros((len(props)))
    moments = [0]*len(props)
    solidity = np.zeros((len(props)))
    # total_processes = np.zeros((len(props)))
    # avg_p_length = np.zeros((len(props)))
    # main_process = np.zeros((len(props)))

    # properties that can be found from sklearn.measure.regionprops
    counter = 0
    for item in props:
        X[counter] = item.centroid[0]
        Y[counter] = item.centroid[1]
        perimeter[counter] = item.perimeter
        areas[counter] = item.area
        eccentricity[counter] = item.eccentricity
        inertia_tensor[counter] = item.inertia_tensor
        label[counter] = item.label
        # max_intensity[counter] = item.max_intensity
        # mean_intensity[counter] = item.mean_intensity
        moments[counter] = item.moments
        solidity[counter] = item.solidity
        counter = counter + 1

    # properties associated with processes as found from skan
    mglia = branch_data_short['skeleton-id'].max()
    nbranches = []
    avg_p_length = []
    main_process = []

    xs = []
    ys = []

    ncount = 0
    for i in branch_data_short['skeleton-id'].unique():
        bcount = branch_data_short[branch_data_short[
                                   'skeleton-id'] == i]['skeleton-id'].count()
        bavg = np.mean(branch_data_short[
                       branch_data_short['skeleton-id'] == i][
                       'branch-distance'])
        blong = np.max(branch_data_short[
                       branch_data_short['skeleton-id'] == i][
                       'branch-distance'])
        xcoord = np.mean([np.mean(
                         branch_data_short[branch_data_short[
                          'skeleton-id'] == i]['img-coord-0-0']),
                         np.mean(branch_data_short[
                          branch_data_short['skeleton-id'] == i][
                          'img-coord-1-0'])])
        xs.append(xcoord)
        ycoord = np.mean([np.mean(branch_data_short[
                         branch_data_short['skeleton-id'] == i][
                         'img-coord-0-1']),
                         np.mean(branch_data_short[
                          branch_data_short['skeleton-id'] == i][
                          'img-coord-1-1'])])
        ys.append(ycoord)
        nbranches.append(bcount)
        avg_p_length.append(bavg)
        main_process.append(blong)

    nbranches_ord = [0]*len(X)
    avg_p_length_ord = [0]*len(X)
    main_process_ord = [0]*len(X)

    for i in range(0, len(xs)):
        #print(i, xs[i], ys[i])
        skel_id = i
        min_function = np.square(xs[i] - X)+np.square(ys[i] - Y)
        try:
            mglia_id = np.argmin(min_function)
            nbranches_ord[mglia_id] = nbranches[skel_id]
        except IndexError:
            mglia_ids = np.argsort(min_function)
            for idd in mglia_ids:
                if idd < len(nbranches):
                    mglia_id = idd
                    break
            nbranches_ord[mglia_id] = nbranches[skel_id
                                                ] + nbranches_ord[mglia_id]
        avg_p_length_ord[mglia_id] = avg_p_length[skel_id
                                                  ] + avg_p_length_ord[mglia_id]
        main_process_ord[mglia_id] = main_process[skel_id
                                                  ] + main_process_ord[mglia_id]
        # print(mglia_id)
        # print(np.min(min_function))

    factor = umppx
    features = pd.DataFrame({'X': X*factor,
                             'Y': Y*factor,
                             'perimeter': perimeter*factor,
                             'area': areas*factor*factor,
                             'eccentricity': eccentricity,
                             'inertia_tensor': inertia_tensor,
                             'label': label,
                             # 'max intensity' : max_intensity,
                             # 'mean intensity' : mean_intensity,
                             'moments': moments,
                             'solidity': solidity,
                             'total_branches': nbranches_ord,
                             'average_branch': [x*factor for x in avg_p_length_ord],
                             'main_branch': [x*factor for x in main_process_ord]
                             })

    return features


def features_hist(features, feature, bin_size=100, bin_range=5000):
    """Plots a histogram of a desired cell feature

    Parameters
    ----------
    features : pandas.core.frames.DataFrame
        Features output from mglia_features
    feature : string
        Column name contained in features
    bin_size : int or float
        Changes resolution of histogram bars
    bin_range : int or float
        Upper limit to plot of feature data

    """

    xlabel = "Microglia {} (pixels)".format(feature)
    ylabel = "Count"

    nbins = bin_range/bin_size + 1
    test_bins = np.linspace(0, bin_range, nbins)
    dist = features[feature]

    histogram, test_bins = np.histogram(dist, bins=test_bins)

    # Plot_general_histogram_code
    avg = np.mean(dist)
    fig = plt.figure(figsize=(16, 6))

    plt.rc('axes', linewidth=2)
    plot = histogram
    bins = test_bins
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:])/2
    bar = plt.bar(center, plot, align='center', width=width)
    plt.axvline(avg)
    plt.xlabel(xlabel, fontsize=30)
    plt.ylabel(ylabel, fontsize=30)
    plt.tick_params(axis='both', which='major', labelsize=20)

    plt.show()

#
# def read_xmlpoints(xmlfile, umppx=0.62, offset=(17000, -1460)):
#     """Reads xy data from XML dataframe from Nikon ND Analysis software.
#
#     Designed to convert micron coordinates at which videos were collected
#     to pixels within a large tilescan image which contains coordinates at which
#     videos were collected.
#
#     Parameters
#     ----------
#     xmlfile : string
#         Filename of XML file to be read
#     umppx : int or float
#         Microns per pixel ratio at which images were collected. If None, no
#         conversion will be performed.
#     offset : Coordinates of upper left hand corner of collected tilescan image?
#         Required for accurate conversion from microns to pixels
#
#     Returns
#     -------
#     xmlpoints : list of tuple of float
#         XY points contained in XML file and converted to pixels
#
#
#     """
#     tree = et.parse(xmlfile)
#     root = tree.getroot()
#
#     y = []
#     x = []
#     xmlpoints = []
#     counter = 0
#
#     for point in root[0]:
#         if counter > 1:
#             x = float(point[2].attrib['value'])
#             y = float(point[3].attrib['value'])
#             if umppx is None:
#                 xmlpoints.append((x, y))
#             else:
#                 xmlpoints.append(((x-offset[0])/umppx, (y-offset[1])/umppx))
#         counter = counter + 1
#
#     return xmlpoints
#
#
# def crop_to_videodims(cell_image, channel=None, vidpoint=(600, 600), dim=512,
#                       fname='test.tif'):
#     """Crops subimage from large tilescan image
#
#     Designed to cut subimages from tilescan images that align with videos
#     collected at points within that image. Meant to register videos collected
#     via fluorescent imaging with tilescan images collected via confocal imaging.
#
#     Parameters
#     ----------
#     cell_image : numpy.ndarray
#         Large tilescan image
#     channel : int
#         If multichannel is True, reads in image corresponding to this channel in
#         file. Currently can't handly multichannel images
#     vidpoint : tuple of float or int
#         XY coordinates in pixels at which to crop
#     dim : int
#         Dimensions of desired output image
#     fname : string
#         Desired filename of output image. If None, saving step will be skipped
#
#     """
#
#     ndim = dim
#     if channel is None:
#         subim = cell_image[int(vidpoint[0]-ndim/2):int(vidpoint[0]+ndim/2),
#                            int(vidpoint[1]-ndim/2):int(vidpoint[1]+ndim/2)]
#
#     if fname is not None:
#         sio.imsave(fname, subim)
#
#     return subim
