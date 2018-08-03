import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import skimage.io as sio
import matplotlib.pyplot as plt
import skimage
from skimage.morphology import square, opening, closing, skeletonize
from skimage.measure import regionprops, label
from skan import csr, draw


def fuzzy_contrast(folder, image_file, figsize=(10, 10), show=False):
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
    fname = '{}/{}'.format(folder, image_file)
    test_image = sio.imread(fname)
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
    sio.imsave(folder+'/'+output, rf_image)

    return rf_image


def binary_image(folder, image_file, threshold=2, figsize=(10, 10), op_image=False, close=False, show=False,
                 multichannel=False, channel=0, default_name=True, fname='test.png'):
    """
    Create binary image from input image with optional opening step.

    Parameters
    ----------

    Returns
    -------

    Examples
    --------

    """

    fname = '{}/{}'.format(folder, image_file)
    if multichannel:
        test_image = sio.imread(fname)[:, :, channel]
    else:
        test_image = sio.imread(fname)
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
    
    if default_name:
        output = "clean_{}.png".format(image_file.split('.')[0])
    else:
        output = fname

    sio.imsave(folder+'/'+output, op_image)

    return op_image


def clean_image(folder, image_file, threshold=2, figsize=(10, 10), op_image=False, close=False, show=False,
                area_thresh=50, multichannel=False, channel=0, default_name=True, fname='test.png'):
    """
    Create binary image from input image with optional opening step.

    Parameters
    ----------

    Returns
    -------

    Examples
    --------

    """

    fname = '{}/{}'.format(folder, image_file)
    if multichannel:
        test_image = sio.imread(fname)[:, :, channel]
    else:
        test_image = sio.imread(fname)
    bi_image = test_image > threshold

    if open is True:
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
    
    ##Labelling and cleaning up image.
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
    
    if default_name:
        output = "short_{}.png".format(image_file.split('.')[0])
    else:
        output = fname

    sio.imsave(folder+'/'+output, short_image)

    return short_image, short_props


def label_image(folder, image_file, area_thresh=50, figsize=(10, 10), show=False, default_name=True, fname='test1.png'):
    """
    Create label image and calculate region properties.

    Parameters
    ----------

    Returns
    -------

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
    
    if default_name:
        output = "short_{}.png".format(image_file.split('.')[0])
    else:
        output = fname

    sio.imsave(folder+'/'+output, short_image)

    return short_image, short_props


def skeleton_image(folder, image_file, threshold=50, area_thresh=50, figsize=(10, 10), show=False, multichannel=False, channel=0,
                   disp_binary = True, default_name=True, fname='test2.png'):
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
    fname = '{}/{}'.format(folder, image_file)
    image0 = sio.imread(fname)
    if multichannel:
        image0 = np.ceil(255* (image0[:, :, channel] / image0[:, :, channel].max())).astype(int)
    else:
        image0 = np.ceil(255* (image0[:, :] / image0[:, :].max())).astype(int)
    image0 = skimage.filters.median(image0)
    filt = 'filt_{}.png'.format(image_file.split('.')[0])
    sio.imsave(folder+'/'+filt, image0)

    #threshold the image
    binary0 = binary_image(folder, filt, threshold=threshold, close=True, show=False)
    clean = 'clean_{}'.format(filt)

    #label image
    short_image, props = label_image(folder, clean, area_thresh=area_thresh, show=False)
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
        fig, ax = plt.subplots(figsize=figsize)
        if disp_binary:
            draw.overlay_euclidean_skeleton_2d(short_image, branch_data_short,
                                               skeleton_color_source='branch-type', axes=ax)
        else:
            draw.overlay_euclidean_skeleton_2d(image0, branch_data_short,
                                               skeleton_color_source='branch-type', axes=ax)
        
    if default_name:
        output = 'skel_{}'.format(short)
    else:
        output = fname
    plt.savefig('{}/{}'.format(folder, output))

    return skeleton0, branch_data_short, nbranches, short_image, props


def mglia_features(props, branch_data_short, convert=False, umppx=1):
    """
    Note: raises error if skeleton is missing from any cell in the binary image.
    """
    
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
    #total_processes = np.zeros((len(props)))
    #avg_p_length = np.zeros((len(props)))
    #main_process = np.zeros((len(props)))

    #properties that can be found from sklearn.measure.regionprops
    counter = 0
    for item in props:
        X[counter] = item.centroid[0]
        Y[counter] = item.centroid[1]
        perimeter[counter] = item.perimeter
        areas[counter] = item.area
        eccentricity[counter] = item.eccentricity
        inertia_tensor[counter] = item.inertia_tensor
        label[counter] = item.label
        #max_intensity[counter] = item.max_intensity
        #mean_intensity[counter] = item.mean_intensity
        moments[counter] = item.moments
        solidity[counter] = item.solidity
        counter = counter + 1
        
    #properties associated with processes as found from skan
    mglia = branch_data_short['skeleton-id'].max()
    nbranches = []
    avg_p_length = []
    main_process = []
    
    xs = []
    ys = []

    ncount = 0
    for i in branch_data_short['skeleton-id'].unique():
        bcount = branch_data_short[branch_data_short['skeleton-id']==i]['skeleton-id'].count()
        bavg = np.mean(branch_data_short[branch_data_short['skeleton-id']==i]['branch-distance'])
        blong = np.max(branch_data_short[branch_data_short['skeleton-id']==i]['branch-distance'])
        xcoord = np.mean([np.mean(branch_data_short[branch_data_short['skeleton-id']==i]['img-coord-0-0']), 
                         np.mean(branch_data_short[branch_data_short['skeleton-id']==i]['img-coord-1-0'])])
        xs.append(xcoord)
        ycoord = np.mean([np.mean(branch_data_short[branch_data_short['skeleton-id']==i]['img-coord-0-1']), 
                         np.mean(branch_data_short[branch_data_short['skeleton-id']==i]['img-coord-1-1'])])
        ys.append(ycoord)
        nbranches.append(bcount)
        avg_p_length.append(bavg)
        main_process.append(blong)
        
    nbranches_ord = [0]*len(nbranches)
    avg_p_length_ord = [0]*len(nbranches)
    main_process_ord = [0]*len(nbranches)

    for i in range(0, len(xs)):
        #print(xs[i], ys[i])
        skel_id = i
        min_function = np.square(xs[i] - X)+np.square(ys[i] - Y)
        mglia_id = np.argmin(min_function)
        nbranches_ord[mglia_id] = nbranches[skel_id]
        avg_p_length_ord[mglia_id] = avg_p_length[skel_id]
        main_process_ord[mglia_id] = main_process[skel_id]
        #print(mglia_id)
        #print(np.min(min_function))
    
    if convert:
        factor = umppx
    else:
        factor = 1

    features = pd.DataFrame({ 'X' : X*factor,
                              'Y' : Y*factor,
                              'perimeter' : perimeter*factor,
                              'area' : areas*factor*factor,
                              'eccentricity' : eccentricity,
                              'inertia_tensor' : inertia_tensor,
                              'label' : label,
                              #'max intensity' : max_intensity,
                              #'mean intensity' : mean_intensity,
                              'moments' : moments,
                              'solidity' : solidity,
                              'total_branches' : nbranches_ord,
                              'average_branch' : [x*factor for x in avg_p_length_ord],
                              'main_branch' : [x*factor for x in main_process_ord]
                            })
    
    return features


def features_hist(features, feature, bin_size=100, bin_range=5000):
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

    
def read_xmlpoints(xmlfile, converttopix = True, umppx=0.62, offset=(17000, -1460)):
    tree = et.parse(xmlfile)
    root = tree.getroot()

    y = []
    x = []
    xmlpoints = []
    counter = 0
    
    for point in root[0]:
        if counter > 1:
            x = float(point[2].attrib['value'])
            y = float(point[3].attrib['value'])
            if converttopix:
                xmlpoints.append(((x-offset[0])/umppx,(y-offset[1])/umppx))
            else:
                xmlpoints.append((x, y))
        counter = counter + 1

    return xmlpoints


def crop_to_videodims(cell_image, multichannel = False, vidpoint=(600, 600), defaultdims=True, dim=512, save=True,
                      fname='test.tif'):
    
    if defaultdims:
        ndim = 512
    else:
        ndim = dim

    if not multichannel:
        subim = cell_image[int(vidpoint[0]-ndim/2):int(vidpoint[0]+ndim/2), int(vidpoint[1]-ndim/2):int(vidpoint[1]+ndim/2)]

    if save:
        sio.imsave(fname, subim)
        
    return subim



