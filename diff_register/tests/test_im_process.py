import os.path as op
import numpy as np
import skimage.io as sio
from skimage.morphology import square, opening, closing
import diff_register.im_process as imp


def test_fuzzy_contrast():
    np.random.seed(seed=1)
    testim = np.random.sample(size=(10, 10))
    sio.imsave('test.tif', testim)

    fuzzim = imp.fuzzy_contrast('.', 'test.tif', show=False)
    assert np.average(fuzzim) == 111.15

    # test 2
    np.random.seed(seed=1)
    testim = np.random.sample(size=(10, 10, 2))
    sio.imsave('test.tif', testim)

    fuzzim = imp.fuzzy_contrast('.', 'test.tif', channel=1, show=False)
    assert np.average(fuzzim) == 121.17


def test_binary_image():
    np.random.seed(seed=1)
    testim = np.random.sample(size=(10, 10))
    sio.imsave('test.tif', testim)

    testbi = imp.binary_image('.', 'test.tif', threshold=0.5,
                              show=False, imname='testbi.tif')
    assert np.average(testbi) == 124.95

    # test 2
    np.random.seed(seed=1)
    testim = np.random.exponential(1, size=(40, 40))
    testim = testim / np.max(testim)
    sio.imsave('test.tif', testim)
    testbi = imp.binary_image('.', 'test.tif', threshold=0.01,
                              ajar=True, imname='testbi.tif')
    assert np.round(np.average(testbi), 1) == 223.6

    # test 3
    testbi = imp.binary_image('.', 'test.tif', threshold=0.01,
                              ajar=True, close=True,
                              imname='testbi.tif')
    assert np.round(np.average(testbi), 1) == 244.5

    # test 4
    testbi = imp.binary_image('.', 'test.tif', threshold=0.01,
                              show=True, imname=None)
    assert op.isfile('clean_test.tif')


def test_clean_image():
    # test 1
    np.random.seed(seed=1)
    testim = np.random.exponential(1, size=(40, 40))
    testim = testim / np.max(testim)
    sio.imsave('test.tif', testim)
    testcl, props = imp.clean_image('.', 'test.tif', threshold=0.01,
                                    area_thresh=10, ajar=True,
                                    close=True, imname=None)
    assert np.round(np.average(testcl), 1) == 244.5

    # test 2
    testcl, props = imp.clean_image('.', 'test.tif', threshold=0.01,
                                    area_thresh=10, ajar=True,
                                    close=False, imname=None)
    assert np.round(np.average(testcl), 1) == 222.6

    # test 3
    testcl, props = imp.clean_image('.', 'test.tif', threshold=0.01,
                                    area_thresh=10, ajar=False,
                                    close=False, imname='clim.tif')
    assert np.round(np.average(testcl), 1) == 238.3
    assert op.isfile('clim.tif')


def test_label_image():
    # test 1
    np.random.seed(seed=1)
    testim = np.random.exponential(1, size=(40, 40))
    testim = testim / np.max(testim)
    sio.imsave('test.tif', testim)

    testbi = imp.binary_image('.', 'test.tif', threshold=0.01,
                              show=False, imname='testbi.tif')

    # test 2
    testcl, props = imp.label_image('.', 'testbi.tif', area_thresh=10,
                                    imname='clim.tif')
    assert np.round(np.average(testcl), 1) == 238.3
    assert op.isfile('clim.tif')

    testcl, props = imp.label_image('.', 'testbi.tif', area_thresh=1,
                                    imname=None)
    assert op.isfile('short_testbi.tif')


def test_skeleton_image():
    np.random.seed(seed=1)
    testim = np.random.exponential(1, size=(40, 40))
    testim = testim / np.max(testim)
    sio.imsave('test.tif', testim)

    skeleton = imp.skeleton_image('.', 'test.tif', threshold=0.3,
                                  area_thresh=0.1, tofilt=False, ajar=False,
                                  close=False, imname='skel.tif',
                                  channel=None, show=False)
    assert np.round(np.average(skeleton.im), 3) == 0.136
    assert np.sum(skeleton.nbran) == 51
    assert (51, 14) == skeleton.branchdat.values.shape
    assert op.isfile('skel.tif')

    skeleton = imp.skeleton_image('.', 'test.tif', threshold=0.01,
                                  area_thresh=0.0, tofilt=True, ajar=True,
                                  close=True, imname=None,
                                  channel=None, show=False)
    assert np.round(np.average(skeleton.im), 3) == 0.002
    assert np.sum(skeleton.nbran) == 1
    assert (1, 14) == skeleton.branchdat.values.shape
    assert op.isfile('skel_filt_test.tif')


def test_mglia_features():
    np.random.seed(seed=1)
    testim = np.random.binomial(1, 0.1, size=(40, 40))
    testim = testim / np.max(testim)
    # testim = closing(opening(opening(testim, square(3)),
    # linsquare(3)), square(3))
    testim = opening(closing(closing(testim, square(3)), square(3)), square(3))
    sio.imsave('test.tif', testim)

    skeleton = imp.skeleton_image('.', 'test.tif', threshold=0.4, area_thresh=1,
                                  tofilt=False, ajar=False, close=True,
                                  imname='skel.tif', channel=None)
    mfeat = imp.mglia_features(skeleton)

    assert 12 == np.sum(mfeat['total_branches'].values)
    assert 15.8 == np.round(np.mean(mfeat['area'].values), 1)
    assert 2.36 == np.round(np.mean(mfeat['average_branch'].values), 2)
    assert 0.96 == np.round(np.mean(mfeat['solidity'].values), 2)


def test_features_hist():
    print()


def test_read_xml_points():
    print()


def test_crop_to_video_dims():
    print()
