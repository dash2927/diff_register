import os.path as op
import numpy as np
import skimage.io as sio
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
    assert os.path.isfile('clim.tif')


def test_label_image():
    print()


def test_skeleton_image():
    print()


def test_mglia_features():
    print()


def test_features_hist():
    print()


def test_read_xml_points():
    print()


def test_crop_to_video_dims():
    print()
