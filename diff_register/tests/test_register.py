import os.path as op
import numpy as np
import xml.etree.ElementTree as et
import skimage.io as sio
from skimage.morphology import opening, closing, square
import diff_register.register as reg


def test_read_xml_points():
    # Create test XML file
    root = et.Element("variant", version='1.0')
    doc = et.SubElement(root, "no_name", runtype='CLxListVariant')

    et.SubElement(doc, "bIncludeZ", runtype='bool', value='true')
    et.SubElement(doc, "bPFSEnabled", runtype='bool', value='false')

    point1 = et.SubElement(doc, "Point00000",
                           runtype='NDSetupMultipointListItem')
    et.SubElement(point1, "bChecked", runtype="bool", value='true')
    et.SubElement(point1, "strName", runtype="DLxStringW", value='true')
    et.SubElement(point1, "dXPosition", runtype="double", value='250.0')
    et.SubElement(point1, "dYPosition", runtype="double", value='250.0')
    et.SubElement(point1, "dZPosition", runtype="double", value='50.0')
    et.SubElement(point1, "dPFSOffset", runtype="double", value='-1')
    et.SubElement(point1, "baUserData", runtype="CLxByteArray", value='')

    point2 = et.SubElement(doc, "Point00001",
                           runtype='NDSetupMultipointListItem')
    et.SubElement(point2, "bChecked", runtype="bool", value='true')
    et.SubElement(point2, "strName", runtype="DLxStringW", value='true')
    et.SubElement(point2, "dXPosition", runtype="double", value='750.0')
    et.SubElement(point2, "dYPosition", runtype="double", value='750.0')
    et.SubElement(point2, "dZPosition", runtype="double", value='100.0')
    et.SubElement(point2, "dPFSOffset", runtype="double", value='-1')
    et.SubElement(point2, "baUserData", runtype="CLxByteArray", value='')

    tree = et.ElementTree(root)
    tree.write("testxml.xml")

    # Create test image
    np.random.seed(seed=1)
    testim = np.random.binomial(1, 0.1, size=(1000, 1000))
    testim = testim / np.max(testim)
    testim = opening(closing(closing(closing(closing(closing(testim, square(3)),
                     square(3)), square(3)),
                     square(3)), square(3)), square(3))
    sio.imsave('cellim.tif', testim)

    xmlmod = reg.read_xmlpoints('testxml.xml', 'cellim.tif',
                                umppx=1, offset=(10, 0))

    assert xmlmod == [(760.0, 250.0), (260.0, 750.0)]


def test_crop_to_video_dims():
    print('test')


def test_downsample():
    print('test')


def test_plot_together():
    print('test')


def test_initial_alignment():
    print('test')


def test_fine_alignment():
    print('test')
