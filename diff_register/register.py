def downsample(image, ratio):
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
