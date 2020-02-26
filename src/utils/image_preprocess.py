import numpy as np
import cv2

def path_to_pts(d, s_x=1, s_y=1):
    """Converts path from a list of dictionaries with x and y positions
    into a list of points.

    Arguments:
        d {list} -- List of dictionaries, each with 'x' and 'y' keys
                    corresponding to float values.

    Keyword Arguments:
        s_x {float} -- Stretch factor of the width (default: {1})
        s_y {float} -- Stretch factor of the height (default: {1})

    Returns:
        [numpy.ndarray] -- Numpy array with new pts
    """
    pts = []
    for i in d:
        pt = [int(i['x']*s_x), int(i['y']*s_y)]  # stretch points to new pixels
        pts.append(pt)
    return np.array(pts)


def polygon_to_box(poly_path):
    raise NotImplementedError


def format_annotations(data, s_x=1, s_y=1):
    """Formats the annotations from nested dictionaries
    to shape used by cv2.polylines

    Arguments:
        data {list} -- List of all annotations for a single image.

    Keyword Arguments:
        s_x {float} --  Stretch factor of the width.
                        Used by path_to_pts (default: {1})
        s_y {float} --  Stretch factor of the height.
                        Used by path_to_pts (default: {1})

    Returns:
        [list] -- List of all formatted annotations for a single image.
    """
    annot_list = []
    for annot in data:
        poly_path = annot['polygon']['path']  # grab path
        # convert to pts in format for cv2.polylines
        poly_path = path_to_pts(poly_path, resize,
                                s_x, s_y).reshape((-1, 1, 2))
        annot_list.append(poly_path)
    return annot_list


def resize(img, annotations, target_height, target_width):
    """Resizes a single image along with all its annotations
    into given target size. The points of polygon shape annotations
    are remapped to fit the new size of the image

    Arguments:
        img {numpy array} -- Can be any height and width,
                             but with 3 channels for RGB
        annotations {dict} -- Contains all annotation data
                              as formatted by Darwin.
        target_height {integer} -- Pixel height of the target image
        target_width {integer} -- Pixel width of the target image

    Returns:
        [tuple] --  Image in desired shape, Altered annotations
                    in same format and order as input.

    Note:
        Ensure only value of 'annotations' from JSON file
        is passed to annotations argument.
    """

    # set original and target dimensions
    original_dim = (img.shape[1], img.shape[0])
    target_dim = (target_width, target_height)
    # stretch factors
    stretch_x = target_dim[0]/original_dim[0]
    stretch_y = target_dim[1]/original_dim[1]

    # resize image
    resized_img = cv2.resize(img, target_dim, interpolation=cv2.INTER_LINEAR)

    # resize annotations
    pts = format_annotations(annotations, resize=True,
                             s_x=stretch_x, s_y=stretch_y)

    return resized_img, pts


def normalize_img(img, scaling='minmax'):
    """Normalizes pixel values based on given method.

    Arguments:
        img {numpy array} -- Image. Can be any shape.

    Keyword Arguments:
        scaling {string} -- Dictates method for normalization.
                            Can take value 'minmax' or 'zscore'.
                            (default: {'minmax'})

    Returns:
        [numpy array] -- rescaled image.

    Note:
        Both currently supported scalings are LOCAL methods.
        For example, two images with different maximum values
        (one with max=255 and another with max=188) will both
        have maximum values of 1 after normalization.
    """
    if scaling == 'minmax':
        # (X - min) / (max - min)
        return np.interp(img, (img.min(), img.max()), (0, +1))
    elif scaling == 'zscore':
        flat_img = img.flatten()
        pixel_mean = np.mean(flat_img)  # get mean
        pixel_std = np.std(flat_img)  # get standard deviation
        return (img - pixel_mean) / pixel_std  # (X - mean) / std
    else:
        raise NotImplementedError('Currently only minmax normalization and \
                                  zscore standardization are supported.')


def get_dataset(configs):
    """
    configs["dataset_root"] = "path/to/root/of/datset"
    --- use this to grab images, do whatever preprocessing is needed,
    and output like this:

    Returns:
        List: tensor images (3x300x300), list: annotations
    """
    raise NotImplementedError