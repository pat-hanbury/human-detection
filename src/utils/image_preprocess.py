import numpy as np
import cv2
import json
from torch import Tensor, stack
from os import listdir


def path_to_pts(d, target_dim, s_x=1, s_y=1):
    """Converts path from a list of dictionaries with x and y positions
    into a list of points.

    Arguments:
        d {list}                -- List of dictionaries, each with 'x' and 'y' keys
                                   corresponding to float values.
        target_dim {tuple[int]} -- tuple containing (target_width, target_height)

    Keyword Arguments:
        s_x {float} -- Stretch factor of the width (default: {1})
        s_y {float} -- Stretch factor of the height (default: {1})

    Returns:
        [numpy.ndarray] -- Numpy array with new pts
    """
    (target_width, target_height) = target_dim
    pts = []
    for i in d:
        pt = [i['x']*s_x/target_width, i['y']*s_y/target_height]  # stretch points to new pixels
        pts.append(pt)
    return np.array(pts)


def polygon_to_box(poly_paths):
    """Turn polygon bounding annotations into a square bounding box.

    Arguments:
        poly_paths {numpy array} -- Array containing polygon points in order,
                                    formatted to be compatible with cv2.polylines


    Returns:
        {numpy array}            -- Length 5 1D numpy float array with pixel locations for left, right, top, and bottom
                                    values for indices 0-4 and constant of bounding box for indices. 5th index is 1
                                    representing the class. For SSD300 people detection application, class will always 
                                    be 1 (person)
    """
    initialized = False
    # expand bounding box
    for pt in poly_paths:
        pt = pt[0]
        if not initialized:
            # initialize bounding box
            left = pt[0]
            right = pt[0]
            top = pt[1]
            bottom = pt[1]
            initialized = True

        x, y = pt
        # update x bounds
        if x < left:
            left = x
        elif x > right:
            right = x
        # update y bounds
        if y < bottom:
            bottom = y
        elif y > top:
            top = y
    return np.asarray([left, bottom, right, top, 1.0])  # the `1` is added to format the annotation as needed for SSD300


def format_annotations(data, target_dim, s_x=1, s_y=1):
    """Formats the annotations from nested dictionaries
    to shape used by cv2.polylines

    Arguments:
        data {list}             -- List of all annotations for a single image.
        target_dim {tuple[int]} -- tuple containing (target_width, target_height)

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
        poly_path = path_to_pts(poly_path, target_dim, s_x, s_y).reshape((-1, 1, 2))
        annot_list.append(poly_path)
    return annot_list


def resize(img, annotations, target_height, target_width):
    """Resizes a single image along with all its annotations
    into given target size. The points of polygon shape annotations
    are remapped to fit the new size of the image

    Arguments:
        img {numpy array}       -- Can be any height and width,
                                   but with 3 channels for RGB
        annotations {dict}      -- Contains all annotation data
                                   as formatted by Darwin.
        target_height {integer} -- Pixel height of the target image
        target_width {integer}  -- Pixel width of the target image

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
    pts = format_annotations(annotations, target_dim, s_x=stretch_x, s_y=stretch_y)

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

    Raises:
        TypeError -- when scaling option is not recognized.

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
        return (img - img.mean()) / img.std()  # (X - mean) / std
    else:
        raise TypeError("'{}' invalid keyword argument. ".format(scaling) +
                        "Only minmax normalization and "
                        "zscore standardization are supported.")


def get_dataset(configs, height, width, scaling='minmax'):
    """Grabs images, preprocesses them, and returns list of tensors.
    Also preprocesses annotations.

    Arguments:
        configs {dict} -- Contains config options for training.
                          Uses "dataset_root" and "batch_size".
                                 should contain both images and annotation files.
        height {int} -- Pixel height required by the model.
        width {int} -- Pixel width required by the model.

    Keyword Arguments:
        scaling {str} -- Describes the normalization method. (default: {'minmax'})

    Returns:
        List: tensor batches (Bx300x300x3), List: List: tensor (Nx4)
        -- Two ordered lists (image batches, annotation batch lists). The
           annotation batch lists contain tensors of varying length due to
           there being different number of annotations per image.
    """
    dataset_path = configs["dataset_root"]
    batch_size = configs["batch_size"]
    # files, sorted so that img & annots are in same order
    image_names = sorted([i for i in listdir(dataset_path) if i[-4:] == '.png'])
    ann_names = sorted([i for i in listdir(dataset_path) if i[-5:] == '.json'])

    # import images and convert to tensor
    images = []
    batch_img = []
    annotations = []
    batch_ann = []

    # for each image
    for i, iname, aname in zip(range(1, len(ann_names) + 1),
                               image_names,
                               ann_names):
        # get corresponding image/annot
        temp_img = cv2.imread(dataset_path + iname)  # image
        with open(dataset_path + aname) as ann_file:  # annotation
            temp_ann = json.load(ann_file)
        # normalize image
        temp_img = normalize_img(temp_img, scaling)
        # resize
        temp_img, temp_ann = resize(temp_img, temp_ann['annotations'],
                                    height, width)
        # bounding boxes
        temp_boxes = []
        for poly in temp_ann:
            temp_boxes.append(Tensor(polygon_to_box(poly)))

        # append as tensors
        batch_img.append(Tensor(temp_img))
        batch_ann.append(stack(temp_boxes))
        if i % batch_size == 0:
            images.append(stack(batch_img).permute(0, 3, 1, 2))
            annotations.append(batch_ann)
            batch_img = []
            batch_ann = []

    # last batch smaller than batch size
    if len(batch_img) > 0:
        images.append(stack(batch_img).permute(0, 3, 1, 2))  # (32, 300, 300, 3) (32, 3, 300, 300)
        annotations.append(batch_ann)

    return images, annotations
