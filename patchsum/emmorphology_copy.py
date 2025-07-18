from collections import OrderedDict
from skimage.measure import regionprops
import numpy as np
import pandas as pd


def get_morphology_from_mask(label_image):
    """Calculate morphological features from segmentation mask

    Args:
        label_image: labeled image mask where pixel intensity of positive integers is CELL_ID and zero is background.

    Returns:
        morph_data (pandas.DataFrame): a dataframe with the following columns of morphology features.

            CELL_ID (int):
                Cell ids unique to the image/acquisition_id.

            Orientation.Orientation (float):
                Angle between the horizonal axis and the major axis of the ellipse
                that has the same second moments as the mask,
                The value is in the interval [-pi/2, pi/2] counter-clockwise.

            Size.Area (int):
                Number of pixels of the mask.

            Size.Area_ConvexHull (int):
                Number of pixels of convex hull image, which is the smallest convex polygon that encloses the mask.

            Size.Area_Filled (int):
                Number of pixels of the mask will all the holes filled in.

            Size.MajorAxisLength (float):
                The length of the major axis of the ellipse that has the same
                normalized second central moments as the mask.

            Size.MinorAxisLength (float):
                The length of the minor axis of the ellipse that has the same
                normalized second central moments as the mask.

            Shape.Circularity (float):
                Circularity of the mask.
                The value is in the interval [0, 1]. When it is 1, the mask is a perfect circle.

            Shape.Eccentricity (float):
                Eccentricity of the ellipse that has the same second-moments as the mask.
                The value is in the interval [0, 1). When it is 0, the ellipse is a circle.

            Shape.MinorMajorAxisRatio (float):
                Ratio of minor to major axis of the ellipse
                that has the same second-moments as the object region.

            Shape.Extent (float):
                Ratio of area of the mask to its bounding box.

            Shape.Solidity (float):
                Ratio of pixels of the mask to the convex hull image.

    """
    # compute region properties of each mask
    rprops = regionprops(label_image)

    # list of feature names
    featname_map = OrderedDict({
        'CELL_ID': 'label',
        'Centroid.X': None,
        'Centroid.Y': None,
        'Orientation.Orientation': 'orientation',
        'Size.Area': 'area',
        'Size.Area_ConvexHull': 'area_convex',
        'Size.Area_Filled': 'area_filled',
        'Size.MajorAxisLength': 'axis_major_length',
        'Size.MinorAxisLength': 'axis_minor_length',
        'Shape.Circularity': None,
        'Shape.Eccentricity': 'eccentricity',
        'Shape.MinorMajorAxisRatio': None,
        'Shape.Extent': 'extent',
        'Shape.Solidity': 'solidity',
    })
    feature_list = featname_map.keys()
    mapped_feats = [k for k, v in featname_map.items() if v is not None]

    # create pandas.DataFrame with the features
    numFeatures = len(feature_list)
    numLabels = len(rprops)
    fdata = pd.DataFrame(np.zeros((numLabels, numFeatures)),
                         columns=feature_list)

    for i, nprop in enumerate(rprops):

        # get features from skimage.measure.regionprops
        for name in mapped_feats:
            fdata.at[i, name] = nprop[featname_map[name]]

        # get centroid coordinates
        fdata.at[i, 'Centroid.X'] = nprop.centroid[0]
        fdata.at[i, 'Centroid.Y'] = nprop.centroid[1]

        # compute 'Shape.Circularity'
        numerator = 4 * np.pi * nprop.area
        denominator = nprop.perimeter ** 2
        if denominator > 0:
            fdata.at[i, 'Shape.Circularity'] = numerator / denominator

        # compute 'Shape.MinorMajorAxisRatio'
        if nprop.axis_major_length > 0:
            fdata.at[i, 'Shape.MinorMajorAxisRatio'] = nprop.axis_minor_length / nprop.axis_major_length
        else:
            fdata.at[i, 'Shape.MinorMajorAxisRatio'] = 1

    fdata['CELL_ID'] = fdata['CELL_ID'].astype(int)

    return fdata
