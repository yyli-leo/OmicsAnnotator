import numpy as np
import matplotlib.pyplot as plt
# import multiprocess as mp
from scipy.stats import percentileofscore
from .single_channel_featurize import SingleChannelHistogram


def get_cells_within_roi(xyrange, cell_seg_df):
    """ Get Cell IDs within a specified ROI """
    x0, x1, y0, y1 = xyrange
    assert x1 > x0
    assert y1 > y0
    sub_df = cell_seg_df[
        (cell_seg_df['X'] >= x0) & (cell_seg_df['X'] <= x1) & (cell_seg_df['Y'] >= y0) & (cell_seg_df['Y'] <= y1)
    ]
    return list(sub_df['CELL_ID'])


def subset_feature_df(feature_df, cell_ids):
    """ Subset feature dataframe to a specified set of cells """
    sub_feature_df = feature_df[feature_df['CELL_ID'].isin(cell_ids)]
    return sub_feature_df


def calculate_signal_average(feature_df, cell_ids):
    """ Calculate average cell signal intensity in the region of interest """
    sub_feat = subset_feature_df(feature_df, cell_ids)
    signal_average = sub_feat.mean(0)
    return signal_average


def batch_zscores(ref_values, query_values):
    """ Calculate z-scores for a batch of query values against reference values. """
    # Make sure everything is 2D: N by C
    ref_values = np.array(ref_values)
    if len(ref_values.shape) == 1:
        ref_values = ref_values.reshape((-1, 1))
    query_values = np.array(query_values)
    if len(query_values.shape) == 1:
        query_values = query_values.reshape((-1, 1))
    assert query_values.shape[1] == ref_values.shape[1]

    ref_means = ref_values.mean(0, keepdims=True)
    ref_stds = ref_values.std(0, keepdims=True)
    z_scores = (query_values - ref_means) / ref_stds
    assert z_scores.shape == query_values.shape
    return z_scores


def batch_percentiles(ref_values, query_values):
    """ Calculate percentiles for a batch of query values against reference values. """
    # Make sure everything is 2D: N by C
    ref_values = np.array(ref_values)
    if len(ref_values.shape) == 1:
        ref_values = ref_values.reshape((-1, 1))
    query_values = np.array(query_values)
    if len(query_values.shape) == 1:
        query_values = query_values.reshape((-1, 1))
    assert query_values.shape[1] == ref_values.shape[1]

    percentiles = [percentileofscore(ref_values[:, i], query_values[:, i]) for i in range(ref_values.shape[1])]
    percentiles = np.stack(percentiles, 1)
    assert percentiles.shape == query_values.shape
    return percentiles


def get_reference_patch_list(cell_seg_df,
                             patch_sizes=[128],
                             n_cells_min=5,
                             n_cells_max=50,
                             count=1e3,
                             seed=123):
    """ Get a list of reference patches for a region

    Args:
        cell_seg_df (pd.DataFrame): DataFrame containing cell centroid coordinates
        patch_sizes (list): list of patch sizes for sampling sliding windows to allow multi-scale sampling,
            also see `get_sliding_window_reference_ROIs`
        n_cells_min (int): see `get_sliding_window_reference_ROIs`
        n_cells_max (int): see `get_sliding_window_reference_ROIs`
        count (int): number of patches to generate
        seed (int): random seed for reproducibility

    Returns:
        list: list of list of CELL_IDs
    """
    patch_list = []
    while len(patch_list) < 2 * count:
        patch_dict = {}
        for ps in patch_sizes:
            patch_dict.update(get_sliding_window_reference_ROIs(
                cell_seg_df, patch_size=ps,
                n_cells_min=n_cells_min, n_cells_max=n_cells_max,
                num_repetition=1, stride_ratio=0.5, seed=seed))
        patch_list.extend([v['cell_ids'] for v in patch_dict.values()])

    patch_list = [patch_list[ii] for ii in 
                  np.random.choice(np.arange(len(patch_list)), (int(count),), replace=False)]
    return patch_list


def get_sliding_window_reference_ROIs(cell_seg_df,
                                      patch_size=128,
                                      n_cells_min=5,
                                      n_cells_max=50,
                                      num_repetition=2,
                                      stride_ratio=0.5,
                                      seed=123):
    """Generate batch of queries for sliding window patches for a region

    Args:
        cell_seg_df (pd.DataFrame): DataFrame containing cell centroid coordinates
        patch_size (int): size of the patch
        n_cells_min (int): minimum number of cells in a patch that will be kept
        n_cells_max (int): maximum number of cells in a patch; if more cells are present, a random subset is taken
        num_repetition (int): number of times to repeat the sliding window process
        stride_ratio (float): ratio of the patch size to the stride size
        seed (int): random seed for reproducibility

    Returns:
        dict: dictionary with patch names: lists of CELL_IDs in the patch
    """
    stride = int(patch_size * stride_ratio)
    xmin, xmax = cell_seg_df['X'].min(), cell_seg_df['X'].max()
    ymin, ymax = cell_seg_df['Y'].min(), cell_seg_df['Y'].max()

    if seed:
        np.random.seed(seed)

    patch_dict = {}
    for _ in range(num_repetition):
        x_start = np.random.randint(xmin - stride, xmin - stride + patch_size)
        y_start = np.random.randint(ymin - stride, ymin - stride + patch_size)
        x_steps = int((xmax - x_start) // stride + 1)
        y_steps = int((ymax - y_start) // stride + 1)

        for i_x in range(x_steps):
            for i_y in range(y_steps):
                xyrange = (
                    x_start + i_x * stride,
                    x_start + i_x * stride + patch_size,
                    y_start + i_y * stride,
                    y_start + i_y * stride + patch_size,
                )
                patch_name = "{}-{}-{}-{}".format(*xyrange)
                if patch_name not in patch_dict:
                    cids = get_cells_within_roi(xyrange, cell_seg_df)
                    if len(cids) < n_cells_min:
                        continue
                    if len(cids) > n_cells_max:
                        cids = np.random.choice(cids, (n_cells_max,), replace=False)
                    ROI = {
                        'xyrange': xyrange,
                        'center_x': (xyrange[0] + xyrange[1]) / 2,
                        'center_y': (xyrange[2] + xyrange[3]) / 2,
                        'cell_ids': cids,
                        'patch_name': patch_name,
                    }
                    patch_dict[patch_name] = ROI
    return patch_dict


def get_ROI_image_feature(ROI, full_img, featurizer=SingleChannelHistogram()):
    """ Get image features for a region of interest """
    x0, x1, y0, y1 = ROI['xyrange']
    x0 = max(0, x0)
    x1 = min(full_img.shape[2], x1)
    y0 = max(0, y0)
    y1 = min(full_img.shape[1], y1)

    ROI_img = full_img[:, int(y0):int(y1), int(x0):int(x1)]
    patch_image_feats = [featurizer.featurize(im) for im in ROI_img]
    patch_image_feats = np.concatenate(patch_image_feats, 0)
    return patch_image_feats


def plot_region_with_ROIs(ROIs, full_img, colors=['w'], use_channels=[], frame_on=False, show_image=True):
    """ Plot ROI centers on a full image """
    if use_channels == []:
        use_channels = sorted(np.random.choice(np.arange(full_img.shape[0]), (3,), replace=False))

    plot_image = full_img[use_channels].transpose((1, 2, 0)).astype(float)
    plot_image = np.clip(plot_image / np.quantile(plot_image, 0.95), 0., 1.)

    if isinstance(ROIs, dict):
        ROIs = [ROIs]

    if isinstance(colors, str):
        colors = [colors] * len(ROIs)
    elif isinstance(colors, list) and len(colors) == 1:
        colors = colors * len(ROIs)

    if show_image:
        plt.imshow(plot_image)
    for ROI, color in zip(ROIs, colors):
        xc, yc = ROI['center_x'], ROI['center_y']
        plt.scatter(xc, yc, c=color, s=5, marker='x')
        if frame_on:
            x0, x1, y0, y1 = ROI['xyrange']
            plt.gca().add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, lw=1))
    return


def plot_region_with_biomarkers(full_img, full_img_bms, channels=[], norm_factors=[], colors=[]):
    """ Plot full image """
    img = np.zeros((full_img.shape[1], full_img.shape[2], 3), dtype=float)
    channels = channels if channels else full_img_bms
    norm_factors = norm_factors if norm_factors else [1] * len(channels)
    assert len(channels) == len(norm_factors)
    default_colors = [
        (1, 0, 0),  # Red
        (0, 1, 0),  # Green
        (0, 0, 1),  # Blue
        (1, 1, 0),  # Yellow
        (0, 1, 1),  # Cyan
        (1, 0, 1),  # Magenta
        (0.5, 0.5, 0.5),  # Gray
    ]
    colors = colors if colors else default_colors[:len(channels)]

    assert len(channels) <= len(colors)

    for bm, nf, color in zip(channels, norm_factors, colors):
        if bm is None:
            continue
        c_img = full_img[full_img_bms.index(bm)]
        img += c_img[..., np.newaxis] * (float(nf) * np.array(color, dtype=float).reshape((1, 1, 3)))

    if np.quantile(img, 0.95) > 1:
        img = img / np.quantile(img, 0.95)
    return img