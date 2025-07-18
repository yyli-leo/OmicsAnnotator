import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import zarr

from .emmorphology_copy import get_morphology_from_mask
from .utils import calculate_signal_average, subset_feature_df, batch_percentiles, batch_zscores


def get_cell_segmentation_df(region_id, root_dir, **kwargs):
    """ Read cell segmentation dataframe """
    seg_path = os.path.join(root_dir, region_id, f"{region_id}.cell_data.csv")
    assert os.path.exists(seg_path), f'Cell segmentation file not found: {seg_path}'
    cell_seg_df = pd.read_csv(seg_path)
    return cell_seg_df


def get_cell_morphology_df(region_id, root_dir, **kwargs):
    """ Read segmentation mask and calculate morphology feature """
    seg_mask_path = os.path.join(root_dir, region_id, 'seg_mask')
    assert os.path.exists(seg_mask_path), f'Segmentation mask file not found: {seg_mask_path}'
    seg_mask = zarr.open_array(seg_mask_path)[:]
    # seg_mask = zarr.open_array(seg_mask_path, synchronizer=None).oindex[:]
    # seg_mask = np.array(zarr.open_array(seg_mask_path, synchronizer=None))

    cell_morph_df = get_morphology_from_mask(seg_mask)
    key_columns = [
        "Orientation.Orientation", "Size.Area", "Size.MajorAxisLength", "Size.MinorAxisLength",
        "Shape.Circularity", "Shape.Eccentricity", "Shape.Solidity",
    ]
    cell_morph_df = cell_morph_df[['CELL_ID'] + key_columns]
    return cell_morph_df


def calculate_densities(cell_seg_df, k=3, radius=70):
    """
    Calculate cell density and compactness metrics within a specified ROI.

    Parameters:
        cell_seg_df (DataFrame): A DataFrame containing columns 'CELL_ID', 'X', and 'Y'.
        k (int): The number of nearest neighbors to consider for compactness analysis (default is 5).
        radius (float): Radius to calculate local cell density (default is 50 units).

    Returns:
        density_metrics (DataFrame): A DataFrame 'mean_k_neighbor_distance' and 'local_density'.
    """
    coordinates = cell_seg_df[["X", "Y"]].values
    tree = cKDTree(coordinates)

    mean_k_neighbor_distances = []
    local_densities = []
    for i, coord in enumerate(coordinates):
        # Get distances to all neighbors
        distances, indices = tree.query(coord, k=k+1)
        mean_k_neighbor_distances.append(np.mean(distances[1:]))

        # Count neighbors within the radius (excluding the cell itself)
        neighbors_within_radius = tree.query_ball_point(coord, radius)
        local_densities.append(len(neighbors_within_radius) - 1)  # Exclude the cell itself

    density_metrics = cell_seg_df[["CELL_ID"]].copy()
    density_metrics["mean_k_neighbor_distance"] = mean_k_neighbor_distances
    density_metrics["local_density"] = local_densities
    return density_metrics


def calculate_orientation_metrics(cell_morph_df, cell_ids):
    """ Calculate mean orientation and circular variance for a set of cell IDs. """
    sub_df = subset_feature_df(cell_morph_df, cell_ids)
    orientations = sub_df['Orientation.Orientation'].values

    # Mean Orientation
    sin_mean = np.mean(np.sin(orientations))
    cos_mean = np.mean(np.cos(orientations))
    mean_angle = np.arctan2(sin_mean, cos_mean)
    # Circular Variance
    R = np.sqrt(sin_mean**2 + cos_mean**2)
    circular_var = 1 - R
    return {"mean_orientation": mean_angle, "circular_variance": circular_var}


def generate_cell_seg_description_for_target_patches(cell_id_target_subsets,
                                                     cell_id_reference_subsets,
                                                     cell_seg_df,
                                                     cell_morph_df,
                                                     k=3,
                                                     radius=70,
                                                     um_per_pixel=0.3775,
                                                     return_raw=False,
                                                     **kwargs):
    """ Generate description of cell distribution and morphology for target subsets

    Args:
        cell_id_target_subsets (list): list of lists, each individual list containing cell ids within the ROI
        cell_id_reference_subsets (list): list of lists, each individual list containing cell ids within a reference ROI
        cell_seg_df (pd.DataFrame): cell segmentation dataframe
        cell_morph_df (pd.DataFrame): cell morphology dataframe
        um_per_pixel (float): microns per pixel
        return_raw (bool, optional): if to return raw z-scores or text description
        **kwargs: additional arguments for text description

    Returns:
        np.ndarray or list: raw z-scores or text description
    """
    assert isinstance(cell_id_target_subsets, list)
    if isinstance(cell_id_target_subsets[0], int):
        cell_id_target_subsets = [cell_id_target_subsets]

    target_summaries = [{} for _ in range(len(cell_id_target_subsets))]

    # Calculate cell density features
    cell_density_df = calculate_densities(cell_seg_df, k=k, radius=radius)
    # Following features need to be compared against broader distribution
    # mean knn distance, local density, cell size
    feat_order = ['mean_k_neighbor_distance', 'local_density', 'Size.Area']

    def worker_fn(cell_ids):
        feats = dict(calculate_signal_average(cell_density_df, cell_ids))
        feats.update(dict(calculate_signal_average(cell_morph_df[["CELL_ID", "Size.Area"]], cell_ids)))
        return [feats[k] for k in feat_order]
    reference_feats = np.array([worker_fn(cids) for cids in cell_id_reference_subsets])
    target_feats = np.array([worker_fn(cids) for cids in cell_id_target_subsets])

    for feat_i, feat in enumerate(feat_order):
        # Calculate z-scores and percentiles for following features
        reference_vals = [val for val in reference_feats[:, feat_i] if val == val]
        zs = batch_zscores(reference_vals, target_feats[:, feat_i])
        pers = batch_percentiles(reference_vals, target_feats[:, feat_i])
        assert zs.shape[1] == pers.shape[1] == 1
        for target_j in range(len(cell_id_target_subsets)):
            if zs[target_j, 0] == zs[target_j, 0]:
                target_summaries[target_j][feat] = {
                    "val": target_feats[target_j, feat_i],
                    "z": float(zs[target_j, 0]),
                    "percentile": float(pers[target_j, 0])}

    # Following features can be directly interpreted on target subsets
    for target_j, cell_ids in enumerate(cell_id_target_subsets):
        sub_df = subset_feature_df(cell_morph_df, cell_ids)
        # Shape characteristics
        for feat in ['Shape.Circularity', 'Shape.Eccentricity', 'Shape.Solidity']:
            target_summaries[target_j][feat] = {"val": sub_df[feat].mean()}
        # Distribution of orientations
        target_summaries[target_j]['Orientation'] = calculate_orientation_metrics(cell_morph_df, cell_ids)

    if return_raw:
        return target_summaries
    else:
        return [seg_summary_to_text(ts, um_per_pixel=um_per_pixel, k=k, **kwargs) for ts in target_summaries]


# ---------------------------------------------------------------------------
# 2)  Updated description generator
# ---------------------------------------------------------------------------
def generate_cell_seg_description_for_target_patches_lung(
        cell_id_target_subsets,
        cell_id_reference_subsets,
        cell_seg_df,
        cell_morph_df,
        k: int = 3,
        radius: int = 70,
        um_per_pixel: float = 0.3775,
        return_raw: bool = False,
        **kwargs):
    """
    See original docstring – behaviour unchanged.
    Only difference: graceful handling of absent shape / orientation columns.
    """
    # Allow caller to pass a DataFrame that is *already* pre-processed,
    # but raise if the minimal required columns are still missing.
    must_haves = ['Size.Area']
    for col in must_haves:
        if col not in cell_morph_df.columns:
            raise ValueError(f"Required column '{col}' missing – "
                             "did you call preprocess_cell_morph_df()?")

    assert isinstance(cell_id_target_subsets, list)
    if cell_id_target_subsets and isinstance(cell_id_target_subsets[0], int):
        cell_id_target_subsets = [cell_id_target_subsets]

    target_summaries = [{} for _ in range(len(cell_id_target_subsets))]

    # ------------------------------------------------------------------ #
    # Density-related features
    # ------------------------------------------------------------------ #
    cell_density_df = calculate_densities(cell_seg_df, k=k, radius=radius)
    feat_order = ['mean_k_neighbor_distance', 'local_density', 'Size.Area']

    def _worker(cids):
        feats = dict(calculate_signal_average(cell_density_df, cids))
        feats.update(dict(calculate_signal_average(
            cell_morph_df[['CELL_ID', 'Size.Area']], cids)))
        return [feats.get(f, np.nan) for f in feat_order]

    reference_feats = np.array([_worker(cids) for cids in cell_id_reference_subsets])
    target_feats    = np.array([_worker(cids) for cids in cell_id_target_subsets])

    for feat_i, feat in enumerate(feat_order):
        reference_vals = [v for v in reference_feats[:, feat_i] if np.isfinite(v)]
        zs   = batch_zscores(reference_vals, target_feats[:, feat_i])
        pers = batch_percentiles(reference_vals, target_feats[:, feat_i])
        for j in range(len(cell_id_target_subsets)):
            if np.isfinite(zs[j, 0]):
                target_summaries[j][feat] = {
                    "val"       : target_feats[j, feat_i],
                    "z"         : float(zs[j, 0]),
                    "percentile": float(pers[j, 0])
                }

    # ------------------------------------------------------------------ #
    # Shape-related metrics (only add if available)
    # ------------------------------------------------------------------ #
    shape_feats = ['Shape.Circularity', 'Shape.Eccentricity', 'Shape.Solidity']
    for j, cids in enumerate(cell_id_target_subsets):
        sub_df = subset_feature_df(cell_morph_df, cids)

        for feat in shape_feats:
            if feat in sub_df.columns:
                target_summaries[j][feat] = {"val": sub_df[feat].mean()}

        # Orientation metrics (optional)
        if 'Orientation' in cell_morph_df.columns:
            target_summaries[j]['Orientation'] = \
                calculate_orientation_metrics(cell_morph_df, cids)

    # ------------------------------------------------------------------ #
    return (target_summaries
            if return_raw
            else [seg_summary_to_text(ts,
                                      um_per_pixel=um_per_pixel,
                                      k=k,
                                      **kwargs)
                  for ts in target_summaries])


def seg_summary_to_text(target_summary, um_per_pixel=0.3775, k=3):
    """ Convert cell segmentation summary dict to text description

    Args:
        target_summary (dict): dictionary containing cell segmentation summary
        um_per_pixel (float): microns per pixel
        k (int): number of nearest neighbors for distance calculation

    Returns:
        str: natural language summary of cell segmentation
    """
    summary = []

    summary.append("**Cell organizations**:")
    # Size (Area)
    if 'Size.Area' in target_summary:
        area = target_summary['Size.Area']
        mean_size = area['val'] * um_per_pixel**2
        line = f"The average cell size, measured by area, is **{mean_size:.0f} µm²**, " + \
            f"which is in the **{area['percentile']:.0f}th percentile** (z-score: {area['z']:.1f})."
        if area['z'] <= -1:
            line += " This indicates that these cells are smaller than the majority."
        elif area['z'] >= 1:
            line += " This indicates that these cells are larger than the majority."
        summary.append(line)

    # Mean k-neighbor distance
    if 'mean_k_neighbor_distance' in target_summary:
        distance = target_summary['mean_k_neighbor_distance']
        mean_dist = distance['val'] * um_per_pixel
        line = f"These cells' average distance to the {k} nearest neighbors is **{mean_dist:.0f} µm**, " + \
            f"which is in the **{distance['percentile']:.0f}th percentile** (z-score: {distance['z']:.1f})."
        if distance['z'] <= -1:
            line += " This indicates that intercellular distances are shorter than the majority."
        elif distance['z'] >= 1:
            line += " This indicates that intercellular distances are longer than the majority."
        summary.append(line)

    # Mean k-neighbor distance
    if 'local_density' in target_summary:
        density = target_summary['local_density']
        line = f"The local density of cells in the vicinity is in the **{density['percentile']:.0f}th percentile** " + \
            f"(z-score: {density['z']:.1f})."
        if density['z'] <= -1:
            line += " This indicates that these cells are more sparsely distributed."
        elif density['z'] >= 1:
            line += " This indicates that these cells are more densely packed."
        summary.append(line)

    summary.append("\n**Cell shapes**:")
    # Shape features
    line = ''
    if 'Shape.Circularity' in target_summary:
        circularity = target_summary['Shape.Circularity']['val']
        line += f"These cells have an average circularity of **{circularity:.2f}**."
        if circularity < 0.5:
            line += " This indicates a more elongated or irregular shape."

    if 'Shape.Eccentricity' in target_summary:
        eccentricity = target_summary['Shape.Eccentricity']['val']
        line += f"The average eccentricity is **{eccentricity:.2f}**."
        if eccentricity > 0.7:
            line += " This indicates highly elongated cells."
        elif eccentricity > 0.5:
            line += " This indicates elongated cells."

    if 'Shape.Solidity' in target_summary:
        solidity = target_summary['Shape.Solidity']['val']
        line += f"The average solidity is **{solidity:.2f}**."
        if solidity < 0.75:
            line += " This indicates irregular boundaries or concave shapes."
    summary.append(line)

    # Orientation
    if 'Orientation' in target_summary:
        circular_var = target_summary['Orientation']['circular_variance']
        line = f"The orientations of these cells have a circular variance of **{circular_var:.2f}**."
        if circular_var < 0.3:
            line += " This indicates uniform and aligned orientations."
        elif circular_var > 0.7:
            line += " This indicates a random distribution of orientations."
        summary.append(line)

    return "\n".join(summary)
