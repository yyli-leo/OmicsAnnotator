import os
import numpy as np
import pandas as pd

from .utils import calculate_signal_average, batch_percentiles, batch_zscores


def get_cell_biomarker_expression_df(region_id, root_dir, channels=[], **kwargs):
    """ Read cell biomarker expression table for a region from disk """
    table_file_path = os.path.join(root_dir, region_id, f'{region_id}.expression.csv')
    assert os.path.exists(table_file_path), f'Cell biomarker expression table not found: {table_file_path}'
    cell_bm_df = pd.read_csv(table_file_path)
    cell_bm_df = cell_bm_df.set_index('CELL_ID')
    if len(channels) > 0:
        cell_bm_df = cell_bm_df[channels]

    cell_bm_df = np.arcsinh(cell_bm_df / (5 * np.quantile(cell_bm_df, 0.2, axis=0) + 1e-5))
    cell_bm_df = cell_bm_df / cell_bm_df.std(0)
    cell_bm_df = cell_bm_df.reset_index()
    return cell_bm_df


def generate_cell_bm_description_for_target_patches(cell_id_target_subsets,
                                                    cell_id_reference_subsets,
                                                    cell_bm_df,
                                                    channels=[],
                                                    return_raw=False,
                                                    **kwargs):
    """ Generate description of biomarker expression for target subsets

    Args:
        cell_id_target_subsets (list): list of lists, each individual list containing cell ids within the ROI
        cell_id_reference_subsets (list): list of lists, each individual list containing cell ids within a reference ROI
        cell_bm_df (pd.DataFrame): cell biomarker expression table for the full region
        channels (list, optional): list of channel names to include in the summary, use an empty list to include all
        return_raw (bool, optional): if to return raw z-scores or text description
        **kwargs: additional arguments for text description

    Returns:
        np.ndarray or list: raw z-scores or text description
    """
    assert isinstance(cell_id_target_subsets, list)
    if isinstance(cell_id_target_subsets[0], int):
        cell_id_target_subsets = [cell_id_target_subsets]

    # Specify channel/biomarkers
    if len(channels) == 0:
        channels = [bm for bm in cell_bm_df.columns if bm != 'CELL_ID']
    else:
        channels = sorted(set(channels) & set(cell_bm_df.columns))

    def worker_fn(cell_ids):
        signal_average = calculate_signal_average(cell_bm_df, cell_ids)
        ar = [signal_average[bm] for bm in channels]
        return ar

    reference_feats = np.array([worker_fn(cids) for cids in cell_id_reference_subsets])
    target_feats = np.array([worker_fn(cids) for cids in cell_id_target_subsets])

    target_summaries = [{} for _ in range(len(cell_id_target_subsets))]
    for bm_i, bm in enumerate(channels):
        reference_vals = [val for val in reference_feats[:, bm_i] if val == val]

        zs = batch_zscores(reference_vals, target_feats[:, bm_i])
        pers = batch_percentiles(reference_vals, target_feats[:, bm_i])
        assert zs.shape[1] == pers.shape[1] == 1

        for target_j in range(len(cell_id_target_subsets)):
            if zs[target_j, 0] == zs[target_j, 0]:
                target_summaries[target_j][bm] = {
                    "z": float(zs[target_j, 0]), "percentile": float(pers[target_j, 0])}
    if return_raw:
        return target_summaries
    else:
        return [bm_summary_to_text(ts, **kwargs) for ts in target_summaries]


def bm_summary_to_text(target_summary,
                       z_threshold=1.5,
                       percentile_threshold=95,
                       always_include_topk=2,
                       include_high=True,
                       include_low=True,
                       include_others=True):
    """ Convert biomarker summary dict to text description

    Args:
        target_summary (dict): dictionary containing biomarker statistics
        z_threshold (float): z-score threshold for categorization
        percentile_threshold (float): percentile threshold for categorization
        always_include_topk (int): number of top expressed biomarkers to be included by default
        include_high (bool): whether to include high expression in the summary
        include_low (bool): whether to include low expression in the summary
        include_others (bool): whether to include moderate expression in the summary

    Returns:
        str: natural language summary of biomarker expression
    """
    # Grouping variables
    highly_expressed = []
    lowly_expressed = []
    others = []

    # Process each biomarker
    for bm_i, biomarker in enumerate(sorted(target_summary.keys(), key=lambda x: -target_summary[x]['z'])):
        stats = target_summary[biomarker]
        z_score = stats['z']
        percentile = stats['percentile']
        if z_score > z_threshold or percentile > percentile_threshold or bm_i < always_include_topk:
            highly_expressed.append((biomarker, z_score, percentile))
        elif z_score < -z_threshold or percentile < 100 - percentile_threshold:
            lowly_expressed.append((biomarker, z_score, percentile))
        else:
            others.append((biomarker, z_score, percentile))

    # Generate natural language summaries
    summary = []
    if highly_expressed and include_high:
        summary.append("**Highly Expressed Biomarkers**:\nThese biomarkers are overexpressed:")
        for biomarker, z, p in sorted(highly_expressed, key=lambda x: x[1], reverse=True):
            summary.append(f"- **{biomarker}**: z-score = {z:.1f}, percentile = {p:.0f}%")

    if lowly_expressed and include_low:
        summary.append("\n**Lowly Expressed Biomarkers**:\nThese biomarkers are underexpressed:")
        for biomarker, z, p in sorted(lowly_expressed, key=lambda x: x[1]):
            summary.append(f"- **{biomarker}**: z-score = {z:.1f}, percentile = {p:.0f}%")

    if others and include_others:
        summary.append("\n**Moderately Expressed Biomarkers**:\n" +
                       "These biomarkers do not show strong trends and have moderate expression levels:")
        summary.append(", ".join([f"**{biomarker}**" for biomarker, z, p in others]))
    return "\n".join(summary)


# def generate_continuous_feature_description_for_target_patches(xyranges,
#                                                                reference_patches,
#                                                                cell_seg_df,
#                                                                continous_signal_df,
#                                                                feature_names=[],
#                                                                value_type='strength'):
#     """ Generate tabular description of generic continuous features

#     Args:
#         xyranges (list): list of tuples (x_left, x_right, y_bottom, y_top) for
#             target ROIs
#         reference_patches (list): list of tuples (x_left, x_right, y_bottom, y_top)
#             for reference ROIs
#         cell_seg_df (pd.DataFrame): cell segmentation table for the full region
#         continuous_signal_df (pd.DataFrame): table of generic continuous features for the full region
#         feature_names (list, optional): list of feature names
#         value_type (str, optional): 'strength' or 'prob'

#     Returns:
#         np.ndarray or list: raw z-scores or text description
#     """
#     if isinstance(xyranges, tuple) and len(xyranges) == 4 and all(isinstance(i, int) for i in xyranges):
#         # Single patch
#         xyranges = [xyranges]

#     feature_cols = [col for col in continous_signal_df.columns if col != 'CELL_ID']
#     feature_names = feature_cols if len(feature_names) == 0 else feature_names
#     assert len(feature_names) == len(feature_cols)

#     def worker_fn(xyr):
#         cell_signal_average = calculate_roi_cell_signal_average(xyr, continous_signal_df, cell_seg_df)
#         ar = [cell_signal_average[col] for col in feature_cols]
#         return ar
#     query_cell_features = np.array([worker_fn(xyr) for xyr in xyranges])

#     if value_type == 'strength':
#         reference_cell_features = np.array([worker_fn(xyr) for xyr in reference_patches])
#         zscores = batch_zscores(reference_cell_features, query_cell_features)
#         percentiles = batch_percentiles(reference_cell_features, query_cell_features)

#         text_descs = []
#         for i in range(len(xyranges)):
#             text_desc = '|Feature Name|Z-Score|Percentile|Description|\n| --- | --- | --- | --- |\n'
#             for j in range(len(feature_cols)):
#                 z = zscores[i, j]
#                 z_desc = simple_z_score_description(z)
#                 percentile = percentiles[i, j]
#                 text_desc += '|{}|{:.1f}|{:.0f}|{}|\n'.format(
#                     feature_names[j], z, percentile, z_desc)
#             text_descs.append(text_desc)
#     elif value_type == 'prob':
#         text_descs = []
#         for i in range(len(xyranges)):
#             text_desc = '|Feature Name|Value|Probability Level|\n| --- | --- | --- |\n'
#             for j in range(len(feature_cols)):
#                 prob = query_cell_features[i, j]
#                 prob_desc = simple_prob_description(prob)
#                 text_desc += '|{}|{:.1f}|{}|\n'.format(feature_names[j], prob, prob_desc)
#             text_descs.append(text_desc)
#     return text_descs
