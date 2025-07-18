import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from collections import Counter

from .utils import get_sliding_window_reference_ROIs, get_ROI_image_feature
from .single_channel_featurize import SingleChannelHistogram


def mosaic_wrapper(region_id,
                   cell_seg_df,
                   cell_bm_df=None,
                   cell_types_df=None,
                   cell_morph_df=None,
                   image=None,
                   image_featurizer=SingleChannelHistogram(),
                   feat_weights={'feat_image': 1.0, 'feat_cell_bm': 1.0,
                                 'feat_cell_types': 1.0, 'feat_cell_morph': 0.2},
                   ROI_size=128,
                   stride_ratio=0.5,
                   n_cells_min=5,
                   n_cells_max=50,
                   num_feat_clusters=10,
                   num_coord_subclusters=2,
                   num_samples_per_cluster=2,
                   cluster_center_method='sample',
                   temp=1.0,
                   seed=123):

    # Format feature dataframes
    if cell_bm_df is not None and 'CELL_ID' in cell_bm_df.columns:
        cell_bm_df = cell_bm_df.set_index('CELL_ID')
    if cell_types_df is not None and 'CELL_ID' in cell_types_df.columns:
        cell_types_df = cell_types_df.set_index('CELL_ID')
        unique_cell_types = sorted(set(cell_types_df['CELL_TYPE']))
    if cell_morph_df is not None and 'CELL_ID' in cell_morph_df.columns:
        cell_morph_df = cell_morph_df.set_index('CELL_ID')
    cell_bm_df = cell_bm_df / cell_bm_df.std(0)
    cell_morph_df = cell_morph_df / cell_morph_df.std(0)

    # Get all ROI exhaustively
    all_ROIs = get_sliding_window_reference_ROIs(
        cell_seg_df, patch_size=ROI_size, n_cells_min=n_cells_min,
        n_cells_max=n_cells_max, stride_ratio=stride_ratio, num_repetition=1, seed=seed)

    # Calculate multiple features
    ROI_feats = {k: {} for k in all_ROIs}
    for k, ROI in all_ROIs.items():
        ROI_feats[k]['coord'] = np.array([ROI['center_x'], ROI['center_y']])
        if image is not None:
            # Raw image feature
            ROI_feats[k]['feat_image'] = get_ROI_image_feature(ROI, image, featurizer=image_featurizer)
        if cell_bm_df is not None:
            ROI_feats[k]['feat_cell_bm'] = np.array(cell_bm_df.loc[ROI['cell_ids']].mean(0))
        if cell_types_df is not None:
            composition = Counter(cell_types_df.loc[ROI['cell_ids'], 'CELL_TYPE'])
            count_vec = np.array([composition.get(ct, 0) for ct in unique_cell_types])
            ROI_feats[k]['feat_cell_types'] = count_vec / count_vec.sum()
        if cell_morph_df is not None:
            ROI_feats[k]['feat_cell_morph'] = np.array(cell_morph_df.loc[ROI['cell_ids']].mean(0))

    # Calculate combined feature
    ROI_names = list(ROI_feats.keys())
    combined_features = []
    for f in ROI_feats[ROI_names[0]]:
        if f.startswith('feat_'):
            fw = feat_weights.get(f, 1.0)
            feat_matrix = np.stack([ROI_feats[roi][f] for roi in ROI_names], 0)
            dist_mat = cdist(feat_matrix, feat_matrix, metric='Euclidean')
            fw = fw / np.quantile(np.triu(dist_mat), 0.95)
            combined_features.append(fw * feat_matrix)
    combined_features = np.concatenate(combined_features, 1)

    # Cluster features
    clustering = KMeans(n_clusters=num_feat_clusters, random_state=seed).fit(combined_features)
    feat_clusters = clustering.labels_

    # Subcluster based on spatial coordinates
    final_clusters = [None] * len(feat_clusters)
    for cl in set(feat_clusters):
        cl_inds = np.array([i for i, _c in enumerate(feat_clusters) if _c == cl])
        coords = np.array([ROI_feats[ROI_names[i]]['coord'] for i in cl_inds])
        clustering = KMeans(n_clusters=num_coord_subclusters, random_state=seed).fit(coords)
        coord_subclusters = clustering.labels_
        for ind, sub_cl in zip(cl_inds, coord_subclusters):
            final_clusters[ind] = (feat_clusters[ind], sub_cl)
    assert len(ROI_names) == len(final_clusters) == combined_features.shape[0]

    # Pick representative ROIs
    cluster_centers = pick_cluster_center(
        combined_features, final_clusters, n_samples=num_samples_per_cluster,
        method=cluster_center_method, temp=temp, seed=seed)
    cluster_centers = {cl: [all_ROIs[ROI_names[i]] for i in cl_centers] for cl, cl_centers in cluster_centers.items()}
    return [all_ROIs[roi] for roi in ROI_names], combined_features, final_clusters, cluster_centers


def pick_cluster_center(features, clusters, n_samples=1, method='center', temp=1.0, seed=123):
    if seed is not None:
        np.random.seed(seed)
    unique_c = set(clusters)
    cluster_centers = {}
    for c in unique_c:
        inds = [i for i, _c in enumerate(clusters) if _c == c]
        if len(inds) < max(4, 2 * n_samples):
            print("Skip cluster %s" % str(c))
            continue
        median_cluster_feature = np.median(np.stack([features[i] for i in inds], 0), 0)
        dists = {i: np.linalg.norm(features[i] - median_cluster_feature, ord=2) for i in inds}
        if method == 'center':
            cluster_centers[c] = sorted(dists.keys(), key=lambda x: dists[x])[:n_samples]
        elif method == 'sample':
            prob_vec = 1 / (np.array([dists[i] for i in inds]) + 1e-5)
            prob_vec = np.clip(prob_vec - np.quantile(prob_vec, 0.5), 0, np.max(prob_vec))
            prob_vec = (prob_vec ** (2 / temp)) / (prob_vec ** (2 / temp)).sum()
            _n_samples = min(n_samples, np.nonzero(prob_vec)[0].shape[0])
            ccs = np.random.choice(inds, (_n_samples,), replace=False, p=prob_vec)
            ccs = sorted(ccs, key=lambda x: dists[x])
            cluster_centers[c] = ccs
    return cluster_centers


# def retrieve_all_from_multiple_pivot(pivot_ROIs, all_ROIs, final_clusters):
#     pivot_ROI_names = [ROI['patch_name'] for ROI in pivot_ROIs]
#     pivot_ind = [i for i, ROI in enumerate(all_ROIs) if ROI['patch_name'] in pivot_ROI_names]
#     pivot_cls = set([final_clusters[i][0] for i in pivot_ind])

#     same_cluster_ROIs = [all_ROIs[i] for i, _cl in enumerate(final_clusters) if _cl[0] in pivot_cls]

#     pivot_ROI_feats = np.array([get_ROI_image_feature(ROI, normalized_img, featurizer) for ROI in pivot_ROIs])
#     same_cluster_ROI_feats = np.array(
#         [get_ROI_image_feature(ROI, normalized_img, featurizer) for ROI in same_cluster_ROIs])
#     dists = np.linalg.norm(np.expand_dims(same_cluster_ROI_feats, 1) - np.expand_dims(pivot_ROI_feats, 0),
#                            axis=2, ord=2)

#     min_dists = [sorted(d)[1] for d in dists]
#     lower_thr = np.quantile(min_dists, 0.1)
#     upper_thr = np.quantile(min_dists, 0.8)
#     retrieved_ROIs = [ROI for ROI, d in zip(same_cluster_ROIs, min_dists) if d < upper_thr]
#     min_dists = [np.clip((d - lower_thr)/(upper_thr - lower_thr), 0., 1.) for d in min_dists if d < upper_thr]
#     retrieved_ROI_colors = [matplotlib.cm.viridis(1 - d) for d in min_dists]
#     return retrieved_ROIs, retrieved_ROI_colors
