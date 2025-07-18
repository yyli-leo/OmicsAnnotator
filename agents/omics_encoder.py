import numpy as np

from agents.base_agent import BaseAgent
from patchsum.utils import get_reference_patch_list, get_cells_within_roi
from patchsum.cell_type_info import get_cell_type_df, generate_cell_type_description_for_target_patches
from patchsum.cell_seg_info import get_cell_segmentation_df, get_cell_morphology_df, generate_cell_seg_description_for_target_patches
from patchsum.processed_bm_info import get_cell_biomarker_expression_df, generate_cell_bm_description_for_target_patches
from patchsum.scgp_helpers import divide_large_connected_components
import scgp
import networkx as nx


# class OmicsEncoder:
#     def __init__(self, region_id, root_dir, biomarkers, cell_type_rename):
#         """
#         Parameters:
#             region_id: str
#             root_dir: str
#             biomarkers: list[str]
#             cell_type_rename: dict[str, str]
#         """
#         self.region_id = region_id
#         self.root_dir = root_dir
#         self.biomarkers = biomarkers
#         self.cell_type_rename = cell_type_rename
#
#     def run(self, xyranges):
#         # Load data
#         cell_seg_df = get_cell_segmentation_df(self.region_id, self.root_dir)
#         cell_morph_df = get_cell_morphology_df(self.region_id, self.root_dir)
#         cell_types_df, _ = get_cell_type_df(self.region_id, self.root_dir)
#         cell_types_df["CELL_TYPE"] = [
#             self.cell_type_rename[ct] for ct in cell_types_df["CELL_TYPE"]
#         ]
#         cell_bm_df = get_cell_biomarker_expression_df(
#             self.region_id, self.root_dir, channels=self.biomarkers
#         )
#
#         # Sample reference patches
#         cell_id_reference_subsets = get_reference_patch_list(
#             cell_seg_df, patch_sizes=[128], n_cells_min=5,
#             n_cells_max=50, count=1000, seed=123
#         )
#
#         # Use the provided XY ranges to define ROIs
#         cell_id_target_subsets = [
#             get_cells_within_roi(xyr, cell_seg_df) for xyr in xyranges
#         ]
#
#         # Generate summaries
#         bm_summaries = generate_cell_bm_description_for_target_patches(
#             cell_id_target_subsets, cell_id_reference_subsets,
#             cell_bm_df, channels=self.biomarkers,
#             return_raw=False, percentile_threshold=90
#         )
#
#         ct_summaries = generate_cell_type_description_for_target_patches(
#             cell_id_target_subsets, cell_types_df, cell_bm_df,
#             n_key_cell_types=3, return_raw=False
#         )
#
#         seg_summaries = generate_cell_seg_description_for_target_patches(
#             cell_id_target_subsets, cell_id_reference_subsets,
#             cell_seg_df, cell_morph_df,
#             um_per_pixel=0.3775, k=3, return_raw=False
#         )
#
#         return {
#             "biomarker_summary": bm_summaries,
#             "cell_type_summary": ct_summaries,
#             "segmentation_summary": seg_summaries,
#         }
#
#     def run_scgp(self, region_objs, scgp_single_region_outputs):
#         cell_seg_df = get_cell_segmentation_df(self.region_id, self.root_dir)
#         cell_morph_df = get_cell_morphology_df(self.region_id, self.root_dir)
#         cell_types_df, _ = get_cell_type_df(self.region_id, self.root_dir)
#         cell_types_df["CELL_TYPE"] = [self.cell_type_rename[ct] for ct in cell_types_df["CELL_TYPE"]]
#         cell_bm_df = get_cell_biomarker_expression_df(self.region_id, self.root_dir, channels=self.biomarkers)
#
#         cell_id_reference_subsets = get_reference_patch_list(
#             cell_seg_df, patch_sizes=[128], n_cells_min=5, n_cells_max=50, count=1e3, seed=123)
#
#         obj, features, model = region_objs[self.region_id]
#         neighbor_graph = scgp.neighborhood.construct_graph(features[1][['spatial']])
#
#         region_annotation = {k: v for k, v in scgp_single_region_outputs.items() if k[0] == self.region_id}
#         region_cluster_labels = [
#             cl for cl in set(region_annotation.values()) if cl >= 0 and list(region_annotation.values()).count(cl) > 0.01 * len(region_annotation)]
#
#         cluster_identifiers = []
#         selected_cell_subsets = []
#         for cl in region_cluster_labels:
#             subgs = neighbor_graph.subgraph([k for k, v in region_annotation.items() if v == cl])
#             subgs = [subg for subg in nx.connected_components(subgs) if len(subg) >= 20]
#             np.random.shuffle(subgs)
#             for subg in subgs[:5]:
#                 outputs = divide_large_connected_components(neighbor_graph, subg, max_size=100)
#                 for output in outputs:
#                     if len(output) >= 20:
#                         cids = sorted([int(c[1]) for c in output])
#                         selected_cell_subsets.append(cids)
#                         cluster_identifiers.append((self.region_id, tuple(cids), cl))
#
#         bm_summaries = generate_cell_bm_description_for_target_patches(
#             selected_cell_subsets, cell_id_reference_subsets,
#             cell_bm_df, channels=self.biomarkers, return_raw=False, percentile_threshold=90)
#
#         ct_summaries = generate_cell_type_description_for_target_patches(
#             selected_cell_subsets,
#             cell_types_df, cell_bm_df, n_key_cell_types=3, return_raw=False)
#
#         seg_summaries = generate_cell_seg_description_for_target_patches(
#             selected_cell_subsets, cell_id_reference_subsets,
#             cell_seg_df, cell_morph_df, um_per_pixel=0.3775, k=3, return_raw=False)
#
#         return {
#             "biomarker_summary": bm_summaries,
#             "cell_type_summary": ct_summaries,
#             "segmentation_summary": seg_summaries,
#         }, cluster_identifiers


class OmicsEncoder:
    # --------------------------------------------------
    # 1. Arbitrary XY ranges
    # --------------------------------------------------
    def run(self, *, region_id, root_dir, biomarkers, cell_type_rename, xyranges):
        # Load data
        cell_seg_df   = get_cell_segmentation_df(region_id, root_dir)
        cell_morph_df = get_cell_morphology_df(region_id, root_dir)
        cell_types_df, _ = get_cell_type_df(region_id, root_dir)
        cell_types_df["CELL_TYPE"] = [cell_type_rename[c] for c in cell_types_df["CELL_TYPE"]]
        cell_bm_df = get_cell_biomarker_expression_df(region_id, root_dir, channels=biomarkers)

        # reference patches
        refs = get_reference_patch_list(cell_seg_df, [128], 5, 50, 1000, 123)

        # targets from xyranges
        targets = [get_cells_within_roi(xyr, cell_seg_df) for xyr in xyranges]

        bm_sum = generate_cell_bm_description_for_target_patches(targets, refs, cell_bm_df,
                                                                 channels=biomarkers, percentile_threshold=90, return_raw=False)
        ct_sum = generate_cell_type_description_for_target_patches(targets, cell_types_df, cell_bm_df,
                                                                   n_key_cell_types=3, return_raw=False)
        seg_sum = generate_cell_seg_description_for_target_patches(targets, refs, cell_seg_df, cell_morph_df,
                                                                   um_per_pixel=0.3775, k=3, return_raw=False)
        return {
            "biomarker_summary": bm_sum,
            "cell_type_summary": ct_sum,
            "segmentation_summary": seg_sum,
        }

    # --------------------------------------------------
    # 2. SCGP cluster based sampling
    # --------------------------------------------------
    def run_scgp(self, *, region_id, root_dir, biomarkers, cell_type_rename,
                 region_objs, scgp_single_region_outputs):
        cell_seg_df   = get_cell_segmentation_df(region_id, root_dir)
        cell_morph_df = get_cell_morphology_df(region_id, root_dir)
        cell_types_df, _ = get_cell_type_df(region_id, root_dir)
        cell_types_df["CELL_TYPE"] = [cell_type_rename[c] for c in cell_types_df["CELL_TYPE"]]
        cell_bm_df = get_cell_biomarker_expression_df(region_id, root_dir, channels=biomarkers)

        refs = get_reference_patch_list(cell_seg_df, [128], 5, 50, 1000, 123)

        obj, features, _ = region_objs[region_id]
        graph = scgp.neighborhood.construct_graph(features[1][["spatial"]])
        region_anno = {k:v for k,v in scgp_single_region_outputs.items() if k[0]==region_id}
        majors = [cl for cl in set(region_anno.values()) if cl>=0 and list(region_anno.values()).count(cl)>0.01*len(region_anno)]

        subsets, identifiers = [], []
        rng = np.random.default_rng(0)
        for cl in majors:
            subg = graph.subgraph([k for k,v in region_anno.items() if v==cl])
            comps = [c for c in nx.connected_components(subg) if len(c)>=20]
            rng.shuffle(comps)
            for comp in comps[:5]:
                parts = divide_large_connected_components(graph, comp, 100)
                for part in parts:
                    if len(part)>=20:
                        cid_list = sorted(int(c[1]) for c in part)
                        subsets.append(cid_list)
                        identifiers.append((region_id, tuple(cid_list), cl))

        bm_sum = generate_cell_bm_description_for_target_patches(subsets, refs, cell_bm_df,
                                                                 channels=biomarkers, percentile_threshold=90, return_raw=False)
        ct_sum = generate_cell_type_description_for_target_patches(subsets, cell_types_df, cell_bm_df,
                                                                   n_key_cell_types=3, return_raw=False)
        seg_sum = generate_cell_seg_description_for_target_patches(subsets, refs, cell_seg_df, cell_morph_df,
                                                                   um_per_pixel=0.3775, k=3, return_raw=False)
        return {
            "biomarker_summary": bm_sum,
            "cell_type_summary": ct_sum,
            "segmentation_summary": seg_sum,
        }, identifiers
