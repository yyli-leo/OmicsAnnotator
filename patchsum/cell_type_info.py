import os
import pandas as pd
from collections import Counter


def get_cell_type_df(region_id, root_dir, **kwargs):
    """ Read cell type dataframe for a region from disk """
    cell_type_path = os.path.join(root_dir, region_id, f'{region_id}.cell_types.csv')
    assert os.path.exists(cell_type_path), f'Cell type file not found: {cell_type_path}'
    cell_type_df = pd.read_csv(cell_type_path)
    cell_type_df.columns = ['CELL_ID', 'VALUE', 'CELL_TYPE']
    cell_type_mapping = dict(zip(cell_type_df['VALUE'], cell_type_df['CELL_TYPE']))
    return cell_type_df, cell_type_mapping


def get_signature_biomarkers(cell_types_df, cell_bm_df):
    """ Get signature biomarkers for each cell type """
    cts = set(cell_types_df['CELL_TYPE'])
    bm_levels = {}
    # Calculate bm average by cell type
    biomarkers = [bm for bm in cell_bm_df.columns if bm != 'CELL_ID']
    for ct in cts:
        cell_ids = cell_types_df[cell_types_df['CELL_TYPE'] == ct]['CELL_ID']
        bms = dict(cell_bm_df[cell_bm_df['CELL_ID'].isin(cell_ids)][biomarkers].mean())
        bm_levels[ct] = bms

    # Re-normalize bm averages across cell types
    for bm in biomarkers:
        bm_level_by_ct = [bm_levels[ct][bm] for ct in cts]
        bm_min, bm_max = min(bm_level_by_ct), max(bm_level_by_ct)
        for ct in cts:
            bm_levels[ct][bm] = (bm_levels[ct][bm] - bm_min)/(bm_max - bm_min)

    # Pick signature biomarkers for each cell type
    signature_biomarkers = {}
    for ct in cts:
        signature_bms = []
        for bm, val in bm_levels[ct].items():
            if val > 0.8:
                signature_bms.append(bm)
        signature_biomarkers[ct] = signature_bms
    return signature_biomarkers


def calculate_cell_type_composition(cell_types_df, cell_ids):
    """ Calculate cell type composition given a list of cell IDs """
    sub_ct_df = cell_types_df[cell_types_df['CELL_ID'].isin(cell_ids)]
    ct_count = dict(Counter(sub_ct_df['CELL_TYPE']))
    comp = {k: v/len(cell_ids) for k, v in ct_count.items()}
    return comp


def generate_cell_type_description_for_target_patches(cell_id_target_subsets,
                                                      cell_types_df,
                                                      cell_bm_df,
                                                      n_key_cell_types=3,
                                                      return_raw=False):
    """ Generate tabular description of cell type composition

    Args:
        cell_id_target_subsets (list): list of lists, each individual list containing cell ids within the ROI
        cell_types_df (pd.DataFrame): cell type dataframe
        cell_bm_df (pd.DataFrame): cell biomarker expression dataframe
        n_key_cell_types (int): number of key cell types to include, 3 to 5
        return_raw (bool): if to return raw compositions or text description

    Returns:
        list: text description
    """
    assert isinstance(cell_id_target_subsets, list)
    if isinstance(cell_id_target_subsets[0], int):
        cell_id_target_subsets = [cell_id_target_subsets]

    signature_biomarkers = get_signature_biomarkers(cell_types_df, cell_bm_df)

    avg_comp = calculate_cell_type_composition(cell_types_df, list(cell_types_df['CELL_ID']))

    target_summaries = []
    for cell_ids in cell_id_target_subsets:
        roi_comp = calculate_cell_type_composition(cell_types_df, cell_ids)
        enrichment = {ct: roi_comp[ct] / (avg_comp[ct] + 1e-5) for ct in roi_comp}

        target_summary = {}
        for ct in roi_comp:
            ct_comp = roi_comp[ct] * 100
            enr = enrichment[ct]
            sig_bms = signature_biomarkers[ct]
            target_summary[ct] = (ct_comp, enr, sig_bms)
        target_summaries.append(target_summary)
    if return_raw:
        return target_summaries
    else:
        return [ct_composition_summary_to_text(ts, n_key_cell_types=n_key_cell_types)
                for ts in target_summaries]


def ct_composition_summary_to_text(target_summary, n_key_cell_types=3):
    """ Convert cell type composition summary dict to text description

    Args:
        target_summary (dict): dictionary containing cell type compositions and enrichment
        n_key_cell_types (int): number of key cell types to include, 3 to 5

    Returns:
        str: natural language summary of cell type compositions
    """
    cts = list(target_summary.keys())
    key_cts = sorted(cts, key=lambda x: target_summary[x][0], reverse=True)[:n_key_cell_types]
    key_cts = [ct for ct in key_cts if target_summary[ct][0] > 0.05]
    for ct in sorted(cts, key=lambda x: target_summary[x][1], reverse=True)[:n_key_cell_types]:
        if ct not in key_cts and target_summary[ct][1] > 1.5 and target_summary[ct][0] > 0.05:
            key_cts.append(ct)

    summary = ["**Major Cell Types**:"]
    for ct_i, ct in enumerate(key_cts):
        ct_comp, enr, sig_bms = target_summary[ct]
        line = f'{ct_i + 1}. **{ct}**:\n'
        line += f'Cells of type "{ct}" make up **{ct_comp:.0f}%** of the total composition. '
        line += f"They are enriched by **{enr:.1f}** compared to region average."
        if len(sig_bms) > 0:
            line += f"This cell type is characterized by the biomarker(s): {','.join([f'**{bm}**' for bm in sig_bms])}"
        summary.append(line)
        summary.append("")
    return "\n".join(summary)
