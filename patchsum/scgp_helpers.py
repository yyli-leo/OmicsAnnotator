import numpy as np
import scgp
from matplotlib import pyplot as plt

from patchsum.cell_seg_info import get_cell_segmentation_df, get_cell_morphology_df, generate_cell_seg_description_for_target_patches
from patchsum.cell_type_info import get_cell_type_df, generate_cell_type_description_for_target_patches
from patchsum.processed_bm_info import get_cell_biomarker_expression_df, generate_cell_bm_description_for_target_patches


def construct_region_objs(root_dir, selected_acq_ids, biomarkers, cell_type_rename):
    region_objs = {}
    for region_id in selected_acq_ids:
        cell_seg_df = get_cell_segmentation_df(region_id, root_dir)
        # cell_morph_df = get_cell_morphology_df(region_id, root_dir)
        cell_types_df, cell_types_mapping = get_cell_type_df(region_id, root_dir)
        cell_types_df['CELL_TYPE'] = [cell_type_rename[ct] for ct in cell_types_df['CELL_TYPE']]
        cell_bm_df = get_cell_biomarker_expression_df(region_id, root_dir, channels=biomarkers)

        obj = scgp.construct_object(
            region_id, cell_seg_df, cell_bm_df,
            index_col='CELL_ID', mode='anndata')
        scgp_partitions, (features, model) = scgp.SCGP_wrapper(
            obj, verbose=False, rp=1e-5, feature_knn=4, smooth_iter=1, smooth_level=1)
        region_objs[region_id] = (obj, features, model)

    return region_objs


def construct_scgp_single_region_outputs(region_objs, selected_acq_ids):
    scgp_single_region_outputs = {}
    for region_id in selected_acq_ids:
        obj, features, model = region_objs[region_id]
        for rp in np.arange(2e-4, 1e-3, 1e-4):
            scgp_partitions = scgp.SCGP_partition(features, model, rp=rp, smooth_iter=2, smooth_level=1, verbose=True)
            if len(filter_significant_classes([int(v) for v in scgp_partitions.values()], threshold=0.01)) > 6:
                break

        print(region_id, rp)
        scgp.plot_all_regions_with_annotations(scgp_partitions, obj, figsize=6)
        scgp_single_region_outputs.update(scgp_partitions)

    return scgp_single_region_outputs


### Helper functions ###
def get_local_neighborhood(region_objs, c, n_spatial_neighbor=1, n_feature_neighbor=0):
    region_id, cell_id = c
    obj, (_, neighbor_df), _ = region_objs[region_id]
    # Spatial neighbors
    subset = set([c])
    for hop in range(n_spatial_neighbor):
        _add_subset = []
        for n in subset:
            _add_subset.extend(neighbor_df.loc[n]['spatial'])
        subset = subset.union(set(_add_subset))
    # Feature neighbors
    subset2 = set([c])
    for hop in range(n_feature_neighbor):
        _add_subset = []
        for n in subset2:
            _add_subset.extend(neighbor_df.loc[n]['feature'])
        subset2 = subset2.union(set(_add_subset))

    subset = subset.union(subset2)
    return subset

# Function to parse the LLM response
def parse_llm_response(response):
    guesses = []
    # Split the response into lines
    lines = response.split("\n")
    flags = [False, False]
    for line_i, line in enumerate(lines):
        # Look for the numbered guesses
        if line.startswith("1.") and not flags[0]:
            guess = lines[line_i]
            for j in range(1, 3):
                if line_i + j < len(lines) and len(lines[line_i + j]) < 50 and 'evidence' not in lines[line_i + j].lower():
                    guess += '\n' + lines[line_i + j]
            guesses.append(guess)
            flags[0] = True

        elif line.startswith("2.") and not flags[1]:
            guess = lines[line_i]
            for j in range(1, 3):
                if line_i + j < len(lines) and len(lines[line_i + j]) < 50 and 'evidence' not in lines[line_i + j].lower():
                    guess += '\n' + lines[line_i + j]
            guesses.append(guess)
            flags[1] = True

    if len(guesses) == 0:
        # Start from the top if there are no numbered guesses
        line_i = 0
        guess = lines[line_i]
        for j in range(1, 5):
            if line_i + j < len(lines) and len(lines[line_i + j]) < 50 and 'evidence' not in lines[line_i + j].lower():
                guess += '\n' + lines[line_i + j]
        guesses.append(guess)
    return guesses


def parse_guess(guess):
    items = guess.replace('\n', ' ').replace('1.', '').replace('2.', '').replace(':', '').split('*')
    items = [item.strip() for item in items if len(item.strip()) > 5 and len(item.strip()) < 100]
    return items


def guess_to_label(guess):
    if 'glomerul' in guess.lower() or 'peritubular capillar' in guess.lower():
        return 'Glomeruli'
    if 'vasculature' in guess.lower() or 'blood vessel' in guess.lower():
        return 'Blood vessel'
    if 'distal' in guess.lower() and 'tub' in guess.lower():
        return 'Distal tubules'
    if 'prox' in guess.lower() and 'tub' in guess.lower():
        return 'Proximal tubules'
    if 'tubul' in guess.lower():
        return 'Proximal tubules'
    if 'interstiti' in guess.lower() or 'basement membrane' in guess.lower():
        return 'Interstitium'
    if 'inflammatory' in guess.lower() or 'immune' in guess.lower():
        return 'Interstitium'
    return 'NA'


label_to_class = {
    'Proximal tubules': 0,
    'Distal tubules': 1,
    'Glomeruli': 2,
    'Blood vessel': 3,
    'Interstitium': 4,
    'Other': -1,
    'NA': -1,
}


def evaluate_llm_responses(text_labels, text_inference_outputs, conf_mat=True):
    """ Evaluate the LLM responses against the true labels. """
    query_cells = list(text_labels.keys())
    y_true = [text_labels[q_cell] for q_cell in query_cells]
    y_pred = []

    # Parse each LLM response into discrete categories of structures in line with the labels
    guessed_entries_by_true_class = {y_t: [] for y_t in set(y_true)}
    for q_cell, y_t in zip(query_cells, y_true):
        guesses = parse_llm_response(text_inference_outputs[q_cell])
        guessed_entries = []
        if len(guesses) > 0:
            guessed_entries_by_true_class[y_t].extend(parse_guess(guesses[0]))
            for guess in guesses:
                guessed_entries.extend([guess_to_label(item) for item in parse_guess(guess)])
        guessed_entries = [item for item in guessed_entries if item != 'NA']
        y_pred.append(guessed_entries)

    if conf_mat:
        confusion_mat = np.zeros((5, 5))
    accuracy_at_k = []
    for y_t, y_p in zip(y_true, y_pred):
        # ### Interstitium could optionally be excluded ###
        # if y_t == 'Interstitium':
        #     continue

        # Accuracy at top-k guesses
        accuracy_at_k.append([y_t in y_p[:_k] for _k in range(1, len(y_p) + 1)])
        # Confusion matrix of the top-1 guess

        if conf_mat:
            for _y_p in y_p[:1]:
                if label_to_class[_y_p] >= 0:  # Exclude others
                    confusion_mat[label_to_class[y_t], label_to_class[_y_p]] += 1

    for _k in range(3):
        flags = [_acc[min(_k, len(_acc) - 1)] for _acc in accuracy_at_k]
        print(f"Accuracy at {_k+1}: {np.mean(flags):.2f}, {sum(flags)}/{len(flags)}")
    if conf_mat:
        plt.figure(figsize=(3, 3))
        plt.imshow(confusion_mat / np.sum(confusion_mat, axis=1, keepdims=True), cmap='Blues', vmin=0, vmax=0.8)
        plt.yticks(np.arange(5), ['Proximal tubules', 'Distal tubules', 'Glomeruli', 'Blood vessel', 'Interstitium'])
        plt.show()
    return y_true, y_pred, guessed_entries_by_true_class


def divide_large_connected_components(neighbor_graph, subg, max_size=100):
    """
    Divides connected components of a graph into smaller subgraphs if they exceed max_size.

    Parameters:
    - graph: The input graph (assumed to be a connected component of the original graph).
    - max_size: Maximum allowable size for a subgraph.

    Returns:
    - List of subgraphs, each with size <= max_size.
    """
    # If the graph size is already within the limit, return it as a single component
    if len(subg) <= max_size:
        return [subg]

    # Start dividing the graph into smaller subgraphs
    outputs = []
    n_subgraphs = np.ceil(len(subg) / (3 * max_size))

    all_nodes = sorted(subg)
    np.random.shuffle(all_nodes)

    for _ in range(int(n_subgraphs)):
        start_node = all_nodes[0]

        subgraph_nodes = set([start_node])
        queue = [start_node]

        for _ in range(5):  # 5 hops
            neighbors = sum([list(neighbor_graph.neighbors(n)) for n in queue], [])
            neighbors = [n for n in neighbors if n in all_nodes and n not in subgraph_nodes]
            subgraph_nodes.update(neighbors)
            if len(subgraph_nodes) >= max_size:
                break
            queue = neighbors

        # Create the subgraph and add it to the list
        outputs.append(subgraph_nodes)
        all_nodes = [node for node in all_nodes if node not in subgraph_nodes]

    return outputs


def filter_significant_classes(input_list, threshold=0.02):
    """
    Filters integer classes that have significant presence (>threshold percentage) in the input list.

    Args:
        input_list (list[int]): List of integers to analyze
        threshold (float): Minimum percentage threshold (default: 0.02 for 2%)

    Returns:
        list[int]: List of integer classes that meet the threshold, sorted by frequency
    """
    if not input_list:
        return []

    # Count occurrences of each integer
    from collections import Counter
    counts = Counter(input_list)
    total = len(input_list)

    # Filter classes that meet the threshold
    significant_classes = [
        num for num, count in counts.items()
        if (count / total) > threshold
    ]

    # Sort by frequency (descending)
    significant_classes.sort(key=lambda x: counts[x], reverse=True)

    return significant_classes