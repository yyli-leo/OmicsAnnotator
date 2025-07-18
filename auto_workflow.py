"""Automated script to construct or resume DKD memory artifacts.

Usage (bash):
-------------
python build_memory_pipeline.py \
    --region_ids s255 s314 \
    --meta_path data/DKD/s255/metadata.yaml \
    --norm_params data/NormParams.pkl \
    --mem_dir memory/DKD/s255_run1 \
    --resume

Key features
------------
* Command‑line arguments control root dir, region list, output dir, resume, etc.
* tqdm progress bars + logging to both stdout & file.
* Check‑pointing after every step (curator, scgp objects, text inference, aggregation, scoring).
* If a crash occurs, you can add --resume to automatically continue from the last checkpoint.
"""

import sys, argparse, pickle, logging, yaml
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
from agents.context_curator import ContextCurator
from agents.omics_encoder import OmicsEncoder
from agents.omics_annotator import OmicsAnnotator
from agents.omics_annotator_aggregate import OmicsAnnotatorAggregate
from agents.pathology_scorer import PathologyScorer
from patchsum.scgp_helpers import (
    construct_region_objs, construct_scgp_single_region_outputs,
)

############################################################
# 1. CLI
############################################################

def get_args():
    ap = argparse.ArgumentParser("Build DKD memory pipeline (auto‑restart)")
    ap.add_argument("--region_ids", nargs="*", default=["s255"], help="Region IDs to process")
    ap.add_argument("--meta_path", required=True, help="Path to metadata.yaml")
    ap.add_argument("--norm_params", required=True, help="Pickle with normalization params")
    ap.add_argument("--mem_dir", default="memory/run", help="Output directory to store checkpoints & logs")
    ap.add_argument("--resume", action="store_true", help="Resume from last checkpoint if exists")
    return ap.parse_args()
# def get_args():
#     ap = argparse.ArgumentParser("Build DKD memory pipeline (auto‑restart)")
#     ap.add_argument("--region_ids", nargs="*", default=["s255"], help="Region IDs to process")
#     ap.add_argument("--meta_path", default="data/DKD/s255/metadata.yaml", help="Path to metadata.yaml")
#     # ap.add_argument("--norm_params", required=True, help="Pickle with normalization params")
#     ap.add_argument("--mem_dir", default="memory/DKD/s255/workflow_test", help="Output directory to store checkpoints & logs")
#     ap.add_argument("--resume", action="store_true", help="Resume from last checkpoint if exists")
#     return ap.parse_args()

############################################################
# 2. helpers
############################################################

def setup_logger(mem_dir):
    log_path = Path(mem_dir) / "build_memory.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, mode="a")],
    )
    logging.info("Logging to %s", log_path)


def save_ckpt(obj, name, mem_dir):
    with open(Path(mem_dir) / f"{name}.pkl", "wb") as f:
        pickle.dump(obj, f)
    logging.info("Checkpoint saved: %s", name)


def load_ckpt(name, mem_dir):
    p = Path(mem_dir) / f"{name}.pkl"
    if p.exists():
        logging.info("Resuming from checkpoint: %s", name)
        return pickle.load(open(p, "rb"))
    return None

############################################################
# 3. main pipeline
############################################################

def main():
    args = get_args()
    Path(args.mem_dir).mkdir(parents=True, exist_ok=True)
    setup_logger(args.mem_dir)

    # load meta
    meta = yaml.safe_load(open(args.meta_path, "r"))
    root_dir = meta["root_dir"]
    tissue_types = meta["tissue_types"]
    major_phenotypes = meta["major_phenotypes"]
    biomarkers = meta["biomarkers"]
    selected_acq_ids = [rid for rid in meta["selected_acq_ids"] if rid.split("_")[0] in args.region_ids]
    cell_type_rename = meta["cell_type_rename"]
    background = meta["background"]

    selected_acq_ids = selected_acq_ids[:1]  # Test only
    logging.info("Test mode.")
    # ---------- Step 1: curator context ----------
    context = load_ckpt("context", args.mem_dir)
    if context is None or not args.resume:
        curator = ContextCurator()
        dialogue = curator.run({
            "tissue_types": ", ".join(tissue_types).lower(),
            "major_phenotypes": major_phenotypes,
            "biomarker_list": biomarkers,
            "background": background,
        })
        context = "---\n" + "\n---\n".join([m.type.upper() + ": " + m.content for m in dialogue])
        save_ckpt(context, "context", args.mem_dir)
    logging.info("Context ready.")

    # ---------- Step 2: region objs & SCGP ----------
    region_objs = load_ckpt("region_objs", args.mem_dir)
    if region_objs is None or not args.resume:
        region_objs = construct_region_objs(root_dir, selected_acq_ids, biomarkers, cell_type_rename)
        save_ckpt(region_objs, "region_objs", args.mem_dir)

    scgp_outputs = load_ckpt("scgp_outputs", args.mem_dir)
    if scgp_outputs is None or not args.resume:
        scgp_outputs = construct_scgp_single_region_outputs(region_objs, selected_acq_ids)
        save_ckpt(scgp_outputs, "scgp_outputs", args.mem_dir)

    # ---------- Step 3: encode each region ----------
    subset_vars = load_ckpt("subset_vars", args.mem_dir) or {}

    for region_id in tqdm(selected_acq_ids, desc="Encode regions"):
        if any(k[0] == region_id for k in subset_vars):
            continue  # already done
        encoder = OmicsEncoder(region_id, root_dir, biomarkers, cell_type_rename)
        summaries, cluster_ids = encoder.run_scgp(region_objs, scgp_outputs)
        for i, identifier in enumerate(cluster_ids):
            subset_vars[identifier] = {
                "tissue_types": ", ".join(tissue_types).lower(),
                "major_phenotypes": major_phenotypes,
                "context": context,
                "bm_summary": summaries["biomarker_summary"][i],
                "ct_summary": summaries["cell_type_summary"][i],
                "seg_summary": summaries["segmentation_summary"][i],
            }
        save_ckpt(subset_vars, "subset_vars", args.mem_dir)

    # ---------- Step 4: per-ROI text inference ----------
    annotator = OmicsAnnotator()
    subset_inputs = load_ckpt("subset_inputs", args.mem_dir) or {}
    subset_outputs = load_ckpt("subset_outputs", args.mem_dir) or {}

    for ident, ctx in tqdm(list(subset_vars.items()), desc="LLM per ROI"):
        if ident in subset_outputs and args.resume:
            continue
        dialogue = annotator.run(ctx)
        subset_inputs[ident] = dialogue[2].content
        subset_outputs[ident] = dialogue[4].content
        if len(subset_outputs) % 20 == 0:
            save_ckpt({"subset_inputs": subset_inputs, "subset_outputs": subset_outputs}, "subset_io", args.mem_dir)
    save_ckpt({"subset_inputs": subset_inputs, "subset_outputs": subset_outputs}, "subset_io", args.mem_dir)

    # ---------- Step 5: aggregation per cluster ----------
    outputs_by_cl = load_ckpt("outputs_by_cl", args.mem_dir) or {}
    agg_agent = OmicsAnnotatorAggregate()
    tmp = {}
    for (reg, cids, cl), txt in subset_outputs.items():
        tmp.setdefault((reg, cl), []).append(txt)

    for (reg, cl), lst in tqdm(list(tmp.items()), desc="Aggregate clusters"):
        if (reg, cl) in outputs_by_cl and args.resume:
            continue
        ctx_vars = {
            "tissue_types": ", ".join(tissue_types).lower(),
            "major_phenotypes": major_phenotypes,
            "context": context,
            "n_obs": len(lst),
            "results": "\n===\n".join(lst)
        }
        agg_dialogue = agg_agent.run(ctx_vars)
        outputs_by_cl[(reg, cl)] = agg_dialogue[-1].content
        save_ckpt(outputs_by_cl, "outputs_by_cl", args.mem_dir)

    # ---------- Step 6: pathology scoring ----------
    scorer = PathologyScorer()
    dr_dialogues = load_ckpt("dr_dialogues", args.mem_dir) or {}
    for ident in tqdm(list(subset_outputs.keys()), desc="Pathology scoring"):
        if ident in dr_dialogues and args.resume:
            continue
        ctx = {
            "tissue_types": ", ".join(tissue_types).lower(),
            "major_phenotypes": major_phenotypes,
            "user_input": subset_inputs[ident],
            "ai_output": subset_outputs[ident],
            "diagnosis": "DKD",
            "phenotype_desc": context,  # reuse long context as placeholder
        }
        dial = scorer.run(ctx)
        dr_dialogues[ident] = dial
        if len(dr_dialogues) % 20 == 0:
            save_ckpt(dr_dialogues, "dr_dialogues", args.mem_dir)
    save_ckpt(dr_dialogues, "dr_dialogues", args.mem_dir)

    logging.info("All steps finished successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("Pipeline crashed: %s", str(e))
        sys.exit(1)
