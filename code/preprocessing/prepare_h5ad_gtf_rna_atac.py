"""Prepare feature-aligned RNA-ATAC inputs from QC h5ad files."""
import argparse
import importlib.util
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

pd.options.future.infer_string = False

DEFAULT_BASE_PREPROCESSOR = Path(__file__).resolve().with_name("prepare_gxr_gtf_rna_atac.py")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rna-h5ad", required=True)
    p.add_argument("--atac-h5ad", required=True)
    p.add_argument("--gtf", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--base-preprocessor",
        default=str(DEFAULT_BASE_PREPROCESSOR),
        help="Companion preprocessor module; defaults to the copy beside this script.",
    )
    p.add_argument("--rna-counts-layer", default="counts")
    p.add_argument("--atac-counts-layer", default="counts")
    p.add_argument("--batch-key", default="batch")
    p.add_argument("--celltype-key", default="cell_type")
    p.add_argument("--dataset-key", default="dataset")
    p.add_argument("--top-genes", type=int, default=5000)
    p.add_argument("--target-sum", type=float, default=1e4)
    p.add_argument("--hvg-flavor", default="seurat")
    p.add_argument("--geneactivity-annotation", default="HAVANA")
    p.add_argument("--strip-atac-chr", action="store_true")
    return p.parse_args()

def load_base(path):
    spec = importlib.util.spec_from_file_location("anchorpt_base_preprocessor", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base preprocessor: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def require_sparse_counts(adata, layer, label):
    if layer not in adata.layers:
        raise KeyError(f"{label} is missing layer {layer!r}")
    matrix = adata.layers[layer]
    matrix = matrix.tocsr().astype(np.float32, copy=False) if sparse.issparse(matrix) else sparse.csr_matrix(np.asarray(matrix, dtype=np.float32))
    if matrix.shape != adata.shape:
        raise ValueError(f"{label} counts shape mismatch: {matrix.shape} vs {adata.shape}")
    if matrix.data.size:
        idx = np.linspace(0, matrix.data.size - 1, min(200000, matrix.data.size), dtype=np.int64)
        sample = matrix.data[idx]
        if np.any(sample < 0) or not np.allclose(sample, np.rint(sample), atol=1e-6):
            raise ValueError(f"{label} counts layer is not nonnegative integer-like")
    return matrix

def contract_obs(adata, modality, batch_key, celltype_key, dataset_key):
    required = [batch_key, celltype_key, dataset_key]
    missing = [key for key in required if key not in adata.obs.columns]
    if missing:
        raise KeyError(f"{modality} obs missing required columns: {missing}")
    for key in required:
        if adata.obs[key].isna().any():
            raise ValueError(f"{modality} obs[{key!r}] contains missing values")
    original = np.asarray(adata.obs_names.astype(str), dtype=object)
    index_values = np.asarray([f"{modality}_{name}" for name in original], dtype=object)
    obs = pd.DataFrame(index=pd.Index(index_values, dtype=object, name="cell"))
    obs["original_barcode"] = pd.Categorical(original)
    obs["modality"] = pd.Categorical(np.full(len(original), modality, dtype=object))
    obs["batch"] = pd.Categorical(adata.obs[batch_key].astype(str).to_numpy(dtype=object))
    obs["cell_type"] = pd.Categorical(adata.obs[celltype_key].astype(str).to_numpy(dtype=object))
    obs["dataset"] = pd.Categorical(adata.obs[dataset_key].astype(str).to_numpy(dtype=object))
    if obs["batch"].str.strip().eq("").any():
        raise ValueError(f"{modality} batch contains empty strings")
    return obs

def main():
    args = parse_args()
    if not hasattr(np, "NaN"):
        np.NaN = np.nan
    # Reuse the shared GTF mapping and variable-gene utilities.
    base = load_base(Path(args.base_preprocessor))
    rna_path, atac_path = Path(args.rna_h5ad), Path(args.atac_h5ad)
    gtf_path, output = Path(args.gtf), Path(args.output)
    print(f"[prepare] reading RNA QC h5ad: {rna_path}", flush=True)
    rna = ad.read_h5ad(rna_path)
    print(f"[prepare] reading ATAC QC h5ad: {atac_path}", flush=True)
    atac = ad.read_h5ad(atac_path)
    rna_counts = require_sparse_counts(rna, args.rna_counts_layer, "RNA")
    atac_counts = require_sparse_counts(atac, args.atac_counts_layer, "ATAC")
    rna_genes = rna.var_names.astype(str).tolist()
    atac_peaks = atac.var_names.astype(str).tolist()
    rna_obs = contract_obs(rna, "RNA", args.batch_key, args.celltype_key, args.dataset_key)
    atac_obs = contract_obs(atac, "ATAC", args.batch_key, args.celltype_key, args.dataset_key)
    print(f"[prepare] RNA shape: {rna_counts.shape}", flush=True)
    print(f"[prepare] ATAC peak shape: {atac_counts.shape}", flush=True)
    print(f"[prepare] RNA batch counts: {rna_obs['batch'].value_counts().to_dict()}", flush=True)
    print(f"[prepare] ATAC batch counts: {atac_obs['batch'].value_counts().to_dict()}", flush=True)
    genes, rna_gene, atac_gene, route_info = base.build_gtf_gene_activity(
        rna_counts=rna_counts.T.tocsr(),
        rna_genes=rna_genes,
        atac_counts=atac_counts.T.tocsr(),
        atac_peaks=atac_peaks,
        atac_obs=atac_obs,
        gtf=gtf_path,
        geneactivity_annotation=base.none_if_requested(args.geneactivity_annotation),
        strip_atac_chr=args.strip_atac_chr,
    )
    # Assemble the common gene space and its normalized training layer.
    x_counts = sparse.vstack([rna_gene, atac_gene], format="csr").astype(np.float32)
    obs = pd.concat([rna_obs, atac_obs], axis=0)
    if not obs.index.is_unique:
        raise ValueError("Output cell identifiers are not unique")
    var_names = np.asarray(genes, dtype=object)
    var = pd.DataFrame(index=pd.Index(var_names, dtype=object, name="gene"))
    adata = ad.AnnData(X=x_counts.copy(), obs=obs, var=var)
    adata.layers["counts"] = x_counts.copy()
    print("[prepare] normalizing train_input", flush=True)
    train = ad.AnnData(X=x_counts.copy(), obs=obs.copy(), var=var.copy())
    sc.pp.normalize_total(train, target_sum=args.target_sum)
    sc.pp.log1p(train)
    adata.X = train.X.astype(np.float32)
    adata.layers["train_input"] = train.X.astype(np.float32)
    # Select genes using balanced RNA and ATAC scores.
    if 0 < args.top_genes < adata.n_vars:
        print(f"[prepare] selecting {args.top_genes} modality-combined HVGs", flush=True)
        keep, hvg_var, hvg_info = base.select_integrated_hvgs(
            train_input=adata.layers["train_input"], n_rna=len(rna_obs),
            genes=genes, top_genes=args.top_genes, flavor=args.hvg_flavor,
        )
        adata.var = adata.var.join(hvg_var, how="left")
        adata = adata[:, keep].copy()
        adata.var["highly_variable"] = True
    else:
        hvg_info = {
            "hvg_mode": "modality_combined", "hvg_flavor": args.hvg_flavor,
            "top_genes": int(args.top_genes), "selected_genes": int(adata.n_vars),
            "rna_hvg_method": "skipped", "atac_hvg_method": "skipped",
        }
        adata.var["highly_variable"] = True
    adata.uns["preprocess"] = {
        **route_info, **hvg_info, "source": "anchorpt_h5ad_gtf_rna_atac",
        "rna_h5ad": str(rna_path), "atac_h5ad": str(atac_path),
        "rna_counts_layer": args.rna_counts_layer, "atac_counts_layer": args.atac_counts_layer,
        "batch_key": args.batch_key, "celltype_key": args.celltype_key,
        "dataset_key": args.dataset_key, "target_sum": float(args.target_sum),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    adata.write_h5ad(output)
    print(f"[prepare] wrote: {output}", flush=True)
    print(f"[prepare] shape: {adata.shape}", flush=True)
    print(f"[prepare] layers: {list(adata.layers.keys())}", flush=True)
    print(f"[prepare] modality counts: {adata.obs['modality'].value_counts().to_dict()}", flush=True)
    print(f"[prepare] batch counts: {adata.obs['batch'].value_counts().to_dict()}", flush=True)

if __name__ == "__main__":
    main()
