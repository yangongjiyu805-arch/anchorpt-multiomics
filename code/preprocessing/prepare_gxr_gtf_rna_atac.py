"""Prepare feature-aligned RNA-ATAC inputs from Matrix Market data."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.io import mmread


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna-dir", required=True)
    parser.add_argument("--atac-dir", required=True)
    parser.add_argument("--route", choices=("gxr", "gtf"), default="gxr")
    parser.add_argument("--gxr-csv")
    parser.add_argument("--gtf")
    parser.add_argument("--geneactivity-annotation", default="none")
    parser.add_argument("--strip-atac-chr", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-genes", type=int, default=5000)
    parser.add_argument("--target-sum", type=float, default=1e4)
    parser.add_argument("--rna-label-col", default="celltype")
    parser.add_argument("--atac-label-col", default="celltype")
    parser.add_argument("--rna-batch-col", default="batch")
    parser.add_argument("--atac-batch-col", default="batch")
    parser.add_argument("--rna-dataset", default="RNA")
    parser.add_argument("--atac-dataset", default="ATAC")
    parser.add_argument("--no-weight-multigene", action="store_true")
    parser.add_argument("--hvg-flavor", default="seurat")
    return parser.parse_args()


def none_if_requested(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if value.lower() in {"", "none", "null", "na"}:
        return None
    return value


def read_vector(path: Path) -> list[str]:
    return pd.read_csv(path, header=None, sep="\t")[0].astype(str).tolist()


def read_matrix(path: Path) -> sparse.csr_matrix:
    return mmread(path).tocsr().astype(np.float32)


def read_metadata(directory: Path) -> pd.DataFrame:
    for name in ("meta_data.csv", "metadata.csv"):
        path = directory / name
        if path.exists():
            meta = pd.read_csv(path, index_col=0)
            meta.index = meta.index.astype(str)
            return meta
    return pd.DataFrame()


def metadata_values(meta: pd.DataFrame, barcodes: list[str], column: str, fallback: str) -> np.ndarray:
    if column and column in meta.columns:
        return meta.reindex(barcodes)[column].fillna(fallback).astype(str).to_numpy()
    return np.full(len(barcodes), fallback, dtype=object)


def make_obs(
    barcodes: list[str],
    meta: pd.DataFrame,
    modality: str,
    batch_col: str,
    label_col: str,
    dataset: str,
) -> pd.DataFrame:
    obs = pd.DataFrame(index=[f"{modality}_{bc}" for bc in barcodes])
    obs["original_barcode"] = barcodes
    obs["modality"] = modality
    obs["batch"] = metadata_values(meta, barcodes, batch_col, dataset)
    obs["cell_type"] = metadata_values(meta, barcodes, label_col, "unknown")
    obs["dataset"] = dataset
    return obs


def require_feature_by_cell_shape(
    matrix: sparse.csr_matrix,
    features: list[str],
    barcodes: list[str],
    name: str,
) -> None:
    if matrix.shape != (len(features), len(barcodes)):
        raise ValueError(f"{name} matrix shape mismatch: {matrix.shape}")


def convert_peak_name(name: str, strip_chr: bool) -> str:
    if ":" in name:
        chrom, rest = name.split(":", 1)
        start, end = rest.split("-", 1)
    else:
        parts = name.split("-")
        if len(parts) < 3:
            parts = name.split("_")
        if len(parts) < 3:
            raise ValueError(f"Unrecognized peak name: {name}")
        chrom, start, end = parts[0], parts[1], parts[2]
    if strip_chr and chrom.startswith("chr"):
        chrom = chrom[3:]
    return f"{chrom}:{start}-{end}"


def to_csr_float32(matrix) -> sparse.csr_matrix:
    if sparse.issparse(matrix):
        return matrix.tocsr().astype(np.float32)
    return sparse.csr_matrix(np.asarray(matrix, dtype=np.float32))


def build_gxr_gene_activity(
    rna_counts: sparse.csr_matrix,
    rna_genes: list[str],
    atac_counts: sparse.csr_matrix,
    atac_peaks: list[str],
    gxr_csv: Path,
    weight_multigene: bool,
) -> tuple[list[str], sparse.csr_matrix, sparse.csr_matrix, dict]:
    # Restrict the peak-gene map to features observed by both modalities.
    gxr = pd.read_csv(gxr_csv)
    if "peak" not in gxr.columns or "gene.name" not in gxr.columns:
        raise ValueError("GxR CSV must contain columns 'peak' and 'gene.name'.")

    rna_gene_set = set(rna_genes)
    atac_peak_set = set(atac_peaks)
    gxr = gxr[["peak", "gene.name"]].dropna().drop_duplicates().copy()
    gxr["peak"] = gxr["peak"].astype(str)
    gxr["gene.name"] = gxr["gene.name"].astype(str)
    gxr = gxr[gxr["peak"].isin(atac_peak_set) & gxr["gene.name"].isin(rna_gene_set)].copy()
    if gxr.empty:
        raise ValueError("No valid GxR rows after intersecting RNA genes and ATAC peaks.")

    mapped_genes = set(gxr["gene.name"])
    common_genes = [gene for gene in rna_genes if gene in mapped_genes]
    print(f"[prepare] shared mapped genes: {len(common_genes)}")
    print(f"[prepare] valid GxR rows: {len(gxr)}")

    rna_gene_to_idx = {gene: idx for idx, gene in enumerate(rna_genes)}
    rna_idx = np.array([rna_gene_to_idx[gene] for gene in common_genes], dtype=np.int64)
    rna_cells_by_genes = rna_counts.T.tocsr()[:, rna_idx]

    peak_to_idx = {peak: idx for idx, peak in enumerate(atac_peaks)}
    gene_to_idx = {gene: idx for idx, gene in enumerate(common_genes)}
    rows = gxr["peak"].map(peak_to_idx).to_numpy(dtype=np.int64)
    cols = gxr["gene.name"].map(gene_to_idx).to_numpy(dtype=np.int64)
    if weight_multigene:
        per_peak = gxr.groupby("peak")["gene.name"].transform("nunique").to_numpy(dtype=np.float32)
        weights = 1.0 / np.maximum(per_peak, 1.0)
    else:
        weights = np.ones(len(gxr), dtype=np.float32)

    peak_to_gene = sparse.csr_matrix(
        (weights, (rows, cols)),
        shape=(len(atac_peaks), len(common_genes)),
        dtype=np.float32,
    )
    print("[prepare] aggregating ATAC peaks to genes with GxR")
    atac_cells_by_genes = (atac_counts.T.tocsr() @ peak_to_gene).tocsr().astype(np.float32)
    route_info = {
        "source": "gxr_rna_atac",
        "gxr": str(gxr_csv),
        "weight_multigene": bool(weight_multigene),
        "valid_gxr_rows": int(len(gxr)),
    }
    return common_genes, rna_cells_by_genes, atac_cells_by_genes, route_info


def build_gtf_gene_activity(
    rna_counts: sparse.csr_matrix,
    rna_genes: list[str],
    atac_counts: sparse.csr_matrix,
    atac_peaks: list[str],
    atac_obs: pd.DataFrame,
    gtf: Path,
    geneactivity_annotation: str | None,
    strip_atac_chr: bool,
) -> tuple[list[str], sparse.csr_matrix, sparse.csr_matrix, dict]:
    try:
        import episcanpy as epi
    except ImportError as exc:
        raise ImportError("The GTF route requires episcanpy to be installed.") from exc

    # Convert peak identifiers before deriving gene activity from the GTF.
    converted_peaks = [convert_peak_name(peak, strip_chr=strip_atac_chr) for peak in atac_peaks]
    atac_var = pd.DataFrame(index=pd.Index(converted_peaks, name="peak"))
    atac_var["original_peak"] = atac_peaks
    atac_peak_profile = ad.AnnData(X=atac_counts.T.tocsr(), obs=atac_obs.copy(), var=atac_var)
    atac_peak_profile.layers["counts"] = atac_peak_profile.X.copy()

    print("[prepare] aggregating ATAC peaks to genes with GTF/episcanpy")
    atac_gene_activity = epi.tl.geneactivity(
        atac_peak_profile,
        str(gtf),
        annotation=geneactivity_annotation,
    )
    atac_gene_activity = atac_gene_activity[:, ~atac_gene_activity.var_names.duplicated()].copy()
    atac_cells_by_all_genes = to_csr_float32(atac_gene_activity.X)

    activity_gene_to_idx = {gene: idx for idx, gene in enumerate(atac_gene_activity.var_names.astype(str))}
    common_genes = [gene for gene in rna_genes if gene in activity_gene_to_idx]
    if not common_genes:
        raise ValueError("No shared genes after intersecting RNA genes and GTF-derived ATAC gene activity.")
    print(f"[prepare] shared GTF-derived genes: {len(common_genes)}")

    rna_gene_to_idx = {gene: idx for idx, gene in enumerate(rna_genes)}
    rna_idx = np.array([rna_gene_to_idx[gene] for gene in common_genes], dtype=np.int64)
    atac_idx = np.array([activity_gene_to_idx[gene] for gene in common_genes], dtype=np.int64)
    rna_cells_by_genes = rna_counts.T.tocsr()[:, rna_idx]
    atac_cells_by_genes = atac_cells_by_all_genes[:, atac_idx]
    route_info = {
        "source": "gtf_rna_atac",
        "gtf": str(gtf),
        "geneactivity_annotation": geneactivity_annotation,
        "strip_atac_chr": bool(strip_atac_chr),
    }
    return common_genes, rna_cells_by_genes, atac_cells_by_genes, route_info


def variance_scores(matrix: sparse.csr_matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        mean = np.asarray(matrix.mean(axis=0)).reshape(-1)
        mean_sq = np.asarray(matrix.multiply(matrix).mean(axis=0)).reshape(-1)
    else:
        values = np.asarray(matrix, dtype=np.float32)
        mean = values.mean(axis=0)
        mean_sq = np.square(values).mean(axis=0)
    scores = mean_sq - np.square(mean)
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


def modality_hvg_scores(
    matrix: sparse.csr_matrix,
    genes: list[str],
    top_genes: int,
    flavor: str,
    label: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    tmp = ad.AnnData(X=matrix.copy(), var=pd.DataFrame(index=pd.Index(genes, name="gene")))
    n_top = min(max(int(top_genes), 1), tmp.n_vars)
    try:
        sc.pp.highly_variable_genes(tmp, n_top_genes=n_top, flavor=flavor)
        if "dispersions_norm" in tmp.var:
            scores = tmp.var["dispersions_norm"].to_numpy(dtype=np.float64)
        elif "variances_norm" in tmp.var:
            scores = tmp.var["variances_norm"].to_numpy(dtype=np.float64)
        elif "highly_variable_rank" in tmp.var:
            ranks = tmp.var["highly_variable_rank"].to_numpy(dtype=np.float64)
            scores = np.nanmax(ranks) - ranks
        else:
            scores = variance_scores(matrix)
        flags = tmp.var["highly_variable"].fillna(False).to_numpy(dtype=bool)
        method = flavor
    except Exception as exc:
        print(f"[prepare] WARNING: {label} HVG failed with {type(exc).__name__}: {exc}")
        print(f"[prepare] WARNING: falling back to variance scores for {label}")
        scores = variance_scores(matrix)
        order = np.argsort(-scores, kind="mergesort")
        flags = np.zeros(len(scores), dtype=bool)
        flags[order[:n_top]] = True
        method = "variance_fallback"
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    return scores, flags, method


def percentile_rank_scores(scores: np.ndarray) -> np.ndarray:
    finite = np.isfinite(scores)
    ranked = np.zeros(len(scores), dtype=np.float64)
    if not finite.any():
        return ranked
    finite_scores = scores[finite]
    finite_positions = np.where(finite)[0]
    order = np.lexsort((finite_positions, -finite_scores))
    n = len(order)
    values = np.linspace(1.0, 1.0 / n, n)
    ranked[finite_positions[order]] = values
    return ranked


def select_integrated_hvgs(
    train_input: sparse.csr_matrix,
    n_rna: int,
    genes: list[str],
    top_genes: int,
    flavor: str,
) -> tuple[np.ndarray, pd.DataFrame, dict]:
    if top_genes <= 0 or top_genes >= len(genes):
        keep = np.ones(len(genes), dtype=bool)
        var_scores = pd.DataFrame(index=pd.Index(genes, name="gene"))
        var_scores["highly_variable"] = True
        return keep, var_scores, {
            "hvg_mode": "modality_combined",
            "hvg_flavor": flavor,
            "top_genes": int(top_genes),
            "selected_genes": int(keep.sum()),
            "rna_hvg_method": "skipped",
            "atac_hvg_method": "skipped",
        }

    rna_matrix = train_input[:n_rna, :]
    atac_matrix = train_input[n_rna:, :]
    rna_scores, rna_flags, rna_method = modality_hvg_scores(
        rna_matrix,
        genes,
        top_genes=top_genes,
        flavor=flavor,
        label="RNA",
    )
    atac_scores, atac_flags, atac_method = modality_hvg_scores(
        atac_matrix,
        genes,
        top_genes=top_genes,
        flavor=flavor,
        label="ATAC",
    )

    # Combine modality-specific rankings with equal contribution.
    rna_rank = percentile_rank_scores(rna_scores)
    atac_rank = percentile_rank_scores(atac_scores)
    combined = 0.5 * rna_rank + 0.5 * atac_rank
    order = np.lexsort((np.arange(len(genes)), -combined))
    selected = order[: min(top_genes, len(genes))]
    keep = np.zeros(len(genes), dtype=bool)
    keep[selected] = True

    var_scores = pd.DataFrame(index=pd.Index(genes, name="gene"))
    var_scores["rna_hvg_score"] = rna_scores
    var_scores["atac_hvg_score"] = atac_scores
    var_scores["rna_hvg_rank_score"] = rna_rank
    var_scores["atac_hvg_rank_score"] = atac_rank
    var_scores["combined_hvg_score"] = combined
    var_scores["rna_highly_variable"] = rna_flags
    var_scores["atac_highly_variable"] = atac_flags
    var_scores["highly_variable"] = keep
    hvg_info = {
        "hvg_mode": "modality_combined",
        "hvg_flavor": flavor,
        "top_genes": int(top_genes),
        "selected_genes": int(keep.sum()),
        "rna_hvg_method": rna_method,
        "atac_hvg_method": atac_method,
    }
    return keep, var_scores, hvg_info


def main() -> None:
    args = parse_args()
    rna_dir = Path(args.rna_dir)
    atac_dir = Path(args.atac_dir)

    if args.route == "gxr" and not args.gxr_csv:
        raise ValueError("--gxr-csv is required when --route gxr.")
    if args.route == "gtf" and not args.gtf:
        raise ValueError("--gtf is required when --route gtf.")

    print(f"[prepare] RNA: {rna_dir}")
    rna_counts = read_matrix(rna_dir / "matrix.mtx")
    rna_genes = read_vector(rna_dir / "features.tsv")
    rna_barcodes = read_vector(rna_dir / "barcodes.tsv")
    rna_meta = read_metadata(rna_dir)

    print(f"[prepare] ATAC: {atac_dir}")
    atac_counts = read_matrix(atac_dir / "matrix.mtx")
    atac_peaks = read_vector(atac_dir / "features.tsv")
    atac_barcodes = read_vector(atac_dir / "barcodes.tsv")
    atac_meta = read_metadata(atac_dir)

    require_feature_by_cell_shape(rna_counts, rna_genes, rna_barcodes, "RNA")
    require_feature_by_cell_shape(atac_counts, atac_peaks, atac_barcodes, "ATAC")

    rna_obs = make_obs(
        rna_barcodes,
        rna_meta,
        "RNA",
        args.rna_batch_col,
        args.rna_label_col,
        args.rna_dataset,
    )
    atac_obs = make_obs(
        atac_barcodes,
        atac_meta,
        "ATAC",
        args.atac_batch_col,
        args.atac_label_col,
        args.atac_dataset,
    )

    if args.route == "gxr":
        common_genes, rna_cells_by_genes, atac_cells_by_genes, route_info = build_gxr_gene_activity(
            rna_counts=rna_counts,
            rna_genes=rna_genes,
            atac_counts=atac_counts,
            atac_peaks=atac_peaks,
            gxr_csv=Path(args.gxr_csv),
            weight_multigene=not args.no_weight_multigene,
        )
    else:
        common_genes, rna_cells_by_genes, atac_cells_by_genes, route_info = build_gtf_gene_activity(
            rna_counts=rna_counts,
            rna_genes=rna_genes,
            atac_counts=atac_counts,
            atac_peaks=atac_peaks,
            atac_obs=atac_obs,
            gtf=Path(args.gtf),
            geneactivity_annotation=none_if_requested(args.geneactivity_annotation),
            strip_atac_chr=args.strip_atac_chr,
        )

    # Assemble aligned counts before creating the normalized training layer.
    x_counts = sparse.vstack([rna_cells_by_genes, atac_cells_by_genes], format="csr").astype(np.float32)
    obs = pd.concat([rna_obs, atac_obs], axis=0)
    var = pd.DataFrame(index=pd.Index(common_genes, name="gene"))
    adata = ad.AnnData(X=x_counts.copy(), obs=obs, var=var)
    adata.layers["counts"] = x_counts.copy()

    print("[prepare] normalizing train_input")
    train = ad.AnnData(X=x_counts.copy(), obs=obs.copy(), var=var.copy())
    sc.pp.normalize_total(train, target_sum=args.target_sum)
    sc.pp.log1p(train)
    adata.X = train.X.astype(np.float32)
    adata.layers["train_input"] = train.X.astype(np.float32)

    if args.top_genes > 0 and args.top_genes < adata.n_vars:
        print(f"[prepare] selecting top {args.top_genes} genes with modality-combined HVG scores")
        keep, hvg_var, hvg_info = select_integrated_hvgs(
            train_input=adata.layers["train_input"],
            n_rna=len(rna_barcodes),
            genes=common_genes,
            top_genes=args.top_genes,
            flavor=args.hvg_flavor,
        )
        adata.var = adata.var.join(hvg_var, how="left")
        adata = adata[:, keep].copy()
        adata.var["highly_variable"] = True
    else:
        hvg_info = {
            "hvg_mode": "modality_combined",
            "hvg_flavor": args.hvg_flavor,
            "top_genes": int(args.top_genes),
            "selected_genes": int(adata.n_vars),
            "rna_hvg_method": "skipped",
            "atac_hvg_method": "skipped",
        }
        adata.var["highly_variable"] = True

    adata.uns["preprocess"] = {
        **route_info,
        **hvg_info,
        "route": args.route,
        "rna_dir": str(rna_dir),
        "atac_dir": str(atac_dir),
        "target_sum": float(args.target_sum),
        "rna_label_col": args.rna_label_col,
        "atac_label_col": args.atac_label_col,
        "rna_batch_col": args.rna_batch_col,
        "atac_batch_col": args.atac_batch_col,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output)
    print(f"[prepare] wrote: {output}")
    print(f"[prepare] shape: {adata.shape}")
    print(f"[prepare] modality counts: {adata.obs['modality'].value_counts().to_dict()}")
    print(f"[prepare] batch counts: {adata.obs['batch'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
