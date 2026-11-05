"""Evaluate an AnchorPT embedding with scIB-style metrics.

The script validates AnnData fields, builds reusable nearest-neighbor graphs,
and computes biological conservation, modality integration, and optional batch
correction scores. It exports individual metrics together with the weighted
aggregate used by the benchmark.
"""

import argparse
import contextlib
import io
import json
import os
import re
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

import scib_metrics
from scib_metrics.nearest_neighbors import NeighborsResults, pynndescent


Kwargs = Dict[str, Any]
MetricType = Union[bool, Kwargs]

_LABELS = "labels"
_BATCH = "batch"
_MODALITY = "modality"
_X_PRE = "X_pre"
_METRIC_TYPE = "Metric Type"
_METRIC_NAME = "Metric Name"
_AGGREGATE_SCORE = "Aggregate score"
_EXCLUDE_FROM_AGGREGATE_METRICS = set()


@dataclass(frozen=True)
class BioScore:
    isolated_labels: MetricType = True
    nmi_ari_cluster_labels_leiden: MetricType = True
    silhouette_label: MetricType = True
    clisi_knn: MetricType = True


@dataclass(frozen=True)
class BatchScore:
    silhouette_batch_b: MetricType = True
    ilisi_knn_b: MetricType = True
    kbet_per_label_b: MetricType = True
    pcr_comparison_b: MetricType = True


@dataclass(frozen=True)
class MixScore:
    silhouette_batch_m: MetricType = True
    ilisi_knn_m: MetricType = True
    kbet_per_label_m: MetricType = True
    graph_connectivity: MetricType = True
    pcr_comparison_m: MetricType = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute scIB-style metrics for an embedding stored in .h5ad."
    )
    parser.add_argument("--input", required=True, help="Input .h5ad file.")
    parser.add_argument(
        "--embedding-key",
        default="latent_shared",
        help="AnnData .obsm key to evaluate.",
    )
    parser.add_argument(
        "--label-key",
        default="cell_type",
        help="AnnData .obs key with biological labels.",
    )
    parser.add_argument(
        "--modality-key",
        default="modality",
        help="AnnData .obs key with modality labels.",
    )
    parser.add_argument(
        "--batch-key",
        default="none",
        help="AnnData .obs key for batch metrics. Use 'none' to skip batch correction metrics.",
    )
    parser.add_argument(
        "--pre-embedding-key",
        default=None,
        help=(
            "Optional AnnData .obsm key for the pre-integrated embedding used by PCR. "
            "If omitted, PCA is computed from .X."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=24,
        help="Number of jobs for nearest-neighbor search.",
    )
    parser.add_argument(
        "--ilisi-knn-m",
        type=int,
        default=90,
        help="Number of neighbors used for modality iLISI. Defaults to 90, matching previous runs.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed used for PCA and approximate nearest-neighbor search.",
    )
    parser.add_argument(
        "--min-max-scale",
        action="store_true",
        help="Min-max scale metric columns before aggregate scores. Off by default.",
    )
    parser.add_argument(
        "--format",
        choices=("tsv", "csv", "json"),
        default="tsv",
        help="Output format.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output file. If omitted, scores are printed to stdout.",
    )
    return parser.parse_args()


def normalize_key(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    if value == "" or value.lower() in {"none", "null", "na", "nan"}:
        return None
    return value


def require_key(container: Any, key: str, container_name: str) -> None:
    if key not in container:
        raise KeyError(f"{container_name} does not contain key {key!r}")


def clean_obs_values(adata: AnnData, keys: List[str]) -> None:
    for key in keys:
        values = adata.obs[key]
        if pd.isna(values).any():
            raise ValueError(f"obs[{key!r}] contains missing values")
        adata.obs[key] = values.astype(str).astype("category")


# Adapt label containers for scib-metrics versions that require NumPy arrays.
def patch_scib_series_inputs() -> None:
    def arr(value: Any) -> np.ndarray:
        return np.asarray(value)

    original = scib_metrics.graph_connectivity

    def graph_connectivity(neighbor_res, labels, *args, _original=original, **kwargs):
        return _original(neighbor_res, arr(labels), *args, **kwargs)

    scib_metrics.graph_connectivity = graph_connectivity

    original = scib_metrics.kbet_per_label

    def kbet_per_label(neighbor_res, batch, labels, *args, _original=original, **kwargs):
        return _original(neighbor_res, arr(batch), arr(labels), *args, **kwargs)

    scib_metrics.kbet_per_label = kbet_per_label

    original = scib_metrics.ilisi_knn

    def ilisi_knn(neighbor_res, labels, *args, _original=original, **kwargs):
        return _original(neighbor_res, arr(labels), *args, **kwargs)

    scib_metrics.ilisi_knn = ilisi_knn

    original = scib_metrics.clisi_knn

    def clisi_knn(neighbor_res, labels, *args, _original=original, **kwargs):
        return _original(neighbor_res, arr(labels), *args, **kwargs)

    scib_metrics.clisi_knn = clisi_knn

    original = scib_metrics.silhouette_label

    def silhouette_label(x, labels, *args, _original=original, **kwargs):
        return _original(x, arr(labels), *args, **kwargs)

    scib_metrics.silhouette_label = silhouette_label

    original = scib_metrics.silhouette_batch

    def silhouette_batch(x, labels, batch, *args, _original=original, **kwargs):
        return _original(x, arr(labels), arr(batch), *args, **kwargs)

    scib_metrics.silhouette_batch = silhouette_batch

    original = scib_metrics.pcr_comparison

    def pcr_comparison(x_pre, x_post, covariate, *args, _original=original, **kwargs):
        return _original(x_pre, x_post, arr(covariate), *args, **kwargs)

    scib_metrics.pcr_comparison = pcr_comparison

    original = scib_metrics.isolated_labels

    def isolated_labels(x, labels, batch, *args, _original=original, **kwargs):
        return _original(x, arr(labels), arr(batch), *args, **kwargs)

    scib_metrics.isolated_labels = isolated_labels

    original = scib_metrics.nmi_ari_cluster_labels_leiden

    def nmi_ari_cluster_labels_leiden(neighbor_res, labels, *args, _original=original, **kwargs):
        return _original(neighbor_res, arr(labels), *args, **kwargs)

    scib_metrics.nmi_ari_cluster_labels_leiden = nmi_ari_cluster_labels_leiden


# Bind each metric to the embedding, metadata, and neighborhood scale it requires.
def run_metric(metric_name: str, ad: AnnData, metric_fn: Callable[..., Any]) -> Any:
    if metric_name == "isolated_labels":
        return metric_fn(ad.X, ad.obs[_LABELS], ad.obs[_MODALITY])
    if metric_name == "nmi_ari_cluster_labels_leiden":
        return metric_fn(ad.uns["15_neighbor_res"], ad.obs[_LABELS])
    if metric_name == "silhouette_label":
        return metric_fn(ad.X, ad.obs[_LABELS])
    if metric_name == "clisi_knn":
        return metric_fn(ad.uns["90_neighbor_res"], ad.obs[_LABELS])
    if metric_name == "silhouette_batch_b":
        return metric_fn(ad.X, ad.obs[_LABELS], ad.obs[_BATCH])
    if metric_name == "pcr_comparison_b":
        return metric_fn(ad.obsm[_X_PRE], ad.X, ad.obs[_BATCH], categorical=True)
    if metric_name == "ilisi_knn_b":
        return metric_fn(ad.uns["90_neighbor_res"], ad.obs[_BATCH])
    if metric_name == "kbet_per_label_b":
        return metric_fn(ad.uns["50_neighbor_res"], ad.obs[_BATCH], ad.obs[_LABELS])
    if metric_name == "graph_connectivity":
        return metric_fn(ad.uns["15_neighbor_res"], ad.obs[_LABELS])
    if metric_name == "silhouette_batch_m":
        return metric_fn(ad.X, ad.obs[_LABELS], ad.obs[_MODALITY])
    if metric_name == "pcr_comparison_m":
        return metric_fn(ad.obsm[_X_PRE], ad.X, ad.obs[_MODALITY], categorical=True)
    if metric_name == "ilisi_knn_m":
        return metric_fn(ad.uns["ilisi_knn_m_neighbor_res"], ad.obs[_MODALITY])
    if metric_name == "kbet_per_label_m":
        return metric_fn(ad.uns["50_neighbor_res"], ad.obs[_MODALITY], ad.obs[_LABELS])
    raise KeyError(f"Unsupported metric {metric_name!r}")


class EmbeddingBench:
    def __init__(
        self,
        adata: AnnData,
        batch_key: str,
        label_key: str,
        modality_key: str,
        embedding_obsm_keys: List[str],
        bio_conservation_metrics: Optional[BioScore],
        batch_correction_metrics: Optional[BatchScore],
        modality_integration_metrics: Optional[MixScore],
        pre_integrated_embedding_obsm_key: Optional[str] = None,
        n_jobs: int = 1,
        random_state: int = 0,
        ilisi_knn_m: int = 90,
    ) -> None:
        self._adata = adata
        self._embedding_obsm_keys = embedding_obsm_keys
        self._pre_integrated_embedding_obsm_key = pre_integrated_embedding_obsm_key
        self._bio_conservation_metrics = bio_conservation_metrics
        self._batch_correction_metrics = batch_correction_metrics
        self._modality_integration_metrics = modality_integration_metrics
        self._results = pd.DataFrame(columns=list(self._embedding_obsm_keys) + [_METRIC_TYPE])
        self._emb_adatas: Dict[str, AnnData] = {}
        self._ilisi_knn_m = int(ilisi_knn_m)
        self._neighbor_values = tuple(sorted({15, 50, 90, self._ilisi_knn_m}))
        self._prepared = False
        self._benchmarked = False
        self._batch_key = batch_key
        self._modality_key = modality_key
        self._label_key = label_key
        self._n_jobs = n_jobs
        self._random_state = random_state
        if self._bio_conservation_metrics is None and self._batch_correction_metrics is None:
            raise ValueError("Either batch or bio metrics must be defined.")
        self._metric_collection_dict: Dict[str, Any] = {}
        if self._bio_conservation_metrics is not None:
            self._metric_collection_dict["Bio conservation"] = self._bio_conservation_metrics
        if self._batch_correction_metrics is not None:
            self._metric_collection_dict["Batch correction"] = self._batch_correction_metrics
        if self._modality_integration_metrics is not None:
            self._metric_collection_dict["Modality integration"] = self._modality_integration_metrics

    def prepare(
        self,
        neighbor_computer: Optional[Callable[[np.ndarray, int], NeighborsResults]] = None,
    ) -> None:
        if self._pre_integrated_embedding_obsm_key is None:
            sc.tl.pca(self._adata, use_highly_variable=False, random_state=self._random_state)
            self._pre_integrated_embedding_obsm_key = "X_pca"

        for emb_key in self._embedding_obsm_keys:
            self._emb_adatas[emb_key] = AnnData(self._adata.obsm[emb_key], obs=self._adata.obs.copy())
            ad = self._emb_adatas[emb_key]
            ad.obs[_BATCH] = np.asarray(self._adata.obs[self._batch_key].values)
            ad.obs[_MODALITY] = np.asarray(self._adata.obs[self._modality_key].values)
            ad.obs[_LABELS] = np.asarray(self._adata.obs[self._label_key].values)
            ad.obsm[_X_PRE] = self._adata.obsm[self._pre_integrated_embedding_obsm_key]

        # Reuse subsets of one maximum-size neighbor graph across neighborhood metrics.
        for ad in self._emb_adatas.values():
            if neighbor_computer is not None:
                neigh_result = neighbor_computer(ad.X, max(self._neighbor_values))
            else:
                neigh_result = pynndescent(
                    ad.X,
                    n_neighbors=max(self._neighbor_values),
                    random_state=self._random_state,
                    n_jobs=self._n_jobs,
                )
            for n_neighbors in self._neighbor_values:
                ad.uns[f"{n_neighbors}_neighbor_res"] = neigh_result.subset_neighbors(n=n_neighbors)
            ad.uns["ilisi_knn_m_neighbor_res"] = ad.uns[f"{self._ilisi_knn_m}_neighbor_res"]

        self._prepared = True

    def benchmark(self) -> None:
        if self._benchmarked:
            warnings.warn(
                "The benchmark has already been run. Running it again will overwrite the previous results.",
                UserWarning,
            )
        if not self._prepared:
            self.prepare()

        for emb_key, ad in self._emb_adatas.items():
            for metric_type, metric_collection in self._metric_collection_dict.items():
                for metric_name, use_metric_or_kwargs in asdict(metric_collection).items():
                    if not use_metric_or_kwargs:
                        continue
                    scib_metric_name = re.sub(r"(_b|_m)$", "", metric_name)
                    metric_fn = getattr(scib_metrics, scib_metric_name)
                    if isinstance(use_metric_or_kwargs, dict):
                        metric_fn = partial(metric_fn, **use_metric_or_kwargs)
                    try:
                        metric_value = run_metric(metric_name, ad, metric_fn)
                    except Exception as exc:
                        warnings.warn(
                            f"Metric {metric_name} failed for embedding {emb_key}: "
                            f"{type(exc).__name__}: {exc}",
                            UserWarning,
                        )
                        metric_value = np.nan

                    if isinstance(metric_value, dict):
                        for suffix, value in metric_value.items():
                            row_name = f"{metric_type}_{metric_name}_{suffix}"
                            self._results.loc[row_name, emb_key] = value
                            self._results.loc[row_name, _METRIC_TYPE] = metric_type
                            self._results.loc[row_name, _METRIC_NAME] = f"{metric_name}_{suffix}"
                    else:
                        row_name = f"{metric_type}_{metric_name}"
                        self._results.loc[row_name, emb_key] = metric_value
                        self._results.loc[row_name, _METRIC_TYPE] = metric_type
                        self._results.loc[row_name, _METRIC_NAME] = metric_name

        self._benchmarked = True

    def get_results(self, min_max_scale: bool = False) -> pd.DataFrame:
        df = self._results.transpose()
        df.index.name = "Embedding"
        df = df.loc[~df.index.isin([_METRIC_TYPE, _METRIC_NAME])]

        if min_max_scale:
            from sklearn.preprocessing import MinMaxScaler

            df = pd.DataFrame(
                MinMaxScaler().fit_transform(df),
                columns=self._results[_METRIC_NAME].values,
                index=df.index,
            )
        else:
            df = pd.DataFrame(
                df.to_numpy(),
                columns=self._results[_METRIC_NAME].values,
                index=df.index,
            )

        df = df.transpose()
        df[_METRIC_TYPE] = self._results[_METRIC_TYPE].values
        df[_METRIC_NAME] = self._results[_METRIC_NAME].values
        aggregate_df = df.loc[~df[_METRIC_NAME].isin(_EXCLUDE_FROM_AGGREGATE_METRICS)]
        aggregate_df = aggregate_df.drop(columns=[_METRIC_NAME])
        per_class_score = aggregate_df.groupby(_METRIC_TYPE).mean().transpose()

        # Combine metric groups with the fixed weights used by the reported benchmark.
        if (
            self._modality_integration_metrics is not None
            and self._batch_correction_metrics is not None
            and self._bio_conservation_metrics is not None
        ):
            per_class_score["Total"] = (
                0.3 * per_class_score["Batch correction"]
                + 0.3 * per_class_score["Modality integration"]
                + 0.4 * per_class_score["Bio conservation"]
            )
        elif (
            self._modality_integration_metrics is not None
            and self._bio_conservation_metrics is not None
            and self._batch_correction_metrics is None
        ):
            per_class_score["Total"] = (
                0.4 * per_class_score["Modality integration"]
                + 0.6 * per_class_score["Bio conservation"]
            )

        df = pd.concat([df.transpose(), per_class_score], axis=1)
        df[per_class_score.columns] = df[per_class_score.columns].astype(object)
        df.loc[_METRIC_TYPE, per_class_score.columns] = _AGGREGATE_SCORE
        df.loc[_METRIC_NAME, per_class_score.columns] = per_class_score.columns
        return df


def dataframe_to_scores(df: pd.DataFrame, embedding_key: str) -> List[Dict[str, object]]:
    if embedding_key not in df.index:
        raise KeyError(f"Result table does not contain embedding row {embedding_key!r}")

    scores: List[Dict[str, object]] = []
    metric_type_row = df.loc[_METRIC_TYPE]
    metric_name_row = df.loc[_METRIC_NAME]
    value_row = df.loc[embedding_key]

    for column in df.columns:
        raw_value = value_row[column]
        if pd.isna(raw_value):
            continue
        value = float(raw_value)
        if not np.isfinite(value):
            continue
        scores.append(
            {
                "metric_type": str(metric_type_row[column]),
                "metric_name": str(metric_name_row[column]),
                "score": value,
            }
        )
    return scores


def scores_to_text(scores: List[Dict[str, object]], output_format: str) -> str:
    if output_format == "json":
        return json.dumps(scores, ensure_ascii=False, sort_keys=False)

    df = pd.DataFrame(scores, columns=["metric_type", "metric_name", "score"])
    if output_format == "csv":
        return df.to_csv(index=False).rstrip("\n")
    return df.to_csv(index=False, sep="\t").rstrip("\n")


def main() -> int:
    args = parse_args()
    batch_key = normalize_key(args.batch_key)
    pre_embedding_key = normalize_key(args.pre_embedding_key)

    warnings.filterwarnings("ignore")
    sc.settings.verbosity = 0
    patch_scib_series_inputs()

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        adata = sc.read_h5ad(args.input)

        require_key(adata.obsm, args.embedding_key, "adata.obsm")
        require_key(adata.obs, args.label_key, "adata.obs")
        require_key(adata.obs, args.modality_key, "adata.obs")

        keys_to_clean = [args.label_key, args.modality_key]
        if batch_key is not None:
            require_key(adata.obs, batch_key, "adata.obs")
            keys_to_clean.append(batch_key)
        if pre_embedding_key is not None:
            require_key(adata.obsm, pre_embedding_key, "adata.obsm")
        clean_obs_values(adata, keys_to_clean)

        if batch_key is None:
            batch_key_for_api = "__dummy_batch__"
            adata.obs[batch_key_for_api] = "batch0"
        else:
            batch_key_for_api = batch_key

        bio_metrics = BioScore()
        batch_metrics = BatchScore() if batch_key is not None else None
        benchmarker = EmbeddingBench(
            adata=adata,
            batch_key=batch_key_for_api,
            label_key=args.label_key,
            modality_key=args.modality_key,
            embedding_obsm_keys=[args.embedding_key],
            bio_conservation_metrics=bio_metrics,
            batch_correction_metrics=batch_metrics,
            modality_integration_metrics=MixScore(),
            pre_integrated_embedding_obsm_key=pre_embedding_key,
            n_jobs=args.n_jobs,
            random_state=args.random_state,
            ilisi_knn_m=args.ilisi_knn_m,
        )
        benchmarker.prepare()
        benchmarker.benchmark()
        results = benchmarker.get_results(min_max_scale=args.min_max_scale)
        scores = dataframe_to_scores(results, args.embedding_key)

    output = scores_to_text(scores, args.format)
    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="") as handle:
            handle.write(output + "\n")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    raise SystemExit(main())

