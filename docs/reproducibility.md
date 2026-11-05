# Reproducibility and evaluation

[Home](../README.md) · [Configuration](configuration.md) · [Data preparation](preprocessing.md)

This guide defines the training records, embedding roles, and evaluation settings needed to compare AnchorPT experiments.

## Included components

The repository provides preprocessing scripts, two training entry points, model and transport objectives, and embedding evaluation. Runnable workflow scripts are provided in [experiment/rna_atac/](../experiment/rna_atac/).

Benchmark datasets, reference annotations, complete per-benchmark configurations, trained checkpoints, and reference score tables are supplied separately. Check environment compatibility as described in [installation](installation.md).

Both training entry points and runner APIs default to 80 refinement epochs. Retain the complete experiment configuration, including dataset splits and preprocessing, with each run.

## Preserve the experimental record

For each run, keep:

1. The repository commit, selected source tree, Python/package versions, and hardware.
2. Raw-data and annotation versions/checksums, preprocessing arguments, and cell/feature order.
3. Split assignments and seeds, including whether biological labels were used to stratify a split.
4. The effective parameter JSON, method-role manifest, training logs, and executed epoch counts.
5. The integrated AnnData file, evaluation arguments, evaluation seed, and all metric outputs.

Use disjoint cell allocations for an unpaired benchmark. Record this separately from whether the model uses cell-type labels. The bundled three-modality split constructor uses canonical cell types as one stratification field; the default model training remains label-free.

## Embedding roles

| AnnData key | Interpretation |
| :--- | :--- |
| `obsm["latent_shared"]` | Primary integration embedding |
| `obsm["latent_specific"]` | Modality-specific representation |
| `obsm["latent_concat"]` | Concatenated diagnostic representation |
| `obsm["latent_shared_stage1_anchor"]` | Stage-I reference, when available |
| `obsm["latent_shared_stage2_raw"]` | Pre-blend Stage-II embedding, exported when blending is enabled |

The default `ANCHORPT_STAGE2_OUTPUT_BLEND=1.0` uses the refined encoder output directly. With a value below 1, the primary output is blended with the Stage-I reference.

The extended runner additionally exports a CORAL-based diagnostic representation under `latent_shared_coral_avg_s055` for two modalities or `latent_shared_coral_avg_s085` otherwise. Evaluate `latent_shared` for the primary analysis; choosing among diagnostic outputs using test labels would change the selection protocol.

## Evaluation settings

Both source trees include the same evaluator implementation. The [quickstart](quickstart.md#4-evaluate) explicitly selects the primary shared embedding.

| Option | Default | Effect |
| :--- | :--- | :--- |
| `--embedding-key` | `latent_shared` | Embedding to evaluate |
| `--label-key` | `cell_type` | Biological labels |
| `--modality-key` | `modality` | Modality identities |
| `--batch-key` | `none` | Omit batch metrics unless a key is supplied |
| `--pre-embedding-key` | Unset | Compute pre-integration PCA from `.X` when unset |
| `--random-state` | `0` | PCA and approximate-neighbor seed |
| `--ilisi-knn-m` | `90` | Modality iLISI neighborhood size |
| `--min-max-scale` | Off | Optional scaling before aggregation |
| `--format` | `tsv` | Also supports CSV and JSON |

Only enable batch metrics when the batch variable is interpretable for the benchmark. A modality-confounded dataset identifier should not automatically be used as a batch key.

The evaluator computes neighbor sets at sizes 15, 50, and 90, together with the requested modality-iLISI size. Inputs must contain enough cells for these neighborhood sizes.

## Aggregate scores

Let `Bio`, `Modality`, and `Batch` denote arithmetic means of the corresponding metric groups in the evaluator.

| Evaluation setting | Total |
| :--- | :--- |
| With batch metrics | `0.4 × Bio + 0.3 × Modality + 0.3 × Batch` |
| Without batch metrics | `0.6 × Bio + 0.4 × Modality` |

The biological group includes isolated-label, Leiden NMI/ARI, label-silhouette, and cLISI metrics. The modality group includes modality silhouette, iLISI, kBET, graph connectivity, and PCR. The batch group includes batch silhouette, iLISI, kBET, and PCR.

Leiden NMI and ARI are included in the biological-conservation mean and therefore contribute to Total. The method manifest records this role under `selection_policy.leiden_role`.

## Missing metrics and repeated runs

Metric failures are converted to missing values inside the benchmarker; the command-line wrapper suppresses warnings, and its output omits nonfinite scores. Group means can therefore use only the surviving metrics. A printed Total does not by itself establish that all constituent metrics succeeded.

Inspect the metric-name inventory and confirm the same constituent metrics across runs and methods. Keep the scaling setting, evaluation seed, neighborhood sizes, preprocessing, and cell subset fixed for a controlled comparison.

Training and evaluation use separate random seeds. Run multiple explicitly recorded training seeds, retain per-run scores, and report both the aggregation rule and variability. Fixed seeds do not guarantee bitwise equality across hardware and library versions.
