# Evaluation and reproducibility

## Run records

Keep the commit, environment, input/annotation versions, splits, seeds, preprocessing arguments, `params.json`, training logs, and scores. Use disjoint cells for unpaired benchmarks and report label-based split stratification separately.

Datasets, reference annotations, checkpoints, and benchmark result tables are not bundled.

## Embeddings

| Key in `obsm` | Role |
| :--- | :--- |
| `latent_shared` | Primary integration embedding |
| `latent_specific` | Modality-specific factors |
| `latent_concat` | Concatenated diagnostic |
| `latent_shared_stage1_anchor` | Stage-I reference, when available |
| `latent_shared_stage2_raw` | Pre-blend output when blending is enabled |

The default blend is 1.0; lower values mix Stage-II output with the Stage-I reference. The multimodal runner also exports CORAL diagnostics under `latent_shared_coral_avg_s055` (two modalities) or `latent_shared_coral_avg_s085` (three modalities). Use `latent_shared` for primary evaluation.

## Evaluation

[Example command](quickstart.md#4-evaluate). Both runners use the same evaluator.

| Option | Default |
| :--- | :--- |
| `--embedding-key` / `--label-key` | `latent_shared` / `cell_type` |
| `--modality-key` / `--batch-key` | `modality` / `none` |
| `--random-state` / `--ilisi-knn-m` | `0` / `90` |
| `--pre-embedding-key` | Unset; PCA from `.X` |
| `--min-max-scale` | Off |
| `--format` | `tsv`; also CSV or JSON |

Enable batch metrics only for an interpretable nuisance variable. Inputs must support neighbor counts of 15, 50, 90, and the selected modality-iLISI count.

## Aggregate scores

Each group is an arithmetic mean:

| Group | Metrics |
| :--- | :--- |
| Bio | Isolated labels, Leiden NMI/ARI, label silhouette, cLISI |
| Modality | Silhouette, iLISI, kBET, graph connectivity, PCR |
| Batch | Silhouette, iLISI, kBET, PCR |

With batch metrics: `Total = 0.4 × Bio + 0.3 × Modality + 0.3 × Batch`.

Without batch metrics: `Total = 0.6 × Bio + 0.4 × Modality`.

Leiden NMI/ARI contribute to Bio and Total. Failed metrics become missing values; warnings are suppressed, nonfinite outputs are omitted, and means use surviving metrics. Check that compared runs contain the same metric inventory.

Keep evaluation settings and cell subsets fixed across methods. Training and evaluation seeds are separate; report per-run scores and variability across training seeds.
