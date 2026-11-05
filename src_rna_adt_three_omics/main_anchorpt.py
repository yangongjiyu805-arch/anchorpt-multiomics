"""Run two- or three-modality AnchorPT from an environment-based configuration.

The entry point validates input metadata and feature masks, constructs the model
runner, dispatches the requested training workflow, and exports embeddings with
the effective parameters required for reproduction.
"""
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import numpy as np
import scanpy as sc
import torch

# Make the bundled package importable in direct and scheduled executions.
src_dir = os.environ.get("ANCHORPT_SRC_DIR", str(Path(__file__).parents[2]))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from anchorpt.runner import AnchorPTRunner


def _bool_env(key, default=False):
    val = os.environ.get(key, "")
    if val.lower() in ("true", "1", "yes"):
        return True
    if val.lower() in ("false", "0", "no"):
        return False
    return default


def _int_env(key, default):
    return int(os.environ.get(key, str(default)))


def _float_env(key, default):
    return float(os.environ.get(key, str(default)))


def _str_env(key, default=""):
    return os.environ.get(key, default)


def _list_int_env(key, default):
    val = os.environ.get(key, "")
    if not val.strip():
        return default
    return [int(x) for x in re.findall(r"-?\d+", val)]


def _list_float_env(key, default=None):
    val = os.environ.get(key, "")
    if not val.strip():
        return default
    return [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", val)]


def _default_manifest_path(params_json):
    params_path = Path(params_json)
    return params_path.with_name(params_path.stem + "_method_manifest.json")



def _build_method_manifest(params):
    return {
        "primary_embedding": "latent_shared",
        "primary_embedding_definition": "Shared encoder latent mean after Stage-II training",
        "selection_policy": {
            "uses_latent_shared_only": True,
            "latent_shared_role": "final_shared_encoder_embedding",
            "latent_concat_role": "diagnostic_only_not_for_selection",
            "leiden_role": "supplementary_diagnostic_not_in_total",
        },
        "main_model": [
            "strict_unpaired_label_free_training",
            "shared_specific_latent_decomposition",
            "shared_reconstruction",
            "AnchorPT_transport_conditioned_pivotal_anchor_contrastive_alignment",
        ],
        "stability_regularizers": {
            "stage1_knn_graph_weight": params.get("stage1_knn_graph_weight"),
            "role": "Stage-I KNN graph stabilization",
        },
        "stage2_output_blend": {
            "value": params.get("stage2_output_blend"),
            "main_configuration_interpretation": (
                "The default configuration uses blend=1.0, so the final shared embedding is the Stage-II encoder output. "
                "Anchor-preserving blend is kept as an ablation/stabilization option, "
                "not as a primary method component."
            ),
        },
        "diagnostic_exports": [
            "latent_shared_stage1_anchor",
            "latent_shared_stage2_raw",
        ],
        "label_free_assertions": {
            "strict_unpaired": params.get("strict_unpaired"),
            "use_training_labels": params.get("use_labels"),
            "batch_key": params.get("batch_key"),
            "celltype_key": params.get("celltype_key"),
        },
    }


def main():
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print("AnchorPT training entrypoint")
        print("Configure runs with ANCHORPT_* environment variables.")
        print("Required for real training: ANCHORPT_INPUT_H5AD and writable output paths.")
        return 0

    # Read paths, data keys, architecture settings, and optimization controls.
    input_h5ad = _str_env("ANCHORPT_INPUT_H5AD")
    output_h5ad = _str_env("ANCHORPT_OUTPUT_H5AD", "output.h5ad")
    params_json = _str_env("ANCHORPT_PARAMS_USED_JSON", "params.json")
    manifest_json = _str_env("ANCHORPT_METHOD_MANIFEST_JSON", "")

    input_layer = _str_env("ANCHORPT_INPUT_LAYER", "train_input")
    modality_key = _str_env("ANCHORPT_MODALITY_KEY", "modality")
    batch_key = _str_env("ANCHORPT_BATCH_KEY", "")
    celltype_key = _str_env("ANCHORPT_CELLTYPE_KEY", "")
    distribution = _str_env("ANCHORPT_DISTRIBUTION", "Normal_positive")
    use_feature_mask = _bool_env("ANCHORPT_USE_FEATURE_MASK", False)
    use_labels = _bool_env("ANCHORPT_USE_TRAINING_LABELS", False)
    strict_unpaired = _bool_env("ANCHORPT_STRICT_UNPAIRED", True)
    # Keep the default workflow free of batch and biological-label supervision.
    if strict_unpaired and (batch_key or celltype_key or use_labels):
        raise ValueError(
            "ANCHORPT_STRICT_UNPAIRED=true forbids batch_key, celltype_key, and training labels. "
            "Set ANCHORPT_STRICT_UNPAIRED=false only for explicit ablations."
        )

    latent_shared = _int_env("ANCHORPT_LATENT_SHARED", 48)
    latent_specific = _int_env("ANCHORPT_LATENT_SPECIFIC", 20)
    hidden_layers = _list_int_env("ANCHORPT_HIDDEN_LAYERS", [128, 128])
    dropout_rate = _float_env("ANCHORPT_DROPOUT_RATE", 0.05)
    multi_encoder = _bool_env("ANCHORPT_MULTI_ENCODER", False)
    decoder_arch = _str_env("ANCHORPT_DECODER_ARCH", "conditional").lower()
    decoder_condition_on_modality = _bool_env("ANCHORPT_DECODER_CONDITION_ON_MODALITY", True)
    encoder_covariates = _bool_env("ANCHORPT_ENCODER_COVARIATES", False)
    encoder_normalization = _str_env("ANCHORPT_ENCODER_NORMALIZATION", "") or None
    decoder_normalization = _str_env("ANCHORPT_DECODER_NORMALIZATION", "") or None

    beta = _float_env("ANCHORPT_BETA", 1.25)
    shared_recon_weight = _float_env("ANCHORPT_SHARED_RECON_WEIGHT", 1.0)

    epoch_num = _int_env("ANCHORPT_EPOCH_NUM", 100)
    batch_size = _int_env("ANCHORPT_BATCH_SIZE", 128)
    lr = _float_env("ANCHORPT_LR", 7e-4)
    adaptlr = _bool_env("ANCHORPT_ADAPTLR", True)
    patience = _int_env("ANCHORPT_PATIENCE", 25)
    balanced_modality_batches = _bool_env("ANCHORPT_BALANCED_MODALITY_BATCHES", True)
    min_samples_per_modality_batch = _int_env("ANCHORPT_MIN_SAMPLES_PER_MODALITY_BATCH", 16)
    seed = int(os.environ.get("ANCHORPT_RANDOM_SEED", os.environ.get("ANCHORPT_SEED", "42")))

    training_mode = _str_env("ANCHORPT_TRAINING_MODE", "two_stage").lower()
    stage1_epoch_num = _int_env("ANCHORPT_STAGE1_EPOCH_NUM", epoch_num)
    stage2_epoch_num = _int_env("ANCHORPT_STAGE2_EPOCH_NUM", 80)
    stage2_lr = _float_env("ANCHORPT_STAGE2_LR", 2e-4)
    stage2_base_weight = _float_env('ANCHORPT_STAGE2_BASE_WEIGHT', 1.0)
    stage2_output_blend = _float_env("ANCHORPT_STAGE2_OUTPUT_BLEND", 1.0)
    stage2_trainable = _str_env("ANCHORPT_STAGE2_TRAINABLE", "shared_encoder")
    stage1_ckpt_path = _str_env("ANCHORPT_STAGE1_CKPT_PATH", "")
    stage2_ckpt_path = _str_env("ANCHORPT_STAGE2_CKPT_PATH", "")
    pivotal_anchor_weight = _float_env("ANCHORPT_PIVOTAL_ANCHOR_WEIGHT", 0.15)
    pivotal_anchor_num = _int_env("ANCHORPT_PIVOTAL_ANCHOR_NUM", 16)
    pivotal_anchor_init = _str_env("ANCHORPT_PIVOTAL_ANCHOR_INIT", "stage1_kmeans")
    anchorpt_assignment_temperature = _float_env("ANCHORPT_ASSIGNMENT_TEMPERATURE", 0.5)
    anchorpt_sinkhorn_epsilon = _float_env("ANCHORPT_SINKHORN_EPSILON", 0.05)
    anchorpt_sinkhorn_rho = _float_env("ANCHORPT_SINKHORN_RHO", 1.0)
    anchorpt_sinkhorn_iters = _int_env("ANCHORPT_SINKHORN_ITERS", 80)
    anchorpt_ema_momentum = _float_env("ANCHORPT_EMA_MOMENTUM", 0.98)
    anchorpt_contrastive_temperature = _float_env("ANCHORPT_CONTRASTIVE_TEMPERATURE", 0.08)
    anchorpt_contrastive_weight = _float_env("ANCHORPT_CONTRASTIVE_WEIGHT", 0.20)
    anchorpt_transport_cost_logit_weight = _float_env("ANCHORPT_TRANSPORT_COST_LOGIT_WEIGHT", 2.50)
    anchorpt_use_global_anchor_transport = _bool_env("ANCHORPT_USE_GLOBAL_ANCHOR_TRANSPORT", True)
    anchorpt_modality_pair_weights = _list_float_env("ANCHORPT_MODALITY_PAIR_WEIGHTS", None)
    stage1_knn_graph_weight = _float_env("ANCHORPT_STAGE1_KNN_GRAPH_WEIGHT", 0.05)
    stage1_knn_graph_k = _int_env("ANCHORPT_STAGE1_KNN_GRAPH_K", 8)
    stage1_knn_graph_max_cells = _int_env("ANCHORPT_STAGE1_KNN_GRAPH_MAX_CELLS", 128)
    stage2_pivotal_anchor_warmup_epochs = _int_env("ANCHORPT_STAGE2_PIVOTAL_ANCHOR_WARMUP_EPOCHS", 5)

    # Apply one configured seed across model fitting, data splitting, and anchor construction.
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load the configured matrix before constructing and fitting the integration runner.
    print(f"Loading data from {input_h5ad}")
    adata = sc.read_h5ad(input_h5ad)
    print(f"Data shape: {adata.shape}")

    layer = input_layer if input_layer else None
    ct_key = celltype_key if (celltype_key and use_labels) else None

    # Validate modality-specific feature indices before constructing reconstruction masks.
    feature_list = None
    if use_feature_mask:
        stored_feature_list = adata.uns.get("feature_list")
        if not isinstance(stored_feature_list, dict):
            raise ValueError(
                "ANCHORPT_USE_FEATURE_MASK=true requires adata.uns['feature_list'] as a dictionary"
            )
        modality_labels = sorted(adata.obs[modality_key].astype(str).unique().tolist())
        missing_labels = [label for label in modality_labels if label not in stored_feature_list]
        extra_labels = [label for label in stored_feature_list if label not in modality_labels]
        if missing_labels or extra_labels:
            raise ValueError(
                f"feature_list/modality mismatch: missing={missing_labels}, extra={extra_labels}"
            )
        feature_list = {}
        for label in modality_labels:
            indices = np.asarray(stored_feature_list[label], dtype=np.int64).reshape(-1)
            if indices.size == 0:
                raise ValueError(f"feature_list[{label!r}] is empty")
            if np.unique(indices).size != indices.size:
                raise ValueError(f"feature_list[{label!r}] contains duplicate indices")
            if indices.min() < 0 or indices.max() >= adata.n_vars:
                raise ValueError(
                    f"feature_list[{label!r}] indices outside [0, {adata.n_vars})"
                )
            feature_list[label] = indices.tolist()
        print(f"Feature masks enabled: { {k: len(v) for k, v in feature_list.items()} }")

    integration = AnchorPTRunner(
        data=adata,
        layer=layer,
        modality_key=modality_key,
        batch_key=batch_key if batch_key else None,
        celltype_key=ct_key,
        distribution=distribution,
        feature_list=feature_list,
        strict_unpaired=strict_unpaired,
    )

    integration.setup(
        hidden_layers=hidden_layers,
        latent_dim_shared=latent_shared,
        latent_dim_specific=latent_specific,
        dropout_rate=dropout_rate,
        beta=beta,
        multi_encoder=multi_encoder,
        encoder_covariates=encoder_covariates,
        shared_recon_weight=shared_recon_weight,
        decoder_arch=decoder_arch,
        decoder_condition_on_modality=decoder_condition_on_modality,
        encoder_normalization=encoder_normalization,
        decoder_normalization=decoder_normalization,
    )

    if training_mode == "base":
        integration.train(
            epoch_num=epoch_num,
            batch_size=batch_size,
            lr=lr,
            adaptlr=adaptlr,
            early_stopping=True,
            patience=patience,
            random_state=seed,
            balanced_modality_batches=balanced_modality_batches,
            min_samples_per_modality_batch=min_samples_per_modality_batch,
        )
    elif training_mode == "two_stage":
        integration.train_two_stage(
            stage1_epoch_num=stage1_epoch_num,
            stage2_epoch_num=stage2_epoch_num,
            batch_size=batch_size,
            lr=lr,
            stage2_lr=stage2_lr,
            stage2_base_weight=stage2_base_weight,
            adaptlr=adaptlr,
            early_stopping=True,
            patience=patience,
            random_state=seed,
            balanced_modality_batches=balanced_modality_batches,
            min_samples_per_modality_batch=min_samples_per_modality_batch,
            pivotal_anchor_weight=pivotal_anchor_weight,
            num_pivotal_anchors=pivotal_anchor_num,
            pivotal_anchor_init=pivotal_anchor_init,
            stage2_trainable=stage2_trainable,
            anchorpt_assignment_temperature=anchorpt_assignment_temperature,
            anchorpt_sinkhorn_epsilon=anchorpt_sinkhorn_epsilon,
            anchorpt_sinkhorn_rho=anchorpt_sinkhorn_rho,
            anchorpt_sinkhorn_iters=anchorpt_sinkhorn_iters,
            anchorpt_ema_momentum=anchorpt_ema_momentum,
            anchorpt_contrastive_temperature=anchorpt_contrastive_temperature,
            anchorpt_contrastive_weight=anchorpt_contrastive_weight,
            anchorpt_transport_cost_logit_weight=anchorpt_transport_cost_logit_weight,
            anchorpt_use_global_anchor_transport=anchorpt_use_global_anchor_transport,
            anchorpt_modality_pair_weights=anchorpt_modality_pair_weights,
            stage1_knn_graph_weight=stage1_knn_graph_weight,
            stage1_knn_graph_k=stage1_knn_graph_k,
            stage1_knn_graph_max_cells=stage1_knn_graph_max_cells,
            stage2_pivotal_anchor_warmup_epochs=stage2_pivotal_anchor_warmup_epochs,
            stage2_output_blend=stage2_output_blend,
            stage1_ckpt_path=stage1_ckpt_path if stage1_ckpt_path else None,
            stage2_ckpt_path=stage2_ckpt_path if stage2_ckpt_path else None,
        )
    else:
        raise ValueError("ANCHORPT_TRAINING_MODE must be 'base' or 'two_stage'.")

    integration.inference(n_samples=1)

    # Persist embeddings, effective parameters, and the method-role manifest.
    result_adata = integration.get_adata()
    Path(output_h5ad).parent.mkdir(parents=True, exist_ok=True)
    result_adata.write_h5ad(output_h5ad)
    print(f"Results saved to {output_h5ad}")

    params = {
        "latent_shared": latent_shared,
        "latent_specific": latent_specific,
        "hidden_layers": hidden_layers,
        "beta": beta,
        "shared_recon_weight": shared_recon_weight,
        "dropout_rate": dropout_rate,
        "multi_encoder": multi_encoder,
        "encoder_covariates": encoder_covariates,
        "decoder_arch": decoder_arch,
        "decoder_condition_on_modality": decoder_condition_on_modality,
        "encoder_normalization": encoder_normalization,
        "decoder_normalization": decoder_normalization,
        "epoch_num": epoch_num,
        "batch_size": batch_size,
        "lr": lr,
        "adaptlr": adaptlr,
        "patience": patience,
        "balanced_modality_batches": balanced_modality_batches,
        "min_samples_per_modality_batch": min_samples_per_modality_batch,
        "training_mode": training_mode,
        "stage1_epoch_num": stage1_epoch_num,
        "stage2_epoch_num": stage2_epoch_num,
        "stage2_lr": stage2_lr,
        'stage2_base_weight': stage2_base_weight,
        "stage2_output_blend": stage2_output_blend,
        "stage2_trainable": stage2_trainable,
        "stage1_ckpt_path": stage1_ckpt_path,
        "stage2_ckpt_path": stage2_ckpt_path,
        "pivotal_anchor_weight": pivotal_anchor_weight,
        "pivotal_anchor_num": pivotal_anchor_num,
        "pivotal_anchor_init": pivotal_anchor_init,
        "anchorpt_assignment_temperature": anchorpt_assignment_temperature,
        "anchorpt_sinkhorn_epsilon": anchorpt_sinkhorn_epsilon,
        "anchorpt_sinkhorn_rho": anchorpt_sinkhorn_rho,
        "anchorpt_sinkhorn_iters": anchorpt_sinkhorn_iters,
        "anchorpt_ema_momentum": anchorpt_ema_momentum,
        "anchorpt_contrastive_temperature": anchorpt_contrastive_temperature,
        "anchorpt_contrastive_weight": anchorpt_contrastive_weight,
        "anchorpt_transport_cost_logit_weight": anchorpt_transport_cost_logit_weight,
        "anchorpt_use_global_anchor_transport": anchorpt_use_global_anchor_transport,
        "anchorpt_modality_pair_weights": anchorpt_modality_pair_weights,
        "stage1_knn_graph_weight": stage1_knn_graph_weight,
        "stage1_knn_graph_k": stage1_knn_graph_k,
        "stage1_knn_graph_max_cells": stage1_knn_graph_max_cells,
        "stage2_pivotal_anchor_warmup_epochs": stage2_pivotal_anchor_warmup_epochs,
        "seed": seed,
        "distribution": distribution,
        "use_feature_mask": use_feature_mask,
        "feature_list_sizes": ({k: len(v) for k, v in feature_list.items()} if feature_list else None),
        "use_labels": use_labels,
        "strict_unpaired": strict_unpaired,
        "modality_key": modality_key,
        "batch_key": batch_key,
        "celltype_key": celltype_key,
        "input_layer": input_layer,
        "input_h5ad": input_h5ad,
    }
    if integration.modality_num == 2:
        params.pop("anchorpt_modality_pair_weights")
    elif integration.modality_num != 3:
        raise ValueError(f"Combined Stage-2 entrypoint supports 2 or 3 modalities; got {integration.modality_num}.")
    Path(params_json).parent.mkdir(parents=True, exist_ok=True)
    Path(params_json).write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Params saved to {params_json}")

    manifest_path = Path(manifest_json) if manifest_json else _default_manifest_path(params_json)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    method_manifest = _build_method_manifest(params)
    manifest_path.write_text(json.dumps(method_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Method manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
