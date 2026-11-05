"""Coordinate model construction, optimization, and export for two or three modalities.

The runner encodes observation metadata, manages stratified data partitions,
initializes transport anchors from pretrained shared means, and records both the
primary representation and an optional moment-matched diagnostic embedding.
"""

import numpy as np
import torch
import torch.utils.data as Data
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import anndata as ad

from .data import OmicsDataset
from .model import AnchorPTNet
from .train import encode_dataset, train_stage1, train_stage2
from .objective import AnchorPTObjective

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


def _psd_power(matrix, power, ridge=1e-4):
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = 0.5 * (matrix + matrix.T)
    dim = matrix.shape[0]
    scale = max(float(np.trace(matrix)) / max(dim, 1), 1e-8)
    values, vectors = np.linalg.eigh(matrix + ridge * scale * np.eye(dim))
    values = np.clip(values, ridge * scale, None)
    return (vectors * np.power(values, power)) @ vectors.T


# Match each modality to the equal-weight average mean and covariance.
def _coral_to_equal_modality_average(latent, groups):
    latent64 = np.asarray(latent, dtype=np.float64)
    group_values = np.asarray(groups).astype(str)
    levels = sorted(np.unique(group_values).tolist())
    if len(levels) < 2:
        raise ValueError("Post-hoc CORAL requires at least two modalities.")
    means = []
    covariances = []
    for level in levels:
        block = latent64[group_values == level]
        if block.shape[0] < 2:
            raise ValueError(f"Post-hoc CORAL modality {level!r} has fewer than two cells.")
        means.append(block.mean(axis=0))
        covariances.append(np.cov(block, rowvar=False, ddof=1))
    target_mean = np.mean(means, axis=0)
    target_covariance = np.mean(covariances, axis=0)
    target_sqrt = _psd_power(target_covariance, 0.5)
    calibrated = np.empty_like(latent64)
    for level, source_mean, source_covariance in zip(levels, means, covariances):
        mask = group_values == level
        source_inv_sqrt = _psd_power(source_covariance, -0.5)
        calibrated[mask] = (
            (latent64[mask] - source_mean) @ source_inv_sqrt @ target_sqrt + target_mean
        )
    return calibrated


def to_model_array(x):
    if issparse(x):
        return x.tocsr(copy=True)
    if isinstance(x, np.ndarray):
        return x.copy()
    raise TypeError(f"Unsupported type: {type(x)}")


class AnchorPTRunner:
    def __init__(
        self,
        data,
        layer=None,
        modality_key="modality",
        batch_key=None,
        celltype_key=None,
        distribution="Normal_positive",
        mask_key=None,
        feature_list=None,
        auto_make_obs_unique=True,
        strict_unpaired=True,
    ):
        super().__init__()
        if strict_unpaired and (batch_key is not None or celltype_key is not None):
            raise ValueError(
                "strict_unpaired=True forbids batch_key and celltype_key. "
                "Set strict_unpaired=False only for explicit ablations."
            )
        self.strict_unpaired = strict_unpaired
        self.anchorpt = None
        self.stage1_anchor_latent = None

        if isinstance(data, list) and isinstance(data[0], ad.AnnData):
            self.adata = ad.concat(data, axis="obs", join="inner", label="modality")
        elif isinstance(data, ad.AnnData):
            self.adata = data
        else:
            raise ValueError("Wrong type of data!")

        if auto_make_obs_unique and (not self.adata.obs_names.is_unique):
            self.adata.obs_names_make_unique()

        if layer is None:
            self.data = to_model_array(self.adata.X)
        else:
            self.data = to_model_array(self.adata.layers[layer])

        # Encode configured observation fields as categorical model inputs.
        label_encoder = LabelEncoder()
        onehot_encoder = OneHotEncoder(sparse_output=False)

        self.modality_label = self.adata.obs[modality_key].to_numpy()
        self.modality = label_encoder.fit_transform(self.modality_label)
        self.modality_ordered = [label for label in label_encoder.classes_]
        self.modality = onehot_encoder.fit_transform(self.modality.reshape(-1, 1))

        if celltype_key is None:
            self.celltype = None
            self.celltype_ordered = None
        else:
            self.celltype_label = self.adata.obs[celltype_key].to_numpy()
            self.celltype = label_encoder.fit_transform(self.celltype_label)
            self.celltype_ordered = [label for label in label_encoder.classes_]
            self.celltype = onehot_encoder.fit_transform(self.celltype.reshape(-1, 1))

        if batch_key is None:
            self.covariates = None
            self.covariates_ordered = None
        else:
            self.covariates_label = self.adata.obs[batch_key].to_numpy()
            self.covariates = label_encoder.fit_transform(self.covariates_label)
            self.covariates_ordered = [label for label in label_encoder.classes_]
            self.covariates = onehot_encoder.fit_transform(self.covariates.reshape(-1, 1))

        self.modality_num = self.modality.shape[1]
        self.celltype_num = self.celltype.shape[1] if self.celltype is not None else 0
        self.covariates_dim = self.covariates.shape[1] if self.covariates is not None else 0

        if mask_key is None:
            self.mask = self.modality
            self.mask_num = self.modality_num
            self.mask_ordered = self.modality_ordered
        else:
            self.mask_label = self.adata.obs[mask_key].to_numpy()
            self.mask = label_encoder.fit_transform(self.mask_label)
            self.mask_ordered = [label for label in label_encoder.classes_]
            self.mask = onehot_encoder.fit_transform(self.mask.reshape(-1, 1))
            self.mask_num = self.mask.shape[1]

        if feature_list is not None:
            self.feat_mask = torch.zeros(self.mask_num, self.data.shape[1])
            feature_list_ordered = [feature_list[label] for label in self.mask_ordered]
            for i, feat_idx in enumerate(feature_list_ordered):
                self.feat_mask[i, feat_idx] = 1
        else:
            self.feat_mask = None

        self.distribution = distribution
        if distribution not in ["ZINB", "NB", "Normal", "Normal_positive"]:
            raise ValueError("Distribution not recognized!")

    def setup(
        self,
        hidden_layers=None,
        latent_dim_shared=48,
        latent_dim_specific=20,
        dropout_rate=0.05,
        beta=1.25,
        multi_encoder=False,
        encoder_covariates=False,
        shared_recon_weight=1.0,
        decoder_arch="conditional",
        decoder_condition_on_modality=True,
        encoder_normalization=None,
        decoder_normalization=None,
        device=None,
    ):
        if hidden_layers is None:
            hidden_layers = [128, 128]

        self.input_dim = self.data.shape[1]
        self.hidden_layers = hidden_layers
        self.latent_dim_shared = latent_dim_shared
        self.latent_dim_specific = latent_dim_specific
        self.dropout_rate = dropout_rate
        self.beta = beta
        self.multi_encoder = multi_encoder
        self.encoder_covariates = encoder_covariates
        self.shared_recon_weight = shared_recon_weight
        self.decoder_arch = decoder_arch
        self.decoder_condition_on_modality = decoder_condition_on_modality
        self.encoder_normalization = encoder_normalization
        self.decoder_normalization = decoder_normalization

        if device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = device
        print("using " + str(self.device))

        self.model = AnchorPTNet(
            device=self.device,
            input_dim=self.input_dim,
            modality_num=self.modality_num,
            covariate_dim=self.covariates_dim,
            celltype_num=self.celltype_num,
            layer_dims=self.hidden_layers,
            latent_dim_shared=self.latent_dim_shared,
            latent_dim_specific=self.latent_dim_specific,
            dropout_rate=self.dropout_rate,
            beta=self.beta,
            feat_mask=self.feat_mask,
            distribution=self.distribution,
            encoder_covariates=self.encoder_covariates,
            multi_encoder=self.multi_encoder,
            shared_recon_weight=self.shared_recon_weight,
            decoder_arch=self.decoder_arch,
            decoder_condition_on_modality=self.decoder_condition_on_modality,
            encoder_normalization=self.encoder_normalization,
            decoder_normalization=self.decoder_normalization,
        ).to(self.device)

        self.train_dataset = OmicsDataset(self.data, self.covariates, self.modality, self.mask, self.celltype)

    def train(
        self,
        epoch_num=100,
        batch_size=128,
        lr=7e-4,
        accumulation_steps=1,
        adaptlr=True,
        valid_prop=0.1,
        early_stopping=True,
        patience=25,
        weighted=False,
        tensorboard=False,
        savepath="./",
        random_state=42,
        balanced_modality_batches=True,
        min_samples_per_modality_batch=16,
    ):
        if tensorboard and SummaryWriter is None:
            print("TensorBoard is not available in current environment; continue without it.")
            self.writer = None
        else:
            self.writer = SummaryWriter(savepath) if tensorboard else None

        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.lr = lr
        self.accumulation_steps = accumulation_steps
        self.adaptlr = adaptlr
        # Preserve modality proportions in the training and validation partitions.
        if valid_prop > 0:
            train_indices, valid_indices = train_test_split(
                np.arange(len(self.train_dataset)),
                test_size=valid_prop,
                stratify=self.modality.argmax(-1),
                random_state=random_state,
            )
            train_dataset = Data.Subset(self.train_dataset, train_indices)
            valid_dataset = Data.Subset(self.train_dataset, valid_indices)
        else:
            train_dataset, valid_dataset = self.train_dataset, self.train_dataset
            train_indices = np.arange(len(self.train_dataset))
            valid_indices = np.arange(len(self.train_dataset))

        self.num_batch = max(1, len(train_dataset) // self.batch_size)

        print("Training start!")
        if weighted:
            weights = 1.0 / np.bincount(self.modality.argmax(-1))
            sample_weights = weights[self.modality.argmax(-1)]
            sample_weights = sample_weights[train_indices]
        else:
            sample_weights = None
        modality_labels = self.modality.argmax(-1)[train_indices]
        validate_modality_labels = self.modality.argmax(-1)[valid_indices]

        train_stage1(
            self.device,
            self.writer,
            train_dataset,
            valid_dataset,
            self.model,
            self.epoch_num,
            self.batch_size,
            self.num_batch,
            self.lr,
            accumulation_steps=self.accumulation_steps,
            adaptlr=self.adaptlr,
            early_stopping=early_stopping,
            patience=patience,
            sample_weights=sample_weights,
            modality_labels=modality_labels,
            validate_modality_labels=validate_modality_labels,
            balanced_modality_batches=balanced_modality_batches,
            min_samples_per_modality_batch=min_samples_per_modality_batch,
            random_state=random_state,
        )

        if self.writer is not None:
            self.writer.close()
        print("Training finished!")

    def train_two_stage(
        self,
        stage1_epoch_num=100,
        stage2_epoch_num=40,
        batch_size=128,
        lr=7e-4,
        stage2_lr=2e-4,
        stage2_base_weight=1.0,
        accumulation_steps=1,
        adaptlr=True,
        valid_prop=0.1,
        early_stopping=True,
        patience=25,
        weighted=False,
        tensorboard=False,
        savepath="./",
        random_state=42,
        balanced_modality_batches=True,
        min_samples_per_modality_batch=16,
        pivotal_anchor_weight=0.15,
        num_pivotal_anchors=16,
        pivotal_anchor_init="stage1_kmeans",
        stage2_trainable="shared_encoder",
        anchorpt_assignment_temperature=0.5,
        anchorpt_sinkhorn_epsilon=0.05,
        anchorpt_sinkhorn_rho=1.0,
        anchorpt_sinkhorn_iters=80,
        anchorpt_ema_momentum=0.98,
        anchorpt_contrastive_temperature=0.08,
        anchorpt_contrastive_weight=0.20,
        anchorpt_transport_cost_logit_weight=2.50,
        anchorpt_use_global_anchor_transport=True,
        anchorpt_modality_pair_weights=None,
        stage1_knn_graph_weight=0.05,
        stage1_knn_graph_k=8,
        stage1_knn_graph_max_cells=128,
        stage2_pivotal_anchor_warmup_epochs=5,
        stage2_output_blend=1.0,
        stage1_ckpt_path=None,
        stage2_ckpt_path=None,
    ):
        if self.modality_num not in (2, 3):
            raise ValueError(f"Combined Stage 2 supports exactly 2 or 3 modalities; got {self.modality_num}.")
        self.stage2_output_blend = float(stage2_output_blend)
        if not 0.0 <= self.stage2_output_blend <= 1.0:
            raise ValueError("stage2_output_blend must be in [0, 1].")
        if stage2_lr is None:
            stage2_lr = lr

        print("Stage 1 pretraining start.")
        self.train(
            epoch_num=stage1_epoch_num,
            batch_size=batch_size,
            lr=lr,
            accumulation_steps=accumulation_steps,
            adaptlr=adaptlr,
            valid_prop=valid_prop,
            early_stopping=early_stopping,
            patience=patience,
            weighted=weighted,
            tensorboard=tensorboard,
            savepath=savepath,
            random_state=random_state,
            balanced_modality_batches=balanced_modality_batches,
            min_samples_per_modality_batch=min_samples_per_modality_batch,
        )
        print("Stage 1 pretraining finished.")

        if stage1_ckpt_path:
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "modality_ordered": self.modality_ordered,
                    "latent_dim_shared": self.latent_dim_shared,
                    "latent_dim_specific": self.latent_dim_specific,
                    "random_state": random_state,
                },
                stage1_ckpt_path,
            )
            print(f"Stage 1 checkpoint saved to {stage1_ckpt_path}")

        if valid_prop > 0:
            train_indices, valid_indices = train_test_split(
                np.arange(len(self.train_dataset)),
                test_size=valid_prop,
                stratify=self.modality.argmax(-1),
                random_state=random_state,
            )
        else:
            train_indices = np.arange(len(self.train_dataset))
            valid_indices = np.arange(len(self.train_dataset))

        print("Initializing AnchorPT pivotal anchors from Stage 1 shared latent.")
        z_shared, _ = self.inference(n_samples=1, batch_size=batch_size, update=False, returns=True)
        self.stage1_anchor_latent = np.asarray(z_shared, dtype=np.float32)
        # Keep one pair for two modalities and use RNA-centered pairs for three.
        if self.modality_num == 2:
            self.anchorpt = AnchorPTObjective(
                latent_dim=self.latent_dim_shared,
                num_pivotal_anchors=num_pivotal_anchors,
                assignment_temperature=anchorpt_assignment_temperature,
                sinkhorn_epsilon=anchorpt_sinkhorn_epsilon,
                sinkhorn_rho=anchorpt_sinkhorn_rho,
                sinkhorn_iters=anchorpt_sinkhorn_iters,
                ema_momentum=anchorpt_ema_momentum,
                contrastive_temperature=anchorpt_contrastive_temperature,
                contrastive_weight=anchorpt_contrastive_weight,
                transport_cost_logit_weight=anchorpt_transport_cost_logit_weight,
                use_global_anchor_transport=anchorpt_use_global_anchor_transport,
            ).to(self.device)
        else:
            self.anchorpt = AnchorPTObjective(
                latent_dim=self.latent_dim_shared,
                num_pivotal_anchors=num_pivotal_anchors,
                num_modalities=self.modality_num,
                modality_names=self.modality_ordered,
                assignment_temperature=anchorpt_assignment_temperature,
                sinkhorn_epsilon=anchorpt_sinkhorn_epsilon,
                sinkhorn_rho=anchorpt_sinkhorn_rho,
                sinkhorn_iters=anchorpt_sinkhorn_iters,
                ema_momentum=anchorpt_ema_momentum,
                contrastive_temperature=anchorpt_contrastive_temperature,
                contrastive_weight=anchorpt_contrastive_weight,
                transport_cost_logit_weight=anchorpt_transport_cost_logit_weight,
                use_global_anchor_transport=anchorpt_use_global_anchor_transport,
                modality_pair_weights=anchorpt_modality_pair_weights,
            ).to(self.device)
            print(
                "Stage 2 AnchorPT RNA-hub modality pairs: "
                f"modality_ordered={self.modality_ordered}, pairs={self.anchorpt.modality_pairs}, "
                f"weights={self.anchorpt.modality_pair_weights.detach().cpu().tolist()}"
            )
        self.train_dataset = OmicsDataset(self.data, self.covariates, self.modality, self.mask, self.celltype, anchor=self.stage1_anchor_latent)
        train_dataset = Data.Subset(self.train_dataset, train_indices)
        valid_dataset = Data.Subset(self.train_dataset, valid_indices)

        if pivotal_anchor_init == "stage1_kmeans":
            self.anchorpt.initialize_from_latents(
                z_shared[train_indices],
                self.modality.argmax(-1)[train_indices],
                random_state=random_state,
            )
        elif pivotal_anchor_init != "random":
            raise ValueError("pivotal_anchor_init must be 'stage1_kmeans' or 'random'.")

        if tensorboard and SummaryWriter is None:
            print("TensorBoard is not available in current environment; continue without it.")
            writer = None
        else:
            writer = SummaryWriter(savepath) if tensorboard else None

        print("Stage 2 pivotal-anchor transport alignment start.")
        train_stage2(
            self.device,
            writer,
            train_dataset,
            valid_dataset,
            self.model,
            self.anchorpt,
            stage2_epoch_num,
            batch_size,
            stage2_lr,
            pivotal_anchor_weight=pivotal_anchor_weight,
            stage2_base_weight=stage2_base_weight,
            adaptlr=adaptlr,
            early_stopping=early_stopping,
            patience=patience,
            trainable=stage2_trainable,
            modality_labels=self.modality.argmax(-1)[train_indices],
            balanced_modality_batches=balanced_modality_batches,
            min_samples_per_modality_batch=min_samples_per_modality_batch,
            random_state=random_state,
            stage1_knn_graph_weight=stage1_knn_graph_weight,
            stage1_knn_graph_k=stage1_knn_graph_k,
            stage1_knn_graph_max_cells=stage1_knn_graph_max_cells,
            pivotal_anchor_warmup_epochs=stage2_pivotal_anchor_warmup_epochs,
            validate_modality_labels=self.modality.argmax(-1)[valid_indices],
        )
        if stage2_ckpt_path:
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "anchorpt_state_dict": self.anchorpt.state_dict(),
                    "modality_ordered": self.modality_ordered,
                    "latent_dim_shared": self.latent_dim_shared,
                    "latent_dim_specific": self.latent_dim_specific,
                    "random_state": random_state,
                },
                stage2_ckpt_path,
            )
            print(f"Stage 2 checkpoint saved to {stage2_ckpt_path}")
        if writer is not None:
            writer.close()
        print("Stage 2 pivotal-anchor transport alignment finished.")

    def inference(self, n_samples=1, dataset=None, batch_size=None, update=True, returns=False):
        if dataset is None:
            dataset = self.train_dataset
        if batch_size is None:
            batch_size = self.batch_size

        if n_samples > 1:
            z_shared, z_specific = zip(*[encode_dataset(self.device, dataset, self.model, batch_size) for _ in range(n_samples)])
            self.z_shared = np.mean(np.stack(z_shared, axis=0), axis=0)
            self.z_specific = np.mean(np.stack(z_specific, axis=0), axis=0)
        else:
            self.z_shared, self.z_specific = encode_dataset(self.device, dataset, self.model, batch_size)

        # Record the primary embeddings and a separate moment-matched diagnostic representation.
        if update:
            if getattr(self, 'stage1_anchor_latent', None) is not None and len(self.stage1_anchor_latent) == len(self.adata):
                self.adata.obsm['latent_shared_stage1_anchor'] = np.asarray(self.stage1_anchor_latent, dtype=np.float32)
            if (
                getattr(self, 'stage1_anchor_latent', None) is not None
                and len(self.stage1_anchor_latent) == len(self.adata)
                and getattr(self, 'stage2_output_blend', 1.0) < 1.0
            ):
                self.adata.obsm['latent_shared_stage2_raw'] = np.asarray(self.z_shared, dtype=np.float32)
                anchor_shared = np.asarray(self.stage1_anchor_latent, dtype=np.float32)
                blend = float(self.stage2_output_blend)
                self.z_shared = blend * self.z_shared + (1.0 - blend) * anchor_shared
            self.z_shared = np.asarray(self.z_shared, dtype=np.float32)
            self.adata.obsm['latent_shared'] = self.z_shared
            coral_latent = _coral_to_equal_modality_average(self.z_shared, self.modality_label)
            if self.modality_num == 2:
                coral_strength = 0.55
                coral_key = 'latent_shared_coral_avg_s055'
            else:
                coral_strength = 0.85
                coral_key = 'latent_shared_coral_avg_s085'
            self.adata.obsm[coral_key] = (
                (1.0 - coral_strength) * self.z_shared + coral_strength * coral_latent
            ).astype(np.float32)
            self.adata.obsm['latent_specific'] = self.z_specific
            self.adata.obsm['latent_concat'] = np.concatenate([self.z_shared, self.z_specific], axis=1)
            print('All results recorded in adata.')

        if returns:
            return self.z_shared, self.z_specific

    def get_adata(self):
        return self.adata

