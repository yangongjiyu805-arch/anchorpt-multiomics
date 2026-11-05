"""Provide optimization and embedding utilities for multimodal AnchorPT."""

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler


class BalanceSampler:

    def __init__(self, modality_labels, batch_size, num_batches=None, random_state=42, min_samples_per_modality=16):
        self.modality_labels = np.asarray(modality_labels)
        self.batch_size = int(batch_size)
        self.min_samples_per_modality = int(min_samples_per_modality)
        self.modalities = np.unique(self.modality_labels)
        self.num_modalities = len(self.modalities)
        if self.num_modalities == 0:
            raise ValueError("No modalities found for balanced batch sampling.")
        if self.batch_size < self.num_modalities:
            raise ValueError(
                f"batch_size={self.batch_size} is smaller than modality_num={self.num_modalities}; "
                "balanced modality batches need at least one sample per modality in a batch."
            )
        if self.batch_size < self.min_samples_per_modality * self.num_modalities:
            raise ValueError(
                f"batch_size={self.batch_size} is too small for min_samples_per_modality="
                f"{self.min_samples_per_modality} and modality_num={self.num_modalities}."
            )
        self.num_batches = int(num_batches) if num_batches is not None else int(np.ceil(len(self.modality_labels) / self.batch_size))
        self.random_state = int(random_state)
        self.epoch = 0
        self.indices_by_modality = {
            modality: np.flatnonzero(self.modality_labels == modality)
            for modality in self.modalities
        }
        empty = [modality for modality, indices in self.indices_by_modality.items() if len(indices) == 0]
        if empty:
            raise ValueError(f"Empty modality groups found: {empty}")

    def __iter__(self):
        rng = np.random.default_rng(self.random_state + self.epoch)
        self.epoch += 1

        base = self.batch_size // self.num_modalities
        remainder = self.batch_size % self.num_modalities

        # Balance modalities per minibatch.
        for batch_idx in range(self.num_batches):
            modality_order = rng.permutation(self.modalities)
            batch = []
            for offset, modality in enumerate(modality_order):
                take = base + int(offset < remainder)
                pool = self.indices_by_modality[modality]
                replace = len(pool) < take
                batch.extend(rng.choice(pool, size=take, replace=replace).tolist())
            rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


class Stopper:
    def __init__(self, patience=10, delta=0.0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"Stopper counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def train_stage1(
    device,
    writer,
    train_dataset,
    validate_dataset,
    model,
    epoch_num,
    batch_size,
    num_batch,
    lr,
    accumulation_steps=1,
    adaptlr=True,
    early_stopping=True,
    patience=25,
    sample_weights=None,
    modality_labels=None,
    validate_modality_labels=None,
    balanced_modality_batches=True,
    min_samples_per_modality_batch=16,
    random_state=42,
):
    small_dataset = len(train_dataset) < batch_size
    drop_last = not small_dataset
    if balanced_modality_batches and modality_labels is not None:
        sampler = BalanceSampler(
            modality_labels=modality_labels,
            batch_size=batch_size,
            num_batches=max(1, num_batch),
            random_state=random_state,
            min_samples_per_modality=min_samples_per_modality_batch,
        )
        train_data = DataLoader(train_dataset, batch_sampler=sampler, num_workers=4, pin_memory=True)
    elif sample_weights is not None:
        sample_weights = torch.tensor(sample_weights, dtype=torch.double)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        train_data = DataLoader(train_dataset, batch_size, shuffle=False, sampler=sampler, drop_last=drop_last, num_workers=4, pin_memory=True)
    else:
        train_data = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=drop_last, num_workers=4, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if adaptlr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epoch_num * max(1, num_batch))

    stopper = Stopper(patience=patience, verbose=True) if early_stopping else None

    global_step = 0
    for epoch in range(epoch_num):
        model.train()
        total_loss = 0.0
        recon_loss = 0.0
        shared_factor_sufficiency_loss = 0.0
        kl_z = 0.0
        has_shared_factor_sufficiency = False

        optimizer.zero_grad(set_to_none=True)
        step_count = 0
        for step, batch in enumerate(train_data):
            view_data, batch_cov, omics_id, domain_id, label_cov = batch[:5]
            view_data, batch_cov, omics_id, domain_id, label_cov = view_data.to(device), batch_cov.to(device), omics_id.to(device), domain_id.to(device), label_cov.to(device)

            _, _, loss, loss_dict = model(view_data, batch_cov, omics_id, domain_id, label_cov, stage="train")
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            step_count += 1

            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()
            recon_loss += loss_dict["recon_loss"]
            if "shared_factor_sufficiency_loss" in loss_dict:
                shared_factor_sufficiency_loss += loss_dict["shared_factor_sufficiency_loss"]
                has_shared_factor_sufficiency = True
            kl_z += loss_dict["kl_z"]

            global_step += 1
            if writer is not None:
                writer.add_scalar("Loss/train", loss.item(), global_step)
                writer.add_scalar("recon_loss/train", loss_dict["recon_loss"], global_step)
                if "shared_factor_sufficiency_loss" in loss_dict:
                    writer.add_scalar("shared_factor_sufficiency_loss/train", loss_dict["shared_factor_sufficiency_loss"], global_step)
                writer.add_scalar("kl_z/train", loss_dict["kl_z"], global_step)

        if step_count > 0 and (step_count % accumulation_steps) != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        denom = max(1, step_count)
        message = "epoch {}: loss={:.4f}, recon={:.4f}".format(
            epoch + 1,
            total_loss / denom,
            recon_loss / denom,
        )
        if has_shared_factor_sufficiency:
            message += ", shared_factor_sufficiency={:.4f}".format(shared_factor_sufficiency_loss / denom)
        message += ", kl={:.4f}".format(kl_z / denom)
        print(message)

        if stopper is not None:
            validate_loss = validate_stage1(
                device,
                validate_dataset,
                model,
                batch_size,
                modality_labels=validate_modality_labels,
                balanced_modality_batches=balanced_modality_batches,
                min_samples_per_modality_batch=min_samples_per_modality_batch,
                random_state=random_state + epoch + 100_000,
            )
            stopper(validate_loss)
            if stopper.early_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break

def validate_stage1(
    device,
    validate_dataset,
    model,
    batch_size,
    modality_labels=None,
    balanced_modality_batches=True,
    min_samples_per_modality_batch=16,
    random_state=42,
):
    model.eval()
    if balanced_modality_batches and modality_labels is not None:
        sampler = BalanceSampler(
            modality_labels=modality_labels,
            batch_size=batch_size,
            num_batches=max(1, int(np.ceil(len(validate_dataset) / batch_size))),
            random_state=random_state,
            min_samples_per_modality=min_samples_per_modality_batch,
        )
        validate_data = DataLoader(validate_dataset, batch_sampler=sampler, num_workers=4, pin_memory=True)
    else:
        validate_data = DataLoader(validate_dataset, batch_size, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in validate_data:
            view_data, batch_cov, omics_id, domain_id, label_cov = batch[:5]
            view_data, batch_cov, omics_id, domain_id, label_cov = view_data.to(device), batch_cov.to(device), omics_id.to(device), domain_id.to(device), label_cov.to(device)
            _, _, loss, _ = model(view_data, batch_cov, omics_id, domain_id, label_cov, stage="eval")
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(1, n_batches)


def _set_stage2_trainable(model, trainable="shared_encoder"):
    if trainable == "all":
        for param in model.parameters():
            param.requires_grad = True
        return
    if trainable != "shared_encoder":
        raise ValueError("stage2_trainable must be 'shared_encoder' or 'all'.")

    for param in model.parameters():
        param.requires_grad = False
    for param in model.encoder_shared.parameters():
        param.requires_grad = True


def _set_stage2_mode(model, trainable="shared_encoder"):
    if trainable == "all":
        model.train()
        return
    model.eval()
    model.encoder_shared.train()


def _metric_value(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _standardized_pairwise_distances(z, eps=1e-6):
    distances = torch.cdist(z.float(), z.float()).pow(2)
    if distances.shape[0] < 2:
        return distances
    mask = ~torch.eye(distances.shape[0], dtype=torch.bool, device=distances.device)
    values = distances[mask]
    mean = values.mean()
    std = values.std(unbiased=False).clamp_min(eps)
    return (distances - mean) / std


# Preserve local neighborhood geometry from the pretrained shared embedding.
def _stage1_knn_graph_loss(z_current, z_anchor, k=8, max_cells=96, eps=1e-6):
    if z_anchor is None or z_anchor.numel() == 0 or z_current.shape[0] < 3:
        return z_current.new_zeros(())
    k = int(k)
    if k <= 0:
        return z_current.new_zeros(())
    n_cells = min(int(max_cells), z_current.shape[0]) if max_cells is not None else z_current.shape[0]
    if n_cells < 3:
        return z_current.new_zeros(())
    k_eff = min(k, n_cells - 1)
    current = z_current[:n_cells].float()
    anchor = z_anchor[:n_cells].float().detach()
    with torch.no_grad():
        anchor_dist = torch.cdist(anchor, anchor).pow(2)
        anchor_dist.fill_diagonal_(float('inf'))
        neighbor_idx = torch.topk(anchor_dist, k=k_eff, dim=1, largest=False).indices
        anchor_edges = anchor_dist.gather(1, neighbor_idx)
        anchor_scale = anchor_edges.mean().clamp_min(eps)
        anchor_edges = anchor_edges / anchor_scale
    current_dist = torch.cdist(current, current).pow(2)
    current_edges = current_dist.gather(1, neighbor_idx)
    current_scale = current_edges.detach().mean().clamp_min(eps)
    current_edges = current_edges / current_scale
    return torch.mean((torch.log1p(current_edges) - torch.log1p(anchor_edges)).pow(2))



# Refresh transport statistics over the training split.
def _refresh_global_anchorpt_anchor_transport_plan(device, train_dataset, model, anchorpt, batch_size):
    if not getattr(anchorpt, 'use_global_anchor_transport', False):
        return None
    loader = DataLoader(train_dataset, batch_size, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)
    was_training = model.training
    model.eval()
    z_list, omics_list = [], []
    with torch.no_grad():
        for batch in loader:
            view_data, batch_cov, omics_id, domain_id, label_cov = batch[:5]
            view_data, batch_cov, omics_id, label_cov = view_data.to(device), batch_cov.to(device), omics_id.to(device), label_cov.to(device)
            z_shared = model.encode_shared_mu(view_data, batch_cov, omics_id, label_cov).detach()
            z_list.append(z_shared)
            omics_list.append(omics_id.detach())
    z_all = torch.cat(z_list, dim=0)
    m_all = torch.cat(omics_list, dim=0)
    metrics = anchorpt.refresh_global_anchor_transport_plan(z_all, m_all)
    if was_training:
        model.train()
    return metrics



def train_stage2(
    device,
    writer,
    train_dataset,
    validate_dataset,
    model,
    anchorpt,
    epoch_num,
    batch_size,
    lr,
    pivotal_anchor_weight=0.15,
    stage2_base_weight=1.0,
    adaptlr=True,
    early_stopping=True,
    patience=25,
    trainable='shared_encoder',
    modality_labels=None,
    balanced_modality_batches=True,
    min_samples_per_modality_batch=16,
    random_state=42,
    stage1_knn_graph_weight=0.05,
    stage1_knn_graph_k=8,
    stage1_knn_graph_max_cells=128,
    pivotal_anchor_warmup_epochs=5,
    validate_modality_labels=None,
):
    stage2_base_weight = float(stage2_base_weight)
    if stage2_base_weight < 0:
        raise ValueError('stage2_base_weight must be non-negative.')
    _set_stage2_trainable(model, trainable=trainable)
    # Include all three modalities in refinement batches.
    if (
        getattr(anchorpt, 'num_modalities', 2) == 3
        and balanced_modality_batches
        and modality_labels is not None
    ):
        sampler = BalanceSampler(
            modality_labels=modality_labels,
            batch_size=batch_size,
            num_batches=None,
            random_state=random_state,
            min_samples_per_modality=min_samples_per_modality_batch,
        )
        train_data = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=4,
            pin_memory=True,
        )
    else:
        train_data = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=False, num_workers=4, pin_memory=True)

    if (
        getattr(anchorpt, 'num_modalities', 2) == 3
        and balanced_modality_batches
        and validate_modality_labels is not None
    ):
        validate_sampler = BalanceSampler(
            modality_labels=validate_modality_labels,
            batch_size=batch_size,
            num_batches=None,
            random_state=random_state + 100_000,
            min_samples_per_modality=min_samples_per_modality_batch,
        )
        validate_data = DataLoader(
            validate_dataset,
            batch_sampler=validate_sampler,
            num_workers=4,
            pin_memory=True,
        )
    else:
        validate_data = DataLoader(validate_dataset, batch_size, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)

    params = [p for p in model.parameters() if p.requires_grad] + list(anchorpt.parameters())
    if not params:
        raise ValueError('No trainable parameters found for Stage 2.')
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = None
    if adaptlr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=epoch_num * max(1, len(train_data)),
        )
    stopper = Stopper(patience=patience, verbose=True) if early_stopping else None

    global_step = 0
    for epoch in range(epoch_num):
        _set_stage2_mode(model, trainable=trainable)
        anchorpt.train()
        totals = {
            'loss': 0.0,
            'base_loss': 0.0,
            'stage2_base_weight': 0.0,
            'anchorpt_loss': 0.0,
            'anchorpt_anchor_transport_energy': 0.0,
            'anchorpt_transport_conditioned_contrastive_loss': 0.0,
            'anchorpt_transport_cost_logit_weight': 0.0,
            'anchorpt_contrastive_weight': 0.0,
            'stage1_knn_graph_loss': 0.0,
            'effective_pivotal_anchor_weight': 0.0,
            'anchorpt_global_anchor_transport_active': 0.0,
            'anchorpt_anchor_transport_mass': 0.0,
            'anchorpt_anchor_transport_entropy': 0.0,
            'anchorpt_anchor_transport_row_confidence': 0.0,
            'anchorpt_active_pivotal_anchors': 0.0,
            'anchorpt_low_mass_ratio': 0.0,
            'anchorpt_min_anchor_codebook_distance': 0.0,
            'anchorpt_min_dynamic_anchor_representation_distance': 0.0,
        }
        step_count = 0
        if pivotal_anchor_warmup_epochs and pivotal_anchor_warmup_epochs > 0:
            warmup_scale = min(1.0, float(epoch + 1) / float(pivotal_anchor_warmup_epochs))
        else:
            warmup_scale = 1.0
        effective_pivotal_anchor_weight = pivotal_anchor_weight * warmup_scale

        global_metrics = _refresh_global_anchorpt_anchor_transport_plan(device, train_dataset, model, anchorpt, batch_size)
        if global_metrics is not None:
            print(
                'stage2 epoch {} global_anchor_transport: ready={:.0f}, mass={:.4f}, entropy={:.4f}, active_anchor={:.2f}, low_mass={:.3f}'.format(
                    epoch + 1,
                    global_metrics.get('global_anchor_transport_ready', 0.0),
                    global_metrics.get('global_anchor_transport_mass', 0.0),
                    global_metrics.get('global_anchor_transport_entropy', 0.0),
                    global_metrics.get('global_active_pivotal_anchors', 0.0),
                    global_metrics.get('global_low_mass_ratio', 0.0),
                )
            )
            _set_stage2_mode(model, trainable=trainable)
            anchorpt.train()

        for batch in train_data:
            view_data, batch_cov, omics_id, domain_id, label_cov = batch[:5]
            anchor = batch[5] if len(batch) > 5 else None
            view_data, batch_cov, omics_id, domain_id, label_cov = view_data.to(device), batch_cov.to(device), omics_id.to(device), domain_id.to(device), label_cov.to(device)
            anchor = anchor.to(device) if anchor is not None and anchor.numel() > 0 else None
            optimizer.zero_grad(set_to_none=True)

            mu_shared, _, base_loss, _ = model(view_data, batch_cov, omics_id, domain_id, label_cov, stage='train')
            anchorpt_loss, anchorpt_metrics = anchorpt(mu_shared, omics_id)
            knn_graph_loss = mu_shared.new_zeros(())
            knn_graph_weight = float(stage1_knn_graph_weight)
            if anchor is not None and knn_graph_weight > 0:
                knn_graph_loss = _stage1_knn_graph_loss(
                    mu_shared,
                    anchor,
                    k=stage1_knn_graph_k,
                    max_cells=stage1_knn_graph_max_cells,
                )
            # Combine variational, anchor-contrastive, and neighborhood-preservation objectives.
            loss = (
                stage2_base_weight * base_loss
                + effective_pivotal_anchor_weight * anchorpt_loss
                + knn_graph_weight * knn_graph_loss
            )
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            model.eval()
            with torch.no_grad():
                mu_current = model.encode_shared_mu(view_data, batch_cov, omics_id, label_cov)
                anchorpt.update_ema(mu_current, omics_id)
            _set_stage2_mode(model, trainable=trainable)
            anchorpt.train()

            if scheduler is not None:
                scheduler.step()

            totals['loss'] += loss.item()
            totals['base_loss'] += base_loss.item()
            totals['stage2_base_weight'] += float(stage2_base_weight)
            totals['stage1_knn_graph_loss'] += _metric_value(knn_graph_loss)
            totals['effective_pivotal_anchor_weight'] += float(effective_pivotal_anchor_weight)
            for key in totals:
                if key in ('loss', 'base_loss', 'stage2_base_weight', 'stage1_knn_graph_loss', 'effective_pivotal_anchor_weight'):
                    continue
                if key in anchorpt_metrics:
                    totals[key] += _metric_value(anchorpt_metrics[key])

            global_step += 1
            step_count += 1
            if writer is not None:
                writer.add_scalar('Loss/stage2', loss.item(), global_step)
                writer.add_scalar('base_loss/stage2', base_loss.item(), global_step)
                writer.add_scalar('stage2_base_weight/stage2', float(stage2_base_weight), global_step)
                writer.add_scalar('stage1_knn_graph_loss/stage2', _metric_value(knn_graph_loss), global_step)
                writer.add_scalar('effective_pivotal_anchor_weight/stage2', float(effective_pivotal_anchor_weight), global_step)
                for key, value in anchorpt_metrics.items():
                    writer.add_scalar(f'{key}/stage2', _metric_value(value), global_step)
        denom = max(1, step_count)
        message = (
            'stage2 epoch {}: loss={:.4f}, base={:.4f}, bweight={:.2f}, anchorpt={:.4f}, '
            'anchor_transport_energy={:.4f}, tc_contrast={:.4f}, cost_logit_w={:.2f}, contrast_w={:.2f}, '
            'knn_graph={:.4f}, anchor_weight={:.4f}, global={:.2f}, '
            'T_mass={:.4f}, T_entropy={:.4f}, row_conf={:.4f}, active_anchor={:.2f}, low_mass={:.3f}, '
            'min_codebook_dist={:.4f}, min_state_dist={:.4f}'
        ).format(
            epoch + 1,
            totals['loss'] / denom,
            totals['base_loss'] / denom,
            totals['stage2_base_weight'] / denom,
            totals['anchorpt_loss'] / denom,
            totals['anchorpt_anchor_transport_energy'] / denom,
            totals['anchorpt_transport_conditioned_contrastive_loss'] / denom,
            totals['anchorpt_transport_cost_logit_weight'] / denom,
            totals['anchorpt_contrastive_weight'] / denom,
            totals['stage1_knn_graph_loss'] / denom,
            totals['effective_pivotal_anchor_weight'] / denom,
            totals['anchorpt_global_anchor_transport_active'] / denom,
            totals['anchorpt_anchor_transport_mass'] / denom,
            totals['anchorpt_anchor_transport_entropy'] / denom,
            totals['anchorpt_anchor_transport_row_confidence'] / denom,
            totals['anchorpt_active_pivotal_anchors'] / denom,
            totals['anchorpt_low_mass_ratio'] / denom,
            totals['anchorpt_min_anchor_codebook_distance'] / denom,
            totals['anchorpt_min_dynamic_anchor_representation_distance'] / denom,
        )
        print(message)

        if stopper is not None:
            validate_loss = validate_stage2(
                device,
                validate_data,
                model,
                anchorpt,
                pivotal_anchor_weight=pivotal_anchor_weight,
                stage2_base_weight=stage2_base_weight,
                stage1_knn_graph_weight=stage1_knn_graph_weight,
                stage1_knn_graph_k=stage1_knn_graph_k,
                stage1_knn_graph_max_cells=stage1_knn_graph_max_cells,
            )
            stopper(validate_loss)
            if stopper.early_stop:
                print(f'Stage 2 early stopping at epoch {epoch + 1}')
                break

def validate_stage2(
    device,
    validate_data,
    model,
    anchorpt,
    pivotal_anchor_weight=0.15,
    stage2_base_weight=1.0,
    stage1_knn_graph_weight=0.05,
    stage1_knn_graph_k=8,
    stage1_knn_graph_max_cells=128,
):
    stage2_base_weight = float(stage2_base_weight)
    if stage2_base_weight < 0:
        raise ValueError('stage2_base_weight must be non-negative.')
    model.eval()
    anchorpt.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in validate_data:
            view_data, batch_cov, omics_id, domain_id, label_cov = batch[:5]
            anchor = batch[5] if len(batch) > 5 else None
            view_data, batch_cov, omics_id, domain_id, label_cov = view_data.to(device), batch_cov.to(device), omics_id.to(device), domain_id.to(device), label_cov.to(device)
            anchor = anchor.to(device) if anchor is not None and anchor.numel() > 0 else None
            mu_shared, _, base_loss, _ = model(view_data, batch_cov, omics_id, domain_id, label_cov, stage='eval')
            anchorpt_loss, _ = anchorpt(mu_shared, omics_id)
            knn_graph_loss = mu_shared.new_zeros(())
            knn_graph_weight = float(stage1_knn_graph_weight)
            if anchor is not None and knn_graph_weight > 0:
                knn_graph_loss = _stage1_knn_graph_loss(
                    mu_shared,
                    anchor,
                    k=stage1_knn_graph_k,
                    max_cells=stage1_knn_graph_max_cells,
                )
            total_loss += (
                stage2_base_weight * base_loss
                + pivotal_anchor_weight * anchorpt_loss
                + knn_graph_weight * knn_graph_loss
            ).item()
            n_batches += 1
    return total_loss / max(1, n_batches)

def encode_dataset(device, inference_dataset, model, batch_size):
    model.eval()
    inference_data = DataLoader(inference_dataset, batch_size, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)

    z1_list, z2_list = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in inference_data:
            view_data, batch_cov, omics_id, domain_id, label_cov = batch[:5]
            view_data, batch_cov, omics_id, domain_id, label_cov = view_data.to(device), batch_cov.to(device), omics_id.to(device), domain_id.to(device), label_cov.to(device)
            z1, z2, loss, _ = model(view_data, batch_cov, omics_id, domain_id, label_cov, stage="eval")
            z1_list.append(z1.detach().cpu().numpy())
            z2_list.append(z2.detach().cpu().numpy())
            total_loss += loss.item()

    z_shared = np.concatenate(z1_list, axis=0)
    z_specific = np.concatenate(z2_list, axis=0)
    num_batch = np.ceil(len(inference_dataset) / batch_size)
    print(f"inference: loss={total_loss / max(1, num_batch):.4f}")
    return z_shared, z_specific