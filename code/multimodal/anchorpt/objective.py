"""Implement population transport for two- and three-modality AnchorPT.

Each modality owns a pivotal-anchor population. Two modalities use one transport
pair, while three modalities use RNA-centered pairs with optional pair weights;
the resulting plans provide soft targets for symmetric contrastive learning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AnchorPTObjective(nn.Module):

    def __init__(
        self,
        latent_dim,
        num_pivotal_anchors=16,
        assignment_temperature=0.5,
        sinkhorn_epsilon=0.05,
        sinkhorn_rho=1.0,
        sinkhorn_iters=80,
        ema_momentum=0.98,
        contrastive_temperature=0.08,
        contrastive_weight=0.20,
        transport_cost_logit_weight=2.50,
        min_assignment_mass=1e-4,
        use_global_anchor_transport=True,
        eps=1e-8,
        num_modalities=2,
        modality_names=None,
        modality_pair_weights=None,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.num_pivotal_anchors = int(num_pivotal_anchors)
        if self.num_pivotal_anchors < 2:
            raise ValueError("AnchorPT requires at least two pivotal anchors.")
        self.num_modalities = int(num_modalities)
        if self.num_modalities not in (2, 3):
            raise ValueError(f"Combined objective supports exactly 2 or 3 modalities; got {self.num_modalities}.")
        self.modality_names = tuple(str(x) for x in modality_names) if modality_names is not None else None
        self.modality_pairs = self._rna_hub_modality_pairs()
        self.register_buffer(
            "modality_pair_weights",
            self._build_modality_pair_weights(modality_pair_weights),
            persistent=False,
        )
        self.assignment_temperature = float(assignment_temperature)
        self.sinkhorn_epsilon = float(sinkhorn_epsilon)
        self.sinkhorn_rho = float(sinkhorn_rho)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.ema_momentum = float(ema_momentum)
        self.contrastive_temperature = float(contrastive_temperature)
        self.contrastive_weight = float(contrastive_weight)
        self.transport_cost_logit_weight = float(transport_cost_logit_weight)

        self.min_assignment_mass = float(min_assignment_mass)
        self.use_global_anchor_transport = bool(use_global_anchor_transport)
        self.eps = float(eps)

        # Maintain one trainable anchor population and detached steering summary per modality.
        self.pivotal_anchors = nn.Parameter(torch.randn(self.num_modalities, self.num_pivotal_anchors, self.latent_dim) * 0.02)
        self.register_buffer("steering_pivotal_anchors", torch.zeros(self.num_modalities, self.num_pivotal_anchors, self.latent_dim))
        self.register_buffer("ema_anchor_usage", torch.full((self.num_modalities, self.num_pivotal_anchors), 1.0 / self.num_pivotal_anchors))
        self.register_buffer("ema_initialized", torch.tensor(False, dtype=torch.bool))
        transport_shape = (self.num_pivotal_anchors, self.num_pivotal_anchors) if self.num_modalities == 2 else (len(self.modality_pairs), self.num_pivotal_anchors, self.num_pivotal_anchors)
        self.register_buffer("global_anchor_transport", torch.zeros(transport_shape))
        self.register_buffer("global_anchor_transport_ready", torch.tensor(False, dtype=torch.bool))
        with torch.no_grad():
            self.steering_pivotal_anchors.copy_(self.pivotal_anchors.detach())
            self.global_anchor_transport.copy_(self._anchor_transport_from_steering_anchors())
            self.ema_initialized.fill_(True)

    @torch.no_grad()
    def initialize_from_latents(self, z_shared, modality_labels, random_state=42):
        z_np = self._to_numpy(z_shared).astype(np.float32, copy=False)
        labels_np = self._to_numpy(modality_labels).astype(np.int64, copy=False).reshape(-1)
        if z_np.ndim != 2 or z_np.shape[1] != self.latent_dim:
            raise ValueError(
                f"z_shared must have shape (n, {self.latent_dim}); got {z_np.shape}."
            )
        if z_np.shape[0] != labels_np.shape[0]:
            raise ValueError("z_shared and modality_labels must have the same number of rows.")
        observed_modalities = set(np.unique(labels_np))
        expected_modalities = set(range(self.num_modalities))
        if self.num_modalities == 2:
            if not observed_modalities.issubset({0, 1}):
                raise ValueError("AnchorPT currently supports exactly two modalities encoded as 0 and 1.")
        elif not observed_modalities.issubset(expected_modalities):
            raise ValueError(
                f"AnchorPT modality labels must be encoded in [0, {self.num_modalities}); "
                f"got {sorted(observed_modalities)}."
            )

        centers = []
        for modality in range(self.num_modalities):
            values = z_np[labels_np == modality]
            if values.shape[0] == 0:
                raise ValueError(f"Cannot initialize AnchorPT: modality {modality} has no cells.")
            centers.append(self._kmeans_or_sample(values, random_state + modality))

        centers = torch.tensor(np.stack(centers, axis=0), dtype=self.pivotal_anchors.dtype, device=self.pivotal_anchors.device)
        self.pivotal_anchors.copy_(centers)
        self.steering_pivotal_anchors.copy_(centers)

        anchor_usages = []
        for modality in range(self.num_modalities):
            values = torch.tensor(
                z_np[labels_np == modality],
                dtype=self.pivotal_anchors.dtype,
                device=self.pivotal_anchors.device,
            )
            anchor_assignments = self._soft_anchor_assign(values, self.pivotal_anchors[modality])
            anchor_usages.append(anchor_assignments.mean(dim=0))
        anchor_usage = torch.stack(anchor_usages, dim=0)
        self.ema_anchor_usage.copy_(self._normalize_rows(anchor_usage))
        self.global_anchor_transport.copy_(self._anchor_transport_from_steering_anchors())
        self.global_anchor_transport_ready.fill_(False)
        self.ema_initialized.fill_(True)

    # Use the single RNA modality as the hub when three modalities are configured.
    def _rna_hub_modality_pairs(self):
        if self.num_modalities == 2:
            return ((0, 1),)
        if self.num_modalities != 3:
            raise ValueError(f"Combined objective supports exactly 2 or 3 modalities; got {self.num_modalities}.")
        if self.modality_names is None:
            raise ValueError("RNA-hub AnchorPT for >=3 modalities requires modality_names.")
        if len(self.modality_names) != self.num_modalities:
            raise ValueError(
                f"modality_names length must equal num_modalities={self.num_modalities}; "
                f"got {len(self.modality_names)}."
            )
        rna_matches = []
        for idx, name in enumerate(self.modality_names):
            normalized = str(name).strip().lower().replace("-", "_")
            tokens = [token for token in normalized.split("_") if token]
            if normalized == "rna" or "rna" in tokens:
                rna_matches.append(idx)
        if len(rna_matches) != 1:
            raise ValueError(
                "RNA-hub AnchorPT requires exactly one RNA modality name; "
                f"got modality_names={self.modality_names}."
            )
        hub = rna_matches[0]
        return tuple((hub, idx) for idx in range(self.num_modalities) if idx != hub)

    def _build_modality_pair_weights(self, modality_pair_weights):
        if modality_pair_weights is None or modality_pair_weights == "":
            weights = [1.0] * len(self.modality_pairs)
        elif isinstance(modality_pair_weights, str):
            weights = [float(x.strip()) for x in modality_pair_weights.split(",") if x.strip()]
        else:
            weights = [float(x) for x in modality_pair_weights]
        if len(weights) != len(self.modality_pairs):
            raise ValueError(
                f"modality_pair_weights length must match Stage-2 pairs {self.modality_pairs}; "
                f"got weights={weights}."
            )
        if any(w < 0 for w in weights):
            raise ValueError(f"modality_pair_weights must be non-negative; got {weights}.")
        if sum(weights) <= 0:
            raise ValueError(f"At least one modality pair weight must be positive; got {weights}.")
        return torch.tensor(weights, dtype=torch.float32)

    @staticmethod
    def _to_numpy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _kmeans_or_sample(self, values, random_state):
        if values.shape[0] >= self.num_pivotal_anchors:
            try:
                from sklearn.cluster import KMeans

                km = KMeans(n_clusters=self.num_pivotal_anchors, n_init=10, random_state=random_state)
                return km.fit(values).cluster_centers_.astype(np.float32, copy=False)
            except Exception:
                pass

        rng = np.random.default_rng(random_state)
        replace = values.shape[0] < self.num_pivotal_anchors
        indices = rng.choice(values.shape[0], size=self.num_pivotal_anchors, replace=replace)
        return values[indices].astype(np.float32, copy=False)

    def _soft_anchor_assign(self, z, pivotal_anchors):
        distances = torch.cdist(z.float(), pivotal_anchors.float()).pow(2)
        return torch.softmax(-distances / max(self.assignment_temperature, self.eps), dim=1)

    def _assignment_conditioned_dynamic_anchor_representations(self, z, pivotal_anchors):
        anchor_assignments = self._soft_anchor_assign(z, pivotal_anchors)
        mass = anchor_assignments.sum(dim=0)
        dynamic_anchor_representations = anchor_assignments.transpose(0, 1) @ z
        dynamic_anchor_representations = dynamic_anchor_representations / mass.clamp_min(self.eps).unsqueeze(1)
        low_mass = mass < self.min_assignment_mass
        if torch.any(low_mass):
            dynamic_anchor_representations = torch.where(low_mass.unsqueeze(1), pivotal_anchors, dynamic_anchor_representations)
        anchor_usage = mass / mass.sum().clamp_min(self.eps)
        return dynamic_anchor_representations, anchor_usage, mass

    def _normalize_rows(self, x):
        x = torch.clamp(x, min=self.eps)
        return x / x.sum(dim=1, keepdim=True).clamp_min(self.eps)

    @torch.no_grad()
    def _bootstrap_ema(self, dynamic_anchor_representations, anchor_usages):
        self.steering_pivotal_anchors.copy_(torch.stack(dynamic_anchor_representations, dim=0).detach())
        self.ema_anchor_usage.copy_(self._normalize_rows(torch.stack(anchor_usages, dim=0).detach()))
        self.global_anchor_transport.copy_(self._anchor_transport_from_steering_anchors())
        self.ema_initialized.fill_(True)

    @torch.no_grad()
    def refresh_global_anchor_transport_plan(self, z_shared, modality_onehot):
       
        dynamic_anchor_representations, anchor_usages, assignment_masses, ok = self.batch_statistics(z_shared, modality_onehot)
        if not ok:
            return {
                "global_anchor_transport_ready": 0.0,
                "global_anchor_transport_mass": 0.0,
                "global_anchor_transport_entropy": 0.0,
                "global_active_pivotal_anchors": 0.0,
                "global_low_mass_ratio": 1.0,
            }

        stacked_dynamic_anchor_representations = torch.stack(dynamic_anchor_representations, dim=0).detach()
        stacked_anchor_usage = self._normalize_rows(torch.stack(anchor_usages, dim=0).detach())
        self.steering_pivotal_anchors.copy_(stacked_dynamic_anchor_representations)
        self.ema_anchor_usage.copy_(stacked_anchor_usage)
        self.global_anchor_transport.copy_(self._anchor_transport_from_steering_anchors())
        self.global_anchor_transport_ready.fill_(True)
        if self.num_modalities == 2:
            global_transport_mass = self.global_anchor_transport.sum()
            global_transport_entropy = self._anchor_transport_entropy(self.global_anchor_transport)
        else:
            global_transport_mass = self.global_anchor_transport.sum(dim=(1, 2)).mean()
            global_transport_entropy = torch.stack([
                self._anchor_transport_entropy(plan) for plan in self.global_anchor_transport
            ]).mean()
        return {
            "global_anchor_transport_ready": 1.0,
            "global_anchor_transport_mass": float(global_transport_mass.detach().cpu().item()),
            "global_anchor_transport_entropy": float(global_transport_entropy.detach().cpu().item()),
            "global_active_pivotal_anchors": float(self._active_pivotal_anchors(assignment_masses).detach().cpu().item()),
            "global_low_mass_ratio": float(self._low_mass_ratio(assignment_masses).detach().cpu().item()),
        }

    @torch.no_grad()
    def update_ema(self, z_shared, modality_onehot):
        if self.use_global_anchor_transport:
            return False
        dynamic_anchor_representations, anchor_usages, _, ok = self.batch_statistics(z_shared, modality_onehot)
        if not ok:
            return False
        if not bool(self.ema_initialized.item()):
            self._bootstrap_ema(dynamic_anchor_representations, anchor_usages)
            return True
        momentum = self.ema_momentum
        self.steering_pivotal_anchors.mul_(momentum).add_(torch.stack(dynamic_anchor_representations, dim=0).detach(), alpha=1.0 - momentum)
        anchor_usage = momentum * self.ema_anchor_usage + (1.0 - momentum) * torch.stack(anchor_usages, dim=0).detach()
        self.ema_anchor_usage.copy_(self._normalize_rows(anchor_usage))
        return True

    def batch_statistics(self, z_shared, modality_onehot):
        modality_index = torch.argmax(modality_onehot, dim=1)
        dynamic_anchor_representations = []
        anchor_usages = []
        assignment_masses = []
        for modality in range(self.num_modalities):
            selected = modality_index == modality
            if not torch.any(selected):
                return dynamic_anchor_representations, anchor_usages, assignment_masses, False
            dynamic_anchor_representation, anchor_usage, mass = self._assignment_conditioned_dynamic_anchor_representations(z_shared[selected], self.pivotal_anchors[modality])
            dynamic_anchor_representations.append(dynamic_anchor_representation)
            anchor_usages.append(anchor_usage)
            assignment_masses.append(mass)
        return dynamic_anchor_representations, anchor_usages, assignment_masses, True

    # Estimate a relaxed plan between normalized anchor-usage distributions.
    def _unbalanced_sinkhorn(self, a, b, cost):
        a = torch.clamp(a.float(), min=self.eps)
        b = torch.clamp(b.float(), min=self.eps)
        a = a / a.sum().clamp_min(self.eps)
        b = b / b.sum().clamp_min(self.eps)

        scale = cost.detach().mean().clamp_min(self.eps)
        cost = cost.float() / scale
        kernel = torch.exp(-cost / max(self.sinkhorn_epsilon, self.eps)).clamp_min(self.eps)
        power = self.sinkhorn_rho / (self.sinkhorn_rho + self.sinkhorn_epsilon)

        u = torch.ones_like(a)
        v = torch.ones_like(b)
        for _ in range(self.sinkhorn_iters):
            u = (a / (kernel @ v).clamp_min(self.eps)).pow(power)
            v = (b / (kernel.transpose(0, 1) @ u).clamp_min(self.eps)).pow(power)
        return u.unsqueeze(1) * kernel * v.unsqueeze(0)

    def _anchor_transport_from_steering_anchors(self):
        if self.num_modalities == 2:
            cost = torch.cdist(
                self.steering_pivotal_anchors[0].float(),
                self.steering_pivotal_anchors[1].float(),
            ).pow(2)
            return self._unbalanced_sinkhorn(self.ema_anchor_usage[0], self.ema_anchor_usage[1], cost)

        transports = []
        for left, right in self.modality_pairs:
            cost = torch.cdist(
                self.steering_pivotal_anchors[left].float(),
                self.steering_pivotal_anchors[right].float(),
            ).pow(2)
            transports.append(self._unbalanced_sinkhorn(self.ema_anchor_usage[left], self.ema_anchor_usage[right], cost))
        return torch.stack(transports, dim=0)

    def _anchor_transport_plan(self, pair_index=None):
        if self.use_global_anchor_transport and bool(self.global_anchor_transport_ready.item()):
            anchor_transport = self.global_anchor_transport.detach()
        else:
            with torch.no_grad():
                anchor_transport = self._anchor_transport_from_steering_anchors()
            anchor_transport = anchor_transport.detach()
        if self.num_modalities == 2 or pair_index is None:
            return anchor_transport
        return anchor_transport[pair_index]

    def _transport_conditioned_contrastive_one_direction(self, source, target, anchor_transport, anchor_transport_cost):
        row_mass = anchor_transport.sum(dim=1)
        valid = row_mass > self.eps
        q = anchor_transport / row_mass.clamp_min(self.eps).unsqueeze(1)

        q_safe = q.clamp_min(self.eps)
        entropy = -(q_safe * q_safe.log()).sum(dim=1)
        confidence = (1.0 - entropy / np.log(self.num_pivotal_anchors)).clamp(min=0.0, max=1.0)
        weights = confidence * valid.float()
        if torch.sum(weights) <= self.eps:
            return source.new_zeros(()), source.new_zeros(())

        logits = F.normalize(source, dim=1) @ F.normalize(target, dim=1).transpose(0, 1)
        logits = logits / max(self.contrastive_temperature, self.eps)
        if self.transport_cost_logit_weight != 0.0:
            cost_scale = anchor_transport_cost.detach().mean().clamp_min(self.eps)
            normalized_cost = anchor_transport_cost.float() / cost_scale
            logits = logits - self.transport_cost_logit_weight * normalized_cost.to(
                dtype=logits.dtype,
                device=logits.device,
            )
        log_probs = F.log_softmax(logits, dim=1)
        per_anchor = -(q * log_probs).sum(dim=1)
        loss = torch.sum(weights * per_anchor) / weights.sum().clamp_min(self.eps)
        return loss, confidence[valid].mean()

    def forward(self, z_shared, modality_onehot):
        if modality_onehot.shape[1] != self.num_modalities:
            if self.num_modalities == 2:
                raise ValueError("AnchorPT currently supports exactly two modalities.")
            raise ValueError(f"AnchorPT expected one-hot labels for {self.num_modalities} modalities.")

        dynamic_anchor_representations, anchor_usages, assignment_masses, ok = self.batch_statistics(z_shared, modality_onehot)
        if not ok:
            return self._zero_result(z_shared, reason="missing_modality")

        # Preserve one pair for two modalities and aggregate weighted RNA-hub pairs for three.
        if self.num_modalities == 2:
            anchor_transport = self._anchor_transport_plan()
            dynamic_anchor_cost = torch.cdist(dynamic_anchor_representations[0].float(), dynamic_anchor_representations[1].float()).pow(2)
            anchor_transport_energy = torch.sum(anchor_transport * dynamic_anchor_cost)

            contrast_01, conf_01 = self._transport_conditioned_contrastive_one_direction(
                dynamic_anchor_representations[0],
                dynamic_anchor_representations[1],
                anchor_transport,
                dynamic_anchor_cost,
            )
            contrast_10, conf_10 = self._transport_conditioned_contrastive_one_direction(
                dynamic_anchor_representations[1],
                dynamic_anchor_representations[0],
                anchor_transport.transpose(0, 1),
                dynamic_anchor_cost.transpose(0, 1),
            )
            transport_conditioned_contrastive_loss = 0.5 * (contrast_01 + contrast_10)
            pivotal_anchor_loss = self.contrastive_weight * transport_conditioned_contrastive_loss

            metrics = {
                "anchorpt_loss": pivotal_anchor_loss.detach(),
                "anchorpt_anchor_transport_energy": anchor_transport_energy.detach(),
                "anchorpt_transport_conditioned_contrastive_loss": transport_conditioned_contrastive_loss.detach(),
                "anchorpt_transport_cost_logit_weight": z_shared.new_tensor(self.transport_cost_logit_weight).detach(),
                "anchorpt_contrastive_weight": z_shared.new_tensor(self.contrastive_weight).detach(),

                "anchorpt_anchor_transport_mass": anchor_transport.sum().detach(),
                "anchorpt_anchor_transport_entropy": self._anchor_transport_entropy(anchor_transport).detach(),
                "anchorpt_anchor_transport_row_confidence": (0.5 * (conf_01 + conf_10)).detach(),
                "anchorpt_active_pivotal_anchors": self._active_pivotal_anchors(assignment_masses).detach(),
                "anchorpt_low_mass_ratio": self._low_mass_ratio(assignment_masses).detach(),
                "anchorpt_min_anchor_codebook_distance": self._min_pairwise_distance(self.pivotal_anchors).detach(),
                "anchorpt_min_dynamic_anchor_representation_distance": self._min_pairwise_distance(torch.stack(dynamic_anchor_representations, dim=0)).detach(),
                "anchorpt_global_anchor_transport_active": z_shared.new_tensor(float(self.use_global_anchor_transport and bool(self.global_anchor_transport_ready.item()))).detach(),
            }
            return pivotal_anchor_loss, metrics

        pair_losses = []
        pair_energies = []
        pair_masses = []
        pair_entropies = []
        pair_confidences = []
        pair_metrics = {}
        for pair_index, (left, right) in enumerate(self.modality_pairs):
            anchor_transport = self._anchor_transport_plan(pair_index=pair_index)
            dynamic_anchor_cost = torch.cdist(
                dynamic_anchor_representations[left].float(),
                dynamic_anchor_representations[right].float(),
            ).pow(2)
            anchor_transport_energy = torch.sum(anchor_transport * dynamic_anchor_cost)
            contrast_lr, conf_lr = self._transport_conditioned_contrastive_one_direction(
                dynamic_anchor_representations[left],
                dynamic_anchor_representations[right],
                anchor_transport,
                dynamic_anchor_cost,
            )
            contrast_rl, conf_rl = self._transport_conditioned_contrastive_one_direction(
                dynamic_anchor_representations[right],
                dynamic_anchor_representations[left],
                anchor_transport.transpose(0, 1),
                dynamic_anchor_cost.transpose(0, 1),
            )
            pair_contrastive_loss = 0.5 * (contrast_lr + contrast_rl)
            pair_confidence = 0.5 * (conf_lr + conf_rl)
            pair_mass = anchor_transport.sum()
            pair_entropy = self._anchor_transport_entropy(anchor_transport)
            pair_losses.append(pair_contrastive_loss)
            pair_energies.append(anchor_transport_energy)
            pair_masses.append(pair_mass)
            pair_entropies.append(pair_entropy)
            pair_confidences.append(pair_confidence)
            prefix = f"anchorpt_pair_{left}_{right}"
            pair_metrics[prefix + "_loss"] = (self.contrastive_weight * pair_contrastive_loss).detach()
            pair_metrics[prefix + "_anchor_transport_energy"] = anchor_transport_energy.detach()
            pair_metrics[prefix + "_anchor_transport_mass"] = pair_mass.detach()
            pair_metrics[prefix + "_anchor_transport_entropy"] = pair_entropy.detach()
            pair_metrics[prefix + "_anchor_transport_row_confidence"] = pair_confidence.detach()
            pair_metrics[prefix + "_weight"] = self.modality_pair_weights[pair_index].detach()

        pair_weights = self.modality_pair_weights.to(device=z_shared.device, dtype=torch.float32)
        pair_weight_sum = pair_weights.sum().clamp_min(self.eps)
        transport_conditioned_contrastive_loss = (torch.stack(pair_losses) * pair_weights).sum() / pair_weight_sum
        weighted_anchor_transport_energy = (torch.stack(pair_energies) * pair_weights).sum() / pair_weight_sum
        weighted_anchor_transport_mass = (torch.stack(pair_masses) * pair_weights).sum() / pair_weight_sum
        weighted_anchor_transport_entropy = (torch.stack(pair_entropies) * pair_weights).sum() / pair_weight_sum
        weighted_anchor_transport_row_confidence = (torch.stack(pair_confidences) * pair_weights).sum() / pair_weight_sum
        pivotal_anchor_loss = self.contrastive_weight * transport_conditioned_contrastive_loss
        metrics = {
            "anchorpt_loss": pivotal_anchor_loss.detach(),
            "anchorpt_anchor_transport_energy": weighted_anchor_transport_energy.detach(),
            "anchorpt_transport_conditioned_contrastive_loss": transport_conditioned_contrastive_loss.detach(),
            "anchorpt_transport_cost_logit_weight": z_shared.new_tensor(self.transport_cost_logit_weight).detach(),
            "anchorpt_contrastive_weight": z_shared.new_tensor(self.contrastive_weight).detach(),
            "anchorpt_anchor_transport_mass": weighted_anchor_transport_mass.detach(),
            "anchorpt_anchor_transport_entropy": weighted_anchor_transport_entropy.detach(),
            "anchorpt_anchor_transport_row_confidence": weighted_anchor_transport_row_confidence.detach(),
            "anchorpt_active_pivotal_anchors": self._active_pivotal_anchors(assignment_masses).detach(),
            "anchorpt_low_mass_ratio": self._low_mass_ratio(assignment_masses).detach(),
            "anchorpt_min_anchor_codebook_distance": self._min_pairwise_distance(self.pivotal_anchors).detach(),
            "anchorpt_min_dynamic_anchor_representation_distance": self._min_pairwise_distance(torch.stack(dynamic_anchor_representations, dim=0)).detach(),
            "anchorpt_global_anchor_transport_active": z_shared.new_tensor(float(self.use_global_anchor_transport and bool(self.global_anchor_transport_ready.item()))).detach(),
        }
        metrics.update(pair_metrics)
        return pivotal_anchor_loss, metrics

    def _zero_result(self, z_shared, reason):
        zero = z_shared.new_zeros(())
        metrics = {
            "anchorpt_loss": zero.detach(),
            "anchorpt_anchor_transport_energy": zero.detach(),
            "anchorpt_transport_conditioned_contrastive_loss": zero.detach(),
            "anchorpt_transport_cost_logit_weight": zero.detach(),
            "anchorpt_contrastive_weight": zero.detach(),
   
            "anchorpt_anchor_transport_mass": zero.detach(),
            "anchorpt_anchor_transport_entropy": zero.detach(),
            "anchorpt_anchor_transport_row_confidence": zero.detach(),
            "anchorpt_active_pivotal_anchors": zero.detach(),
            "anchorpt_low_mass_ratio": zero.detach(),
            "anchorpt_min_codebook_distance": zero.detach(),
            "anchorpt_min_state_distance": zero.detach(),
            "anchorpt_global_anchor_transport_active": zero.detach(),
            "anchorpt_skip_" + reason: torch.ones((), device=z_shared.device).detach(),
        }
        return zero, metrics

    def _anchor_transport_entropy(self, anchor_transport):
        mass = anchor_transport.sum().clamp_min(self.eps)
        prob = (anchor_transport / mass).clamp_min(self.eps)
        return -(prob * prob.log()).sum()

    def _active_pivotal_anchors(self, assignment_masses):
        active = []
        for mass in assignment_masses:
            active.append((mass > self.min_assignment_mass).float().sum())
        return torch.stack(active).mean()

    def _low_mass_ratio(self, assignment_masses):
        ratios = []
        for mass in assignment_masses:
            ratios.append((mass <= self.min_assignment_mass).float().mean())
        return torch.stack(ratios).mean()

    def _min_pairwise_distance(self, points):
        values = []
        for modality in range(points.shape[0]):
            distances = torch.cdist(points[modality].float(), points[modality].float()).pow(2)
            mask = ~torch.eye(self.num_pivotal_anchors, dtype=torch.bool, device=distances.device)
            values.append(distances[mask].min())
        return torch.stack(values).min()

