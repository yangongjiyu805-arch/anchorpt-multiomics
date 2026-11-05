"""Define the variational model used by AnchorPT."""

import torch
import torch.nn as nn

from .losses import (
    ZINBLoss,
    kl_loss,
    kl_loss_prior,
    mse_loss,
)


def _normalization_layer(dim, normalization):
    if normalization == "batch":
        return nn.BatchNorm1d(dim)
    if normalization == "layer":
        return nn.LayerNorm(dim)
    if normalization in (None, "none"):
        return None
    raise ValueError(f"Unsupported normalization: {normalization}")


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim=3000,
        layer_dims=None,
        latent_dim=20,
        dropout_rate=0.5,
        normalization="batch",
    ):
        super().__init__()
        if layer_dims is None:
            layer_dims = [128, 128]

        layers = []
        cur = input_dim
        for dim in layer_dims:
            layers.append(nn.Linear(cur, dim))
            norm = _normalization_layer(dim, normalization)
            if norm is not None:
                layers.append(norm)
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout_rate))
            cur = dim

        self.encoder = nn.Sequential(*layers)
        self.mu_layer = nn.Linear(layer_dims[-1], latent_dim)
        self.logvar_layer = nn.Linear(layer_dims[-1], latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class MSEDecoder(nn.Module):
    def __init__(
        self,
        input_dim=3000,
        covariate_dim=1,
        layer_dims=None,
        latent_dim=20,
        dropout_rate=0.5,
        positive_outputs=True,
        normalization="batch",
    ):
        super().__init__()
        if layer_dims is None:
            layer_dims = [128, 128]

        self.covariate_dim = covariate_dim
        layers = []
        cur = latent_dim + covariate_dim
        for dim in reversed(layer_dims):
            layers.append(nn.Linear(cur, dim))
            layers.append(nn.LeakyReLU(0.1))
            norm = _normalization_layer(dim, normalization)
            if norm is not None:
                layers.append(norm)
            layers.append(nn.Dropout(dropout_rate))
            cur = dim
        self.decoder = nn.Sequential(*layers)

        if positive_outputs:
            self.mean_layer = nn.Sequential(nn.Linear(layer_dims[0], input_dim), nn.Softplus())
        else:
            self.mean_layer = nn.Linear(layer_dims[0], input_dim)

    def forward(self, z, decoder_covariates):
        if self.covariate_dim > 0:
            z = torch.cat([z, decoder_covariates], dim=1)
        h = self.decoder(z)
        return self.mean_layer(h)


class ZINBDecoder(nn.Module):
    def __init__(
        self,
        input_dim=3000,
        covariate_dim=1,
        modality_num=2,
        layer_dims=None,
        latent_dim=20,
        dropout_rate=0.5,
        normalization="batch",
    ):
        super().__init__()
        if layer_dims is None:
            layer_dims = [128, 128]

        self.covariate_dim = covariate_dim
        self.modality_num = modality_num

        layers = []
        cur = latent_dim + covariate_dim
        for dim in reversed(layer_dims):
            layers.append(nn.Linear(cur, dim))
            norm = _normalization_layer(dim, normalization)
            if norm is not None:
                layers.append(norm)
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(dropout_rate))
            cur = dim

        self.decoder = nn.Sequential(*layers)
        self.mean_logits_layer = nn.Linear(layer_dims[0], input_dim)
        self.dispersion = nn.Parameter(torch.randn(input_dim))
        self.dispersion_modality = nn.Parameter(torch.randn(modality_num, input_dim))
        self.dropout_layer = nn.Sequential(nn.Linear(layer_dims[0], input_dim), nn.Sigmoid())

    def forward(self, z, decoder_covariates, omics_id, dispersion_strategy="gene-modality", feature_mask=None):
        if self.covariate_dim > 0:
            z = torch.cat([z, decoder_covariates], dim=1)
        h = self.decoder(z)
        logits = self.mean_logits_layer(h)
        if feature_mask is not None:
            if feature_mask.dim() == 1:
                feature_mask = feature_mask.unsqueeze(0)
            feature_mask = feature_mask.to(device=logits.device, dtype=torch.bool)
            logits = logits.masked_fill(~feature_mask, torch.finfo(logits.dtype).min)
        rho = torch.softmax(logits, dim=-1)
        if dispersion_strategy == "gene":
            dispersion = torch.exp(self.dispersion)
        else:
            dispersion = torch.exp(omics_id @ self.dispersion_modality)
        pi = self.dropout_layer(h)
        return rho, dispersion, pi


class NBDecoder(ZINBDecoder):
    def forward(self, z, decoder_covariates, omics_id, dispersion_strategy="gene-modality", feature_mask=None):
        rho, dispersion, _ = super().forward(
            z,
            decoder_covariates,
            omics_id,
            dispersion_strategy=dispersion_strategy,
            feature_mask=feature_mask,
        )
        pi = torch.zeros_like(rho)
        return rho, dispersion, pi


class AnchorPTNet(nn.Module):
    def __init__(
        self,
        device,
        input_dim,
        modality_num,
        covariate_dim=1,
        celltype_num=0,
        layer_dims=None,
        latent_dim_shared=48,
        latent_dim_specific=20,
        dropout_rate=0.05,
        beta=1.25,
        feat_mask=None,
        distribution="Normal_positive",
        encoder_covariates=False,
        multi_encoder=False,
        shared_recon_weight=1.0,
        decoder_arch="conditional",
        decoder_condition_on_modality=True,
        encoder_normalization=None,
        decoder_normalization=None,
        eps=1e-10,
    ):
        super().__init__()
        if layer_dims is None:
            layer_dims = [128, 128]

        self.device = device
        self.eps = eps
        self.beta = beta

        self.input_dim = input_dim
        self.modality_num = modality_num
        self.covariate_dim = covariate_dim
        self.celltype_num = celltype_num
        self.encoder_covariates = encoder_covariates
        self.multi_encoder = multi_encoder
        self.shared_recon_weight = shared_recon_weight
        self.decoder_arch = decoder_arch
        self.decoder_condition_on_modality = decoder_condition_on_modality
        self.latent_dim_shared = latent_dim_shared
        self.latent_dim_specific = latent_dim_specific

        if decoder_arch not in ("per_modality", "conditional"):
            raise ValueError("decoder_arch must be 'per_modality' or 'conditional'.")

        if encoder_normalization is None:
            encoder_normalization = "layer" if multi_encoder else "batch"
        if decoder_normalization is None:
            decoder_normalization = "layer" if decoder_arch == "per_modality" else "batch"

        self.feat_mask = feat_mask.to(self.device) if feat_mask is not None else None

        self.distribution = distribution
        if distribution in ["ZINB", "NB"]:
            self.count_data = True
            self.positive_outputs = True
        elif distribution == "Normal":
            self.count_data = False
            self.positive_outputs = False
        else:
            self.count_data = False
            self.positive_outputs = True
        self.zinb_loss = ZINBLoss() if self.count_data else None

        shared_encoder_dim = input_dim + covariate_dim * encoder_covariates + self.celltype_num
        specific_encoder_dim = shared_encoder_dim + (0 if multi_encoder else modality_num)
        # With one encoder pair, modality identity is appended only to the specific path.
        if multi_encoder:
            self.encoder_shared = nn.ModuleList(
                Encoder(
                    input_dim=shared_encoder_dim,
                    layer_dims=layer_dims,
                    latent_dim=latent_dim_shared,
                    dropout_rate=dropout_rate,
                    normalization=encoder_normalization,
                )
                for _ in range(modality_num)
            )
            self.encoder_specific = nn.ModuleList(
                Encoder(
                    input_dim=specific_encoder_dim,
                    layer_dims=layer_dims,
                    latent_dim=latent_dim_specific,
                    dropout_rate=dropout_rate,
                    normalization=encoder_normalization,
                )
                for _ in range(modality_num)
            )
        else:
            self.encoder_shared = Encoder(
                input_dim=shared_encoder_dim,
                layer_dims=layer_dims,
                latent_dim=latent_dim_shared,
                dropout_rate=dropout_rate,
                normalization=encoder_normalization,
            )
            self.encoder_specific = Encoder(
                input_dim=specific_encoder_dim,
                layer_dims=layer_dims,
                latent_dim=latent_dim_specific,
                dropout_rate=dropout_rate,
                normalization=encoder_normalization,
            )

        dec_latent_dim = latent_dim_shared + latent_dim_specific
        decoder_covariate_dim = covariate_dim + (modality_num if decoder_condition_on_modality else 0)

        # Select a reconstruction head that matches the configured observation distribution.
        def make_decoder(latent_dim):
            if distribution == "ZINB":
                return ZINBDecoder(
                    input_dim=input_dim, covariate_dim=decoder_covariate_dim,
                    modality_num=modality_num, layer_dims=layer_dims,
                    latent_dim=latent_dim, dropout_rate=dropout_rate,
                    normalization=decoder_normalization,
                )
            if distribution == "NB":
                return NBDecoder(
                    input_dim=input_dim, covariate_dim=decoder_covariate_dim,
                    modality_num=modality_num, layer_dims=layer_dims,
                    latent_dim=latent_dim, dropout_rate=dropout_rate,
                    normalization=decoder_normalization,
                )
            return MSEDecoder(
                input_dim=input_dim, covariate_dim=decoder_covariate_dim,
                layer_dims=layer_dims, latent_dim=latent_dim,
                dropout_rate=dropout_rate, positive_outputs=self.positive_outputs,
                normalization=decoder_normalization,
            )

        if decoder_arch == "per_modality":
            self.decoder = nn.ModuleList(make_decoder(dec_latent_dim) for _ in range(modality_num))
            self.shared_factor_sufficiency_decoder = nn.ModuleList(make_decoder(latent_dim_shared) for _ in range(modality_num))
        else:
            self.decoder = make_decoder(dec_latent_dim)
            self.shared_factor_sufficiency_decoder = make_decoder(latent_dim_shared)

        # Learn a modality-conditioned prior for the modality-specific latent variables.
        self.prior_net_specific = nn.Sequential(
            nn.Linear(modality_num + self.celltype_num, 10),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(10, 2 * latent_dim_specific),
        )

    def _empty_covariates(self, view_data):
        return view_data.new_zeros((view_data.shape[0], 0))

    def _base_covariates(self, batch_cov, view_data):
        if self.covariate_dim == 0:
            return self._empty_covariates(view_data)
        if batch_cov.dim() == 1:
            batch_cov = batch_cov.unsqueeze(1)
        return batch_cov

    def _celltype_covariates(self, label_cov, view_data):
        if self.celltype_num == 0:
            return self._empty_covariates(view_data)
        if label_cov.dim() == 1:
            label_cov = label_cov.unsqueeze(1)
        return label_cov

    def _cat_nonempty(self, tensors):
        tensors = [t for t in tensors if t.shape[1] > 0]
        if not tensors:
            raise ValueError("No tensors to concatenate.")
        if len(tensors) == 1:
            return tensors[0]
        return torch.cat(tensors, dim=-1)

    def _decoder_covariates(self, batch_cov, omics_id, view_data):
        tensors = [self._base_covariates(batch_cov, view_data)]
        if self.decoder_condition_on_modality:
            tensors.append(omics_id)
        if all(t.shape[1] == 0 for t in tensors):
            return self._empty_covariates(view_data)
        return self._cat_nonempty(tensors)

    def _route_encoder(self, encoders, encoder_input, omics_id, latent_dim):
        modality_index = torch.argmax(omics_id, dim=1)
        z = encoder_input.new_zeros((encoder_input.shape[0], latent_dim))
        mu = encoder_input.new_zeros((encoder_input.shape[0], latent_dim))
        logvar = encoder_input.new_zeros((encoder_input.shape[0], latent_dim))

        for modality_id, encoder in enumerate(encoders):
            selected = modality_index == modality_id
            if not torch.any(selected):
                continue
            z_selected, mu_selected, logvar_selected = encoder(encoder_input[selected])
            z[selected] = z_selected
            mu[selected] = mu_selected
            logvar[selected] = logvar_selected
        return z, mu, logvar

    def _feature_mask_from_onehot(self, onehot):
        if self.feat_mask is None or onehot.shape[1] != self.feat_mask.shape[0]:
            return None
        return onehot @ self.feat_mask

    def _decode(self, decoder, z, decoder_covariates, omics_id, feature_mask=None):
        if self.count_data:
            return decoder(z, decoder_covariates, omics_id, feature_mask=feature_mask)
        return decoder(z, decoder_covariates)

    def _route_decoder(self, decoders, z, decoder_covariates, omics_id, feature_mask=None):
        if self.decoder_arch == "conditional":
            return self._decode(decoders, z, decoder_covariates, omics_id, feature_mask=feature_mask)

        modality_index = torch.argmax(omics_id, dim=1)
        if self.count_data:
            rho = z.new_zeros((z.shape[0], self.input_dim))
            dispersion = z.new_zeros((z.shape[0], self.input_dim))
            pi = z.new_zeros((z.shape[0], self.input_dim))
            for modality_id, decoder in enumerate(decoders):
                selected = modality_index == modality_id
                if not torch.any(selected):
                    continue
                selected_mask = feature_mask[selected] if feature_mask is not None else None
                rho_selected, dispersion_selected, pi_selected = self._decode(
                    decoder,
                    z[selected],
                    decoder_covariates[selected],
                    omics_id[selected],
                    feature_mask=selected_mask,
                )
                rho[selected] = rho_selected
                dispersion[selected] = dispersion_selected
                pi[selected] = pi_selected
            return rho, dispersion, pi

        out = z.new_zeros((z.shape[0], self.input_dim))
        for modality_id, decoder in enumerate(decoders):
            selected = modality_index == modality_id
            if not torch.any(selected):
                continue
            out[selected] = self._decode(
                decoder,
                z[selected],
                decoder_covariates[selected],
                omics_id[selected],
                feature_mask=None,
            )
        return out

    # Export deterministic encoder means rather than sampled latent variables.
    def encode_shared_mu(self, view_data, batch_cov, omics_id, label_cov):
        if self.count_data:
            view_data = torch.log1p(view_data)

        base_covariates = self._base_covariates(batch_cov, view_data)
        celltype_covariates = self._celltype_covariates(label_cov, view_data)

        shared_input_parts = [view_data]
        if self.encoder_covariates:
            shared_input_parts.append(base_covariates)
        shared_input_parts.append(celltype_covariates)
        shared_input = self._cat_nonempty(shared_input_parts)

        if self.multi_encoder:
            _, mu_shared, _ = self._route_encoder(
                self.encoder_shared,
                shared_input,
                omics_id,
                self.latent_dim_shared,
            )
        else:
            _, mu_shared, _ = self.encoder_shared(shared_input)
        return mu_shared

    def forward(self, view_data, batch_cov, omics_id, domain_id, label_cov, stage="train"):
        view_data_original = view_data
        # Encode log-transformed counts while evaluating reconstruction on the original scale.
        if self.count_data:
            view_data = torch.log1p(view_data)

        base_covariates = self._base_covariates(batch_cov, view_data)
        celltype_covariates = self._celltype_covariates(label_cov, view_data)

        prior_input = self._cat_nonempty([omics_id, celltype_covariates])
        prior_mu, prior_logvar = torch.chunk(self.prior_net_specific(prior_input), 2, dim=-1)

        shared_input_parts = [view_data]
        if self.encoder_covariates:
            shared_input_parts.append(base_covariates)
        shared_input_parts.append(celltype_covariates)
        shared_input = self._cat_nonempty(shared_input_parts)

        specific_input_parts = [view_data]
        if not self.multi_encoder:
            specific_input_parts.append(omics_id)
        if self.encoder_covariates:
            specific_input_parts.append(base_covariates)
        specific_input_parts.append(celltype_covariates)
        specific_input = self._cat_nonempty(specific_input_parts)

        if self.multi_encoder:
            z_shared, mu_shared, logvar_shared = self._route_encoder(
                self.encoder_shared,
                shared_input,
                omics_id,
                self.latent_dim_shared,
            )
            z_specific, mu_specific, logvar_specific = self._route_encoder(
                self.encoder_specific,
                specific_input,
                omics_id,
                self.latent_dim_specific,
            )
        else:
            z_shared, mu_shared, logvar_shared = self.encoder_shared(shared_input)
            z_specific, mu_specific, logvar_specific = self.encoder_specific(specific_input)

        z = torch.cat([z_shared, z_specific], dim=-1)
        decoder_covariates = self._decoder_covariates(batch_cov, omics_id, view_data)
        # Convert observation-domain indicators into per-feature reconstruction masks.
        if self.feat_mask is not None:
            mask = domain_id @ self.feat_mask
        else:
            mask = None

        if self.count_data:
            rho, dispersion, pi = self._route_decoder(self.decoder, z, decoder_covariates, omics_id, feature_mask=mask)
            s = self.sample_sequencing_depth(view_data_original)
        else:
            rho = self._route_decoder(self.decoder, z, decoder_covariates, omics_id, feature_mask=None)
            dispersion = None
            pi = None
            s = None

        if self.count_data:
            recon_loss = self.zinb_loss(view_data_original, rho, dispersion, pi, s, mask, eps=self.eps)
        else:
            recon_loss = mse_loss(view_data_original, rho, mask)

        # Reconstruct from shared factors alone to preserve information in the common latent space.
        if self.shared_recon_weight > 0:
            if self.count_data:
                shared_rho, shared_dispersion, shared_pi = self._route_decoder(
                    self.shared_factor_sufficiency_decoder,
                    z_shared,
                    decoder_covariates,
                    omics_id,
                    feature_mask=mask,
                )
                shared_factor_sufficiency_loss = self.zinb_loss(
                    view_data_original,
                    shared_rho,
                    shared_dispersion,
                    shared_pi,
                    s,
                    mask,
                    eps=self.eps,
                )
            else:
                shared_rho = self._route_decoder(
                    self.shared_factor_sufficiency_decoder,
                    z_shared,
                    decoder_covariates,
                    omics_id,
                    feature_mask=None,
                )
                shared_factor_sufficiency_loss = mse_loss(view_data_original, shared_rho, mask)
        else:
            shared_factor_sufficiency_loss = recon_loss.new_zeros(())

        # Match shared factors to a standard normal and specific factors to the learned conditional prior.
        kl_z = kl_loss_prior(mu_specific, logvar_specific, prior_mu, prior_logvar) + kl_loss(mu_shared, logvar_shared)
        total_loss = (
            recon_loss
            + self.shared_recon_weight * shared_factor_sufficiency_loss
            + self.beta * kl_z
        )

        loss_dict = {
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "shared_factor_sufficiency_loss": shared_factor_sufficiency_loss.item(),
            "kl_z": kl_z.item(),
        }
        return mu_shared, mu_specific, total_loss, loss_dict

    def decode_from_latent(self, z_concat, modality_onehot, covariates=None, library_size=None, feature_mask=None):
        if covariates is None:
            covariates = torch.zeros((z_concat.shape[0], self.covariate_dim), device=z_concat.device)
        decoder_covariates = self._decoder_covariates(covariates, modality_onehot, z_concat)
        if feature_mask is None:
            feature_mask = self._feature_mask_from_onehot(modality_onehot)

        if self.count_data:
            rho, _, _ = self._route_decoder(
                self.decoder,
                z_concat,
                decoder_covariates,
                modality_onehot,
                feature_mask=feature_mask,
            )
            if library_size is None:
                scale = torch.ones((z_concat.shape[0], 1), device=z_concat.device)
            else:
                scale = library_size.reshape(-1, 1).to(z_concat.device)
            return rho * scale
        return self._route_decoder(self.decoder, z_concat, decoder_covariates, modality_onehot, feature_mask=None)

    def decode_shared_from_latent(self, z_shared, modality_onehot, covariates=None, library_size=None, feature_mask=None):
        if covariates is None:
            covariates = torch.zeros((z_shared.shape[0], self.covariate_dim), device=z_shared.device)
        decoder_covariates = self._decoder_covariates(covariates, modality_onehot, z_shared)
        if feature_mask is None:
            feature_mask = self._feature_mask_from_onehot(modality_onehot)

        if self.count_data:
            rho, _, _ = self._route_decoder(
                self.shared_factor_sufficiency_decoder,
                z_shared,
                decoder_covariates,
                modality_onehot,
                feature_mask=feature_mask,
            )
            if library_size is None:
                scale = torch.ones((z_shared.shape[0], 1), device=z_shared.device)
            else:
                scale = library_size.reshape(-1, 1).to(z_shared.device)
            return rho * scale
        return self._route_decoder(self.shared_factor_sufficiency_decoder, z_shared, decoder_covariates, modality_onehot, feature_mask=None)

    # Use observed library sizes unless stochastic batch-level sampling is requested explicitly.
    @staticmethod
    def sample_sequencing_depth(x, strategy="observed"):
        if strategy == "batch_sample":
            mu_s = torch.log(x.sum(dim=1) + 1.0).mean()
            sigma_s = torch.log(x.sum(dim=1) + 1.0).std()
            log_s = mu_s + sigma_s * torch.randn(1, device=x.device)
            s = torch.exp(log_s).repeat(x.shape[0]).unsqueeze(1)
        else:
            log_s = torch.log(torch.clamp(x.sum(dim=1), min=1e-8)).unsqueeze(1)
            s = torch.exp(log_s)
        return s

