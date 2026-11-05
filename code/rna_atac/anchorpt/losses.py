"""Implement reconstruction and latent regularization losses for AnchorPT."""

import torch
import torch.nn as nn


class ZINBLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, rho, dispersion, pi, s, mask=None, eps=1e-8):
        mean = torch.clamp(rho * s, min=eps)
        dispersion = torch.clamp(dispersion, min=eps)

        # Evaluate the negative-binomial negative log-likelihood elementwise.
        t1 = torch.lgamma(dispersion) + torch.lgamma(x + 1.0) - torch.lgamma(x + dispersion)
        t2 = -dispersion * torch.log(dispersion) - x * torch.log(mean) + (dispersion + x) * torch.log(dispersion + mean)
        nb_final = t1 + t2

        # Combine structural zeros with the negative-binomial probability of zero.
        zero_nb = torch.exp(dispersion * (torch.log(dispersion) - torch.log(dispersion + mean)))
        zero_case = -torch.log(pi + (1.0 - pi) * zero_nb + eps)
        nb_case = nb_final - torch.log(1.0 - pi + eps)

        loss = torch.where(x <= eps, zero_case, nb_case)
        # Scale masked feature sums back to the full feature dimension.
        if mask is not None:
            norm = torch.clamp(torch.sum(mask, dim=1), min=1.0)
            mean_loss = torch.mean(torch.sum(loss * mask, dim=1) * (x.shape[1] / norm), dim=0)
        else:
            mean_loss = torch.mean(torch.sum(loss, dim=1), dim=0)
        return mean_loss


def mse_loss(x, y, mask=None):
    loss = (x - y).pow(2)
    if mask is not None:
        norm = torch.clamp(torch.sum(mask, dim=1), min=1.0)
        mean_loss = torch.mean(torch.sum(loss * mask, dim=1) * (x.shape[1] / norm), dim=0)
    else:
        mean_loss = torch.mean(torch.sum(loss, dim=1), dim=0)
    return mean_loss


# KL divergence from the shared posterior to a standard normal prior.
def kl_loss(mu, logvar):
    return torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar), dim=1), dim=0)


# KL divergence between diagonal Gaussian posterior and learned prior.
def kl_loss_prior(mu_q, logvar_q, mu_p, logvar_p):
    kl = -0.5 * torch.sum(
        -torch.exp(logvar_q - logvar_p)
        - ((mu_q - mu_p).pow(2)) * torch.exp(-logvar_p)
        + 1
        - logvar_p
        + logvar_q,
        dim=1,
    )
    return torch.mean(kl, dim=0)

