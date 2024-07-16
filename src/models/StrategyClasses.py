import torch
import torch.nn.functional as F

from src.models.AbstractStrategyInterfaces import LossFunctionStrategy


class NLLLossStrategy(LossFunctionStrategy):
    def compute_loss(self, recon_mu, recon_logvar, x, mu, logvar):
        # TODO: Add dimension as input to the function
        # Negative log-likelihood for Gaussian
        recon_var = torch.exp(recon_logvar)
        nll_loss = 0.5 * torch.sum(
            recon_logvar
            + (x.view(-1, 2) - recon_mu) ** 2 / recon_var
            + torch.log(torch.tensor(2 * torch.pi))
        )

        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return nll_loss + KL, nll_loss, KL


class MSELossStrategy(LossFunctionStrategy):
    def compute_loss(self, recon_x, x, mu, logvar):
        MSE = F.mse_loss(recon_x, x, reduction="sum")
        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + KL, MSE, KL
