'''
Code for iVAE, adapted from the original implementation by Gresele et al. (2020) at
https://github.com/ilkhem/iVAE

Streamlined and refactored for clarity, keeping only the core iVAE and VAE models.
'''

import math
from numbers import Number
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F


# =========================================================================
# UTILITIES & HELPER FUNCTIONS
# =========================================================================

def weights_init(m: nn.Module):
    """Initializes linear layer weights using Xavier Uniform."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)


def reparameterize_gaussian(mean: torch.Tensor, var: Optional[torch.Tensor] = None, logvar: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Standard reparameterization trick for Gaussian distributions: z = mean + eps * std
    Accepts either variance (`var`) or log-variance (`logvar`).
    """
    if logvar is not None:
        std = torch.exp(0.5 * logvar)
    elif var is not None:
        std = torch.sqrt(var)
    else:
        raise ValueError("Must provide either 'var' or 'logvar'")
        
    eps = torch.randn_like(std)
    return mean + eps * std


def _check_inputs(size: Optional[torch.Size], mu: Optional[torch.Tensor], v: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Helper function to ensure distribution inputs are compatible and correctly broadcasted."""
    if size is None and mu is None and v is None:
        raise ValueError("Inputs can't all be None")
    
    if size is not None:
        if mu is None:
            mu = torch.tensor([0.0])
        if v is None:
            v = torch.tensor([1.0])
        if isinstance(v, Number):
            v = torch.tensor([v]).type_as(mu)
        return mu.expand(size), v.expand(size)
        
    elif mu is not None and v is not None:
        if isinstance(v, Number):
            v = torch.tensor([v]).type_as(mu)
        if v.size() != mu.size():
            v = v.expand(mu.size())
        return mu, v
        
    elif mu is not None:
        v = torch.tensor([1.0]).type_as(mu).expand(mu.size())
        return mu, v
        
    elif v is not None:
        mu = torch.tensor([0.0]).type_as(v).expand(v.size())
        return mu, v
    
    raise ValueError(f'Invalid inputs: size={size}, mu_v={(mu, v)})')


# =========================================================================
# DISTRIBUTIONS 
# =========================================================================

class Dist:
    def sample(self, *args): pass
    def log_pdf(self, *args, **kwargs): pass


class Normal(Dist):
    """ Isotropic Gaussian Distribution wrapper """
    def __init__(self, device: Union[str, torch.device] = 'cpu'):
        super().__init__()
        self.device = device
        self.c = torch.tensor(2 * np.pi).to(self.device)
        self.name = 'gauss'

    def sample(self, mu: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return reparameterize_gaussian(mu, var=v)

    def log_pdf(self, x: torch.Tensor, mu: torch.Tensor, v: torch.Tensor, reduce: bool = True, param_shape: Optional[Tuple] = None) -> torch.Tensor:
        if param_shape is not None:
            mu, v = mu.view(param_shape), v.view(param_shape)
        lpdf = -0.5 * (torch.log(self.c) + v.log() + (x - mu).pow(2).div(v))
        return lpdf.sum(dim=-1) if reduce else lpdf


# =========================================================================
# NEURAL NETWORK BUILDING BLOCKS
# =========================================================================

class MLP(nn.Module):
    """ Multi-Layer Perceptron used as the primary building block for Encoders and Decoders. """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: Union[int, List[int]], 
                 n_layers: int, activation: Union[str, List[str]] = 'none', 
                 slope: float = 0.1, device: Union[str, torch.device] = 'cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.device = device
        
        if isinstance(hidden_dim, Number):
            self.hidden_dim = [int(hidden_dim)] * (self.n_layers - 1) # type: ignore
        elif isinstance(hidden_dim, list):
            self.hidden_dim = hidden_dim
        else:
            raise ValueError(f'Wrong argument type for hidden_dim: {hidden_dim}')

        if isinstance(activation, str):
            self.activation = [activation] * (self.n_layers - 1)
        elif isinstance(activation, list):
            self.activation = activation 
        else:
            raise ValueError(f'Wrong argument type for activation: {activation}')

        self._act_f = []
        for act in self.activation:
            if act == 'lrelu':
                self._act_f.append(lambda x: F.leaky_relu(x, negative_slope=slope))
            elif act == 'xtanh':
                self._act_f.append(lambda x: self.xtanh(x, alpha=slope))
            elif act == 'sigmoid':
                self._act_f.append(torch.sigmoid)
            elif act == 'silu':
                self._act_f.append(F.silu)
            elif act == 'none':
                self._act_f.append(lambda x: x)
            else:
                raise ValueError(f'Incorrect activation: {act}')

        _fc_list = []
        if self.n_layers == 1:
            _fc_list.append(nn.Linear(self.input_dim, self.output_dim))
        else:
            _fc_list.append(nn.Linear(self.input_dim, self.hidden_dim[0]))
            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
            _fc_list.append(nn.Linear(self.hidden_dim[-1], self.output_dim))
            
        self.fc = nn.ModuleList(_fc_list)
        self.to(self.device)

    @staticmethod
    def xtanh(x: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
        """Tanh function plus an additional linear term to prevent vanishing gradients."""
        return x.tanh() + alpha * x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for c in range(self.n_layers):
            if c == self.n_layers - 1:
                h = self.fc[c](h) # No activation on output layer
            else:
                h = self._act_f[c](self.fc[c](h))
        return h


# =========================================================================
# CORE VAE / iVAE MODELS
# =========================================================================

class iVAE(nn.Module):
    """ Identifiable Variational Autoencoder leveraging an auxiliary variable 'u'. """
    def __init__(self, latent_dim: int, data_dim: int, aux_dim: int, prior=None, decoder=None, encoder=None,
                 n_layers: int = 3, hidden_dim: int = 50, activation: str = 'lrelu', slope: float = 0.1, 
                 device: Union[str, torch.device] = 'cpu', anneal: bool = False):
        super().__init__()
        
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.aux_dim = aux_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.slope = slope
        self.anneal_params = anneal

        self.prior_dist = Normal(device=device) if prior is None else prior
        self.decoder_dist = Normal(device=device) if decoder is None else decoder
        self.encoder_dist = Normal(device=device) if encoder is None else encoder

        # Prior MLP: Predicts variance based on aux variable `u` (mean is fixed to 0)
        self.prior_mean = torch.zeros(1).to(device)
        self.logl = MLP(aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope, device=device)
        
        # Decoder MLP: Generates data `x` from latent `z`
        self.f = MLP(latent_dim, data_dim, hidden_dim, n_layers, activation=activation, slope=slope, device=device)
        self.decoder_var = 0.01 * torch.ones(1).to(device)
        
        # Encoder MLPs: Infers latent `z` from data `x` + aux `u`
        self.g = MLP(data_dim + aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope, device=device)
        self.logv = MLP(data_dim + aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope, device=device)

        self.apply(weights_init)
        
        # Hyperparameters for ELBO decomposition/annealing: [a, b, c, d, N]
        self._training_hyperparams = [1.0, 1.0, 1.0, 1.0, 1]

    def encoder_params(self, x: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xu = torch.cat((x, u), 1)
        enc_mean = self.g(xu)
        enc_logvar = self.logv(xu)
        enc_var = torch.clamp(enc_logvar.exp(), min=1e-5)
        return enc_mean, enc_var

    def decoder_params(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        recon_mean = self.f(z)
        return recon_mean, self.decoder_var

    def prior_params(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        prior_logvar = self.logl(u)
        return self.prior_mean, prior_logvar.exp()

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> Tuple[Tuple, Tuple, torch.Tensor, Tuple]:
        prior_params = self.prior_params(u)
        encoder_params = self.encoder_params(x, u)
        
        z = self.encoder_dist.sample(*encoder_params)
        decoder_params = self.decoder_params(z)
        
        return decoder_params, encoder_params, z, prior_params

    def elbo(self, x: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        decoder_params, encoder_params, z, prior_params = self.forward(x, u)
        
        # Extract individual parameters for calculating probabilities
        enc_mean, enc_var = encoder_params
        
        log_p_x_given_z = self.decoder_dist.log_pdf(x, *decoder_params)
        log_q_z_given_x_u = self.encoder_dist.log_pdf(z, enc_mean, enc_var)
        log_p_z_given_u = self.prior_dist.log_pdf(z, *prior_params)

        if self.anneal_params:
            a, b, c, d, N = self._training_hyperparams
            M = z.size(0)
            
            # Decomposed Q(z) approximations for Total Correlation (TC)
            log_q_z_tmp = self.encoder_dist.log_pdf(
                z.view(M, 1, self.latent_dim), 
                enc_mean.view(1, M, self.latent_dim),
                enc_var.view(1, M, self.latent_dim), 
                reduce=False
            )
            
            log_q_z = torch.logsumexp(log_q_z_tmp.sum(dim=-1), dim=1, keepdim=False) - np.log(M * N)
            log_q_z_indep = (torch.logsumexp(log_q_z_tmp, dim=1, keepdim=False) - np.log(M * N)).sum(dim=-1)

            elbo_val = (
                a * log_p_x_given_z 
                - b * (log_q_z_given_x_u - log_q_z) 
                - c * (log_q_z - log_q_z_indep) 
                - d * (log_q_z_indep - log_p_z_given_u)
            ).mean()
            return elbo_val, z
        else:
            # Standard ELBO calculation
            return (log_p_x_given_z + log_p_z_given_u - log_q_z_given_x_u).mean(), z

    def anneal(self, N: int, max_epoch: int, epoch: int):
        """ Updates ELBO weights dynamically over time during training. """
        thr = int(max_epoch / 1.6)
        
        self._training_hyperparams[-1] = N
        
        # Smoothly transition weights to exactly 1.0 at 'thr'
        if epoch <= thr:
            progress = epoch / thr
            
            # Start 'a' slightly higher if you want to prioritize early reconstruction, 
            # but smoothly decay it to 1.0. 
            self._training_hyperparams[0] = 2.0 - progress       # a: 2.0 -> 1.0
            self._training_hyperparams[1] = progress             # b: 0.0 -> 1.0
            self._training_hyperparams[2] = progress             # c: 0.0 -> 1.0
            self._training_hyperparams[3] = progress             # d: 0.0 -> 1.0
            self.anneal_params = True
        else:
            # After threshold, fall back to standard unweighted ELBO
            self.anneal_params = False


class VAE(nn.Module):
    """ Standard Variational Autoencoder. """
    def __init__(self, latent_dim: int, data_dim: int, decoder=None, encoder=None,
                 n_layers: int = 3, hidden_dim: int = 50, activation: str = 'lrelu', slope: float = 0.1, 
                 device: Union[str, torch.device] = 'cpu'):
        super().__init__()
        
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.slope = slope

        self.decoder_dist = Normal(device=device) if decoder is None else decoder
        self.encoder_dist = Normal(device=device) if encoder is None else encoder
        self.prior_dist = Normal(device=device)
        
        self.prior_mean = torch.zeros(1).to(device)
        self.prior_var = torch.ones(1).to(device)

        # Decoder MLP
        self.f = MLP(latent_dim, data_dim, hidden_dim, n_layers, activation=activation, slope=slope, device=device)
        self.decoder_var = 0.01 * torch.ones(1).to(device)
        
        # Encoder MLPs
        self.g = MLP(data_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope, device=device)
        self.logv = MLP(data_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope, device=device)

        self.apply(weights_init)

    def encoder_params(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        enc_mean = self.g(x)
        enc_logvar = self.logv(x)
        return enc_mean, enc_logvar.exp()

    def decoder_params(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        recon_mean = self.f(z)
        return recon_mean, self.decoder_var

    def forward(self, x: torch.Tensor) -> Tuple[Tuple, Tuple, torch.Tensor, Tuple]:
        encoder_params = self.encoder_params(x)
        z = self.encoder_dist.sample(*encoder_params)
        decoder_params = self.decoder_params(z)
        
        return decoder_params, encoder_params, z, (self.prior_mean, self.prior_var)

    def elbo(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        decoder_params, encoder_params, z, prior_params = self.forward(x)
        
        log_p_x_given_z = self.decoder_dist.log_pdf(x, *decoder_params)
        log_q_z_given_x = self.encoder_dist.log_pdf(z, *encoder_params)
        log_p_z = self.prior_dist.log_pdf(z, *prior_params)

        return (log_p_x_given_z + log_p_z - log_q_z_given_x).mean(), z