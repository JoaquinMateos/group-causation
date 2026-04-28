import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from .nets import DiscreteIVAE, DiscreteVAE, VAE, iVAE


def _to_2d_float_tensor(data: Union[np.ndarray, torch.Tensor], name: str, device: Union[str, torch.device]) -> torch.Tensor:
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data).to(device, dtype=torch.float32)
    else:
        tensor = data.clone().detach().to(device, dtype=torch.float32)
        
    if tensor.ndim == 1:
        tensor = tensor.view(-1, 1)
    if tensor.ndim != 2:
        raise ValueError(f'{name} must be a 2D tensor. Got shape {tensor.shape}.')
    return tensor


def _safe_standardize(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, unbiased=False, keepdim=True)
    std = torch.where(std == 0, torch.ones_like(std), std)
    return (tensor - mean) / std, mean, std


class _TorchLatentReducer:
    def __init__(
        self,
        latent_dim: int = 2,
        batch_size: int = 256,
        max_iter: int = 70000,
        seed: Optional[int] = None,
        n_layers: int = 3,
        hidden_dim: int = 200,
        lr: float = 1e-2,
        device: Union[str, torch.device] = 'cpu',
        activation: str = 'lrelu',
        slope: float = 0.1,
        discrete: bool = False,
        inference_dim: Optional[int] = None,
        anneal: bool = False,
        scheduler_tol: int = 3,
        standardize: bool = True,
        use_auxiliary: bool = True,
    ):
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.max_iter = int(max_iter)
        self.seed = seed
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.activation = activation
        self.slope = slope
        self.discrete = discrete
        self.inference_dim = inference_dim
        self.anneal = anneal
        self.scheduler_tol = scheduler_tol
        self.standardize = standardize
        self.use_auxiliary = use_auxiliary

        self.model_: Any = None
        self.history_: list[float] = []
        self.params_: Dict[str, Any] = {}
        self.embedding_: Optional[torch.Tensor] = None
        self.device: Union[str, torch.device] = device
        self.data_dim_: Optional[int] = None
        self.aux_dim_: Optional[int] = None
        self.latent_dim_: Optional[int] = None
        self.x_mean_: Optional[torch.Tensor] = None
        self.x_std_: Optional[torch.Tensor] = None
        self.u_mean_: Optional[torch.Tensor] = None
        self.u_std_: Optional[torch.Tensor] = None

    def fit(self, X: Union[np.ndarray, torch.Tensor], U: Optional[Union[np.ndarray, torch.Tensor]] = None):
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        x_tensor = _to_2d_float_tensor(X, 'X', self.device)
        if self.standardize:
            x_tensor, self.x_mean_, self.x_std_ = _safe_standardize(x_tensor)
        else:
            self.x_mean_ = torch.zeros((1, x_tensor.shape[1]), dtype=torch.float32, device=self.device)
            self.x_std_ = torch.ones((1, x_tensor.shape[1]), dtype=torch.float32, device=self.device)

        if self.use_auxiliary:
            if U is None:
                logging.warning('use_auxiliary is True but no auxiliary data provided. Using a constant auxiliary variable.')
                u_tensor = torch.zeros((x_tensor.shape[0], 1), dtype=torch.float32, device=self.device)
            else:
                u_tensor = _to_2d_float_tensor(U, 'U', self.device)
                
            if u_tensor.shape[0] != x_tensor.shape[0]:
                raise ValueError('X and U must have the same number of rows.')
                
            if self.standardize:
                u_tensor, self.u_mean_, self.u_std_ = _safe_standardize(u_tensor)
            else:
                self.u_mean_ = torch.zeros((1, u_tensor.shape[1]), dtype=torch.float32, device=self.device)
                self.u_std_ = torch.ones((1, u_tensor.shape[1]), dtype=torch.float32, device=self.device)
        else:
            u_tensor = None
            self.u_mean_ = None
            self.u_std_ = None

        logging.info(f'Using device: {self.device}')

        latent_dim = self.inference_dim if self.inference_dim is not None else self.latent_dim
        latent_dim = int(np.ceil(float(latent_dim)))
        if latent_dim < 1:
            raise ValueError(f'latent_dim must be >= 1. Got {latent_dim}.')
            
        self.latent_dim_ = latent_dim
        self.data_dim_ = x_tensor.shape[1]
        self.aux_dim_ = 0 if u_tensor is None else u_tensor.shape[1]

        if self.use_auxiliary:
            self.model_ = self._build_auxiliary_model(latent_dim, self.data_dim_, self.aux_dim_, self.device)
        else:
            self.model_ = self._build_model(latent_dim, self.data_dim_, self.device)

        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=self.scheduler_tol, mode='max'
        )

        tensors = [x_tensor]
        if u_tensor is not None:
            tensors.append(u_tensor)
            
        train_dataset = TensorDataset(*tensors)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=0)

        self.model_.train()
        self.history_ = []
        steps = 0

        while steps < self.max_iter:
            epoch_elbo = 0.0
            batch_count = 0

            for batch in train_loader:
                if steps >= self.max_iter:
                    break

                optimizer.zero_grad()

                if self.use_auxiliary:
                    batch_x, batch_u = batch
                    if self.anneal and hasattr(self.model_, 'anneal'):
                        self.model_.anneal(x_tensor.shape[0], self.max_iter, steps + 1)
                    elbo, _ = self.model_.elbo(batch_x, batch_u)
                else:
                    (batch_x,) = batch
                    elbo, _ = self.model_.elbo(batch_x)

                (-elbo).backward()
                optimizer.step()

                epoch_elbo += float(elbo.detach().item())
                batch_count += 1
                steps += 1

            if batch_count == 0:
                break

            mean_elbo = epoch_elbo / batch_count
            self.history_.append(mean_elbo)
            scheduler.step(mean_elbo)

        self.embedding_ = self.transform(X, U)
        self.params_ = self._collect_model_params(X, U)
        return self

    def transform(self, X: Union[np.ndarray, torch.Tensor], U: Optional[Union[np.ndarray, torch.Tensor]] = None) -> torch.Tensor:
        if self.model_ is None:
            raise RuntimeError('The reducer must be fitted before calling transform().')

        x_tensor = self._prepare_x_for_inference(X)
        if self.use_auxiliary:
            u_tensor = self._prepare_u_for_inference(U, x_tensor.shape[0])
            return self._encode(x_tensor, u_tensor)

        return self._encode(x_tensor)

    def fit_transform(self, X: Union[np.ndarray, torch.Tensor], U: Optional[Union[np.ndarray, torch.Tensor]] = None) -> torch.Tensor:
        self.fit(X, U)
        if self.embedding_ is None:
            raise RuntimeError('The fitted embedding is not available.')
        return self.embedding_

    def _build_model(self, latent_dim: int, data_dim: int, device: Union[str, torch.device]):
        if self.discrete:
            return DiscreteVAE(
                latent_dim, data_dim, activation=self.activation, n_layers=self.n_layers,
                hidden_dim=self.hidden_dim, device=device, slope=self.slope,
            )

        return VAE(
            latent_dim, data_dim, activation=self.activation, n_layers=self.n_layers,
            hidden_dim=self.hidden_dim, device=device, slope=self.slope,
        )

    def _build_auxiliary_model(self, latent_dim: int, data_dim: int, aux_dim: int, device: Union[str, torch.device]):
        if self.discrete:
            return DiscreteIVAE(
                latent_dim, data_dim, aux_dim, activation=self.activation, n_layers=self.n_layers,
                hidden_dim=self.hidden_dim, device=device, slope=self.slope,
            )

        return iVAE(
            latent_dim, data_dim, aux_dim, activation=self.activation, device=device,
            n_layers=self.n_layers, hidden_dim=self.hidden_dim, slope=self.slope, anneal=self.anneal,
        )

    def _prepare_x_for_inference(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        x_tensor = _to_2d_float_tensor(X, 'X', self.device)
        if self.standardize:
            if self.x_mean_ is None or self.x_std_ is None:
                raise RuntimeError('Feature normalization statistics are not available.')
            x_tensor = (x_tensor - self.x_mean_) / self.x_std_
        return x_tensor

    def _prepare_u_for_inference(self, U: Optional[Union[np.ndarray, torch.Tensor]], n_samples: int) -> torch.Tensor:
        if U is None:
            if self.aux_dim_ is None or self.aux_dim_ == 0:
                raise RuntimeError('Auxiliary dimension is not available.')
            u_tensor = torch.zeros((n_samples, self.aux_dim_), dtype=torch.float32, device=self.device)
        else:
            u_tensor = _to_2d_float_tensor(U, 'U', self.device)

        if u_tensor.shape[0] != n_samples:
            raise ValueError('X and U must have the same number of rows.')

        if self.standardize:
            if self.u_mean_ is None or self.u_std_ is None:
                raise RuntimeError('Auxiliary normalization statistics are not available.')
            u_tensor = (u_tensor - self.u_mean_) / self.u_std_

        return u_tensor

    def _encode(self, X: torch.Tensor, U: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.model_.eval()
        with torch.no_grad():
            if self.use_auxiliary:
                if U is None:
                    raise ValueError('Auxiliary data is required for this reducer.')
                encoder_params = self.model_.encoder_params(X, U)
            else:
                encoder_params = self.model_.encoder_params(X)

            latent = encoder_params[0]
        return latent

    def _collect_model_params(self, X: Union[np.ndarray, torch.Tensor], U: Optional[Union[np.ndarray, torch.Tensor]] = None) -> Dict[str, Tuple[torch.Tensor, ...]]:
        self.model_.eval()
        with torch.no_grad():
            x_tensor = self._prepare_x_for_inference(X)
            if self.use_auxiliary:
                u_tensor = self._prepare_u_for_inference(U, x_tensor.shape[0])
                decoder_params, encoder_params, _, prior_params = self.model_.forward(x_tensor, u_tensor)
            else:
                decoder_params, encoder_params, _, prior_params = self.model_.forward(x_tensor)

        return {
            'decoder': tuple(item for item in decoder_params),
            'encoder': tuple(item for item in encoder_params),
            'prior': tuple(item for item in prior_params),
        }


class IVAEDimensionalityReduction(_TorchLatentReducer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, use_auxiliary=True, **kwargs)


class VAEDimensionalityReduction(_TorchLatentReducer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, use_auxiliary=False, **kwargs)


def IVAE_wrapper(
    X,
    U,
    batch_size=256,
    max_iter=7e4,
    seed=None,
    n_layers=3,
    hidden_dim=200,
    lr=1e-2,
    device: Union[str, torch.device] = 'cpu',
    activation='lrelu',
    slope=.1,
    discrete=False,
    inference_dim=None,
    anneal=False,
    scheduler_tol=3,
):
    reducer = IVAEDimensionalityReduction(
        latent_dim=inference_dim if inference_dim is not None else 2,
        batch_size=batch_size,
        max_iter=int(max_iter),
        seed=seed,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        lr=lr,
        device=device,
        activation=activation,
        slope=slope,
        discrete=discrete,
        inference_dim=inference_dim,
        anneal=anneal,
        scheduler_tol=scheduler_tol,
    )
    latent = reducer.fit_transform(X, U)
    return latent, reducer.model_, reducer.params_, {'elbo': reducer.history_}


def VAE_wrapper(
    X,
    S=None,
    batch_size=256,
    max_iter=7e4,
    seed=None,
    n_layers=3,
    hidden_dim=200,
    lr=1e-2,
    device: Union[str, torch.device] = 'cpu',
    activation='lrelu',
    slope=.1,
    discrete=False,
    inference_dim=None,
    log_folder=None,
    ckpt_folder=None,
    scheduler_tol=3,
):
    reducer = VAEDimensionalityReduction(
        latent_dim=inference_dim if inference_dim is not None else 2,
        batch_size=batch_size,
        max_iter=int(max_iter),
        seed=seed,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        lr=lr,
        device=device,
        activation=activation,
        slope=slope,
        discrete=discrete,
        inference_dim=inference_dim,
        scheduler_tol=scheduler_tol,
    )
    latent = reducer.fit_transform(X)
    return latent, reducer.model_, reducer.params_, {'elbo': reducer.history_}