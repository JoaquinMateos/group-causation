'''
Torch-based wrappers for the iVAE models adapted to the project API.
'''

import logging
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from .nets import DiscreteIVAE, DiscreteVAE, VAE, iVAE


def _to_2d_float_array(array: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError(f'{name} must be a 2D numpy array. Got shape {values.shape}.')
    return values


def _safe_standardize(array: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = array.mean(axis=0, keepdims=True)
    std = array.std(axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (array - mean) / std, mean, std


def _tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


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
        device: str = 'cpu',
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
        self.embedding_: Optional[np.ndarray] = None
        self.device_: str = device if torch.cuda.is_available() and device == 'cuda' \
                    else device if torch.backends.mps.is_available() and device == 'mps' \
                    else 'cpu'
        self.data_dim_: Optional[int] = None
        self.aux_dim_: Optional[int] = None
        self.latent_dim_: Optional[int] = None
        self.x_mean_: Optional[np.ndarray] = None
        self.x_std_: Optional[np.ndarray] = None
        self.u_mean_: Optional[np.ndarray] = None
        self.u_std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, U: Optional[np.ndarray] = None):
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        x_values = _to_2d_float_array(X, 'X')
        if self.standardize:
            x_values, self.x_mean_, self.x_std_ = _safe_standardize(x_values)
        else:
            self.x_mean_ = np.zeros((1, x_values.shape[1]), dtype=np.float32)
            self.x_std_ = np.ones((1, x_values.shape[1]), dtype=np.float32)

        if self.use_auxiliary:
            if U is None:
                u_values = np.zeros((x_values.shape[0], 1), dtype=np.float32)
            else:
                u_values = _to_2d_float_array(U, 'U')
            if u_values.shape[0] != x_values.shape[0]:
                raise ValueError('X and U must have the same number of rows.')
            if self.standardize:
                u_values, self.u_mean_, self.u_std_ = _safe_standardize(u_values)
            else:
                self.u_mean_ = np.zeros((1, u_values.shape[1]), dtype=np.float32)
                self.u_std_ = np.ones((1, u_values.shape[1]), dtype=np.float32)
        else:
            u_values = None
            self.u_mean_ = None
            self.u_std_ = None

        logging.info(f'Using device: {self.device_}')

        latent_dim = self.inference_dim if self.inference_dim is not None else self.latent_dim
        self.latent_dim_ = latent_dim
        self.data_dim_ = x_values.shape[1]
        self.aux_dim_ = 0 if u_values is None else u_values.shape[1]

        if self.use_auxiliary:
            self.model_ = self._build_auxiliary_model(latent_dim, self.data_dim_, self.aux_dim_, self.device_)
        else:
            self.model_ = self._build_model(latent_dim, self.data_dim_, self.device_)

        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.1,
            patience=self.scheduler_tol,
            mode='max',
        )

        tensors = [torch.from_numpy(x_values)]
        if u_values is not None:
            tensors.append(torch.from_numpy(u_values))
        train_dataset = TensorDataset(*tensors)
        loader_kwargs = {'num_workers': 1, 'pin_memory': True} if self.device_ != 'cpu' else {}
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size, **loader_kwargs)

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
                    batch_x = batch_x.to(self.device_)
                    batch_u = batch_u.to(self.device_)
                    if self.anneal and hasattr(self.model_, 'anneal'):
                        self.model_.anneal(x_values.shape[0], self.max_iter, steps + 1)
                    elbo, _ = self.model_.elbo(batch_x, batch_u)
                else:
                    (batch_x,) = batch
                    batch_x = batch_x.to(self.device_)
                    elbo, _ = self.model_.elbo(batch_x)

                (-elbo).backward()
                optimizer.step()

                epoch_elbo += float(elbo.detach().cpu().item())
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

    def transform(self, X: np.ndarray, U: Optional[np.ndarray] = None) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError('The reducer must be fitted before calling transform().')

        x_values = self._prepare_x_for_inference(X)
        if self.use_auxiliary:
            u_values = self._prepare_u_for_inference(U, x_values.shape[0])
            return self._encode(x_values, u_values)

        return self._encode(x_values)

    def fit_transform(self, X: np.ndarray, U: Optional[np.ndarray] = None) -> np.ndarray:
        self.fit(X, U)
        if self.embedding_ is None:
            raise RuntimeError('The fitted embedding is not available.')
        return self.embedding_

    def _build_model(self, latent_dim: int, data_dim: int, device: str):
        if self.discrete:
            return DiscreteVAE(
                latent_dim,
                data_dim,
                activation=self.activation,
                n_layers=self.n_layers,
                hidden_dim=self.hidden_dim,
                device=device,
                slope=self.slope,
            )

        return VAE(
            latent_dim,
            data_dim,
            activation=self.activation,
            n_layers=self.n_layers,
            hidden_dim=self.hidden_dim,
            device=device,
            slope=self.slope,
        )

    def _build_auxiliary_model(self, latent_dim: int, data_dim: int, aux_dim: int, device: str):
        if self.discrete:
            return DiscreteIVAE(
                latent_dim,
                data_dim,
                aux_dim,
                activation=self.activation,
                n_layers=self.n_layers,
                hidden_dim=self.hidden_dim,
                device=device,
                slope=self.slope,
            )

        return iVAE(
            latent_dim,
            data_dim,
            aux_dim,
            activation=self.activation,
            device=device,
            n_layers=self.n_layers,
            hidden_dim=self.hidden_dim,
            slope=self.slope,
            anneal=self.anneal,
        )

    def _prepare_x_for_inference(self, X: np.ndarray) -> np.ndarray:
        x_values = _to_2d_float_array(X, 'X')
        if self.standardize:
            if self.x_mean_ is None or self.x_std_ is None:
                raise RuntimeError('Feature normalization statistics are not available.')
            x_values = (x_values - self.x_mean_) / self.x_std_
        return x_values

    def _prepare_u_for_inference(self, U: Optional[np.ndarray], n_samples: int) -> np.ndarray:
        if U is None:
            if self.aux_dim_ is None or self.aux_dim_ == 0:
                raise RuntimeError('Auxiliary dimension is not available.')
            u_values = np.zeros((n_samples, self.aux_dim_), dtype=np.float32)
        else:
            u_values = _to_2d_float_array(U, 'U')

        if u_values.shape[0] != n_samples:
            raise ValueError('X and U must have the same number of rows.')

        if self.standardize:
            if self.u_mean_ is None or self.u_std_ is None:
                raise RuntimeError('Auxiliary normalization statistics are not available.')
            u_values = (u_values - self.u_mean_) / self.u_std_

        return u_values

    def _encode(self, X: np.ndarray, U: Optional[np.ndarray] = None) -> np.ndarray:
        self.model_.eval()
        with torch.no_grad():
            x_tensor = torch.from_numpy(X).to(self.device_)
            if self.use_auxiliary:
                if U is None:
                    raise ValueError('Auxiliary data is required for this reducer.')
                u_tensor = torch.from_numpy(U).to(self.device_)
                encoder_params = self.model_.encoder_params(x_tensor, u_tensor)
            else:
                encoder_params = self.model_.encoder_params(x_tensor)

            latent = encoder_params[0]
        return _tensor_to_numpy(latent)

    def _collect_model_params(self, X: np.ndarray, U: Optional[np.ndarray] = None) -> Dict[str, Tuple[np.ndarray, ...]]:
        self.model_.eval()
        with torch.no_grad():
            x_tensor = torch.from_numpy(self._prepare_x_for_inference(X)).to(self.device_)
            if self.use_auxiliary:
                u_tensor = torch.from_numpy(self._prepare_u_for_inference(U, x_tensor.shape[0])).to(self.device_)
                decoder_params, encoder_params, _, prior_params = self.model_.forward(x_tensor, u_tensor)
            else:
                decoder_params, encoder_params, _, prior_params = self.model_.forward(x_tensor)

        return {
            'decoder': tuple(_tensor_to_numpy(item) for item in decoder_params),
            'encoder': tuple(_tensor_to_numpy(item) for item in encoder_params),
            'prior': tuple(_tensor_to_numpy(item) for item in prior_params),
        }


class IVAEDimensionalityReduction(_TorchLatentReducer):
    '''
    Dimensionality reduction for time series using the iVAE architecture.

    The model takes a data matrix with shape (n_samples, n_variables) and an optional
    auxiliary matrix with shape (n_samples, n_auxiliary_variables). When the auxiliary
    matrix is omitted, a constant one-dimensional context is used.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, use_auxiliary=True, **kwargs)


class VAEDimensionalityReduction(_TorchLatentReducer):
    '''
    Dimensionality reduction for time series using the VAE architecture.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, use_auxiliary=False, **kwargs)


def IVAE_wrapper(
    X,
    U=None,
    batch_size=256,
    max_iter=7e4,
    seed=None,
    n_layers=3,
    hidden_dim=200,
    lr=1e-2,
    device='cpu',
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
    device='cpu',
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