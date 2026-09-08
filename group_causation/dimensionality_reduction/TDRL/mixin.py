from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

from group_causation.dimensionality_reduction.dimensionality_reduction_base import DimensionalityReduction



class PLDimensionalityReductionMixin(DimensionalityReduction):
    """
    Mixin que implementa los métodos de DimensionalityReduction 
    directamente dentro de un LightningModule.
    """
    def _prepare_dataloader(self, X: Any, U: Optional[Any] = None, batch_size: int = 32) -> DataLoader:
        """
        Acondiciona la entrada para que coincida con la tupla (x, z, c) 
        que esperan los métodos training_step de los modelos.
        """
        if isinstance(X, DataLoader):
            return X
        elif isinstance(X, torch.Tensor):
            # Tensores dummy para 'z' (latentes reales) y 'c' (dominios)
            z_dummy = torch.zeros(X.shape[0], X.shape[1], self.z_dim)
            c_dummy = U if U is not None else torch.zeros(X.shape[0], dtype=torch.long)
            dataset = TensorDataset(X, z_dummy, c_dummy)
            return DataLoader(dataset, batch_size=batch_size, shuffle=True)
        else:
            raise TypeError("X debe ser un torch.Tensor o un torch.utils.data.DataLoader")

    def fit(self, X: Any, U: Optional[Any] = None, **kwargs) -> 'DimensionalityReduction':
        # Extraemos los parámetros específicos de la interfaz
        batch_size = kwargs.pop('batch_size', 32)
        val_dataloaders = kwargs.pop('val_dataloaders', None)
        
        train_loader = self._prepare_dataloader(X, U, batch_size)
        
        # Instanciamos el Trainer de Lightning con el resto de kwargs (ej. max_epochs, accelerator)
        trainer = pl.Trainer(**kwargs)
        
        # Como este Mixin se usará junto a un LightningModule, 'self' representa el propio modelo
        trainer.fit(self, train_dataloaders=train_loader, val_dataloaders=val_dataloaders)
        return self

    def transform(self, X: Any, U: Optional[Any] = None, **kwargs) -> Any:
        self.eval()
        batch_size = kwargs.get('batch_size', 32)
        
        if isinstance(X, DataLoader):
            data_loader = X
        elif isinstance(X, torch.Tensor):
            dataset = TensorDataset(X)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        else:
            raise TypeError("X debe ser un torch.Tensor o un torch.utils.data.DataLoader")

        embeddings = []
        device = next(self.parameters()).device
        
        with torch.no_grad():
            for batch in data_loader:
                x_batch = batch[0] if isinstance(batch, (list, tuple)) else batch
                x_batch = x_batch.to(device)
                
                # self.net devuelve: (x_recon, mus, logvars, z_est)
                _, mus, _, _ = self.net(x_batch)
                embeddings.append(mus.cpu())
                
        return torch.cat(embeddings, dim=0)