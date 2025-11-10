"""
Model adapter to provide unified interface for all task scenarios.
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union


class ModelAdapter(ABC):
    """
    Abstract base class for model adapters.
    Provides unified interface for all task scenarios.
    """

    @abstractmethod
    def train(self, train_loader, val_loader, **kwargs) -> Tuple[Any, Dict]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            **kwargs: Additional training parameters

        Returns:
            Tuple of (trained_model, training_history)
        """
        pass

    @abstractmethod
    def test(self, test_loader) -> List[Dict]:
        """
        Test the model.

        Args:
            test_loader: Test data loader

        Returns:
            List of test results
        """
        pass

    @abstractmethod
    def integrity_check(self, train_loader) -> Tuple[bool, str]:
        """
        Check model integrity.

        Args:
            train_loader: Training data loader

        Returns:
            Tuple of (is_valid, error_message)
        """
        pass


class BatchModelAdapter(ModelAdapter):
    """
    Adapter for models that support batch processing (ulti, rps).
    These models are typically PyTorch nn.Module classes.
    """

    def __init__(self, model_class):
        self.model_class = model_class

    def train(self, train_loader, val_loader, **kwargs):
        from .pipeline import train_model
        model_instance = self.model_class()
        return train_model(model_instance, train_loader, val_loader, **kwargs)

    def test(self, test_loader):
        from .pipeline import test_model
        model_instance = self.model_class()
        return test_model(model_instance, test_loader)

    def integrity_check(self, train_loader):
        from .pipeline import test_model_integrity
        model_instance = self.model_class()
        return test_model_integrity(model_instance, train_loader)


class CDAAdapter(ModelAdapter):
    """
    Adapter for CDA models that don't support batch processing.
    These models use function-based interfaces.
    """

    def __init__(self, compute_strategy_fn, init_params_fn):
        self.compute_strategy_fn = compute_strategy_fn
        self.init_params_fn = init_params_fn
        self.trained_params = None

    def train(self, train_loader, val_loader, **kwargs):
        from .implementations.cda_pipeline import train_model
        # Convert DataLoader to list of samples for CDA
        train_samples = self._dataloader_to_samples(train_loader)
        val_samples = self._dataloader_to_samples(val_loader)

        # Initialize parameters
        params = self.init_params_fn()

        # Train the model
        self.trained_params = train_model(
            self.compute_strategy_fn,
            params,
            train_samples,
            val_samples,
            **kwargs
        )
        return self.trained_params, {}

    def test(self, test_loader):
        from .implementations.cda_pipeline import test_model
        # Convert DataLoader to list of samples for CDA
        test_samples = self._dataloader_to_samples(test_loader)

        # Check if we have trained parameters
        if self.trained_params is None:
            # If not, initialize parameters for testing
            params = self.init_params_fn()
        else:
            params = self.trained_params

        return test_model(self.compute_strategy_fn, params, test_samples)

    def integrity_check(self, train_loader):
        from .implementations.cda_pipeline import test_model_integrity
        # Convert DataLoader to list of samples for CDA
        train_samples = self._dataloader_to_samples(train_loader)

        # Initialize parameters for integrity check
        params = self.init_params_fn()

        return test_model_integrity(self.compute_strategy_fn, params, train_samples)

    def _dataloader_to_samples(self, dataloader):
        """Convert DataLoader to list of samples for CDA."""
        from .implementations.cda_pipeline import Sample, MarketState
        samples = []
        for batch in dataloader:
            # CDA data loader has batch_size=1, so we can extract the single item
            features = batch[0]  # features dict
            label = batch[1]     # label tensor

            # Create MarketState from features
            state = MarketState(
                H_prices=features["H_prices"].squeeze(0),
                H_expired=features["H_expired"].squeeze(0),
                Q_prices=features["Q_prices"].squeeze(0),
                Q_from_current=features["Q_from_current"].squeeze(0),
                A_prices=features["A_prices"].squeeze(0),
                P_series=features["P_series"].squeeze(0),
                current_time=features["current_time"].squeeze(0)
            )

            # Create Sample
            sample = Sample(
                state=state,
                bid=label.item() if label.numel() == 1 else label.squeeze(0).item()
            )
            samples.append(sample)
        return samples