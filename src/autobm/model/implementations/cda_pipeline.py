from typing import Any, Callable, Tuple, Dict, List
import torch
from dataclasses import asdict
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import time
import numpy as np
from dataclasses import dataclass

@dataclass
class MarketState:
    H_prices: torch.Tensor
    H_expired: torch.Tensor 
    Q_prices: torch.Tensor
    Q_from_current: torch.Tensor
    A_prices: torch.Tensor
    P_series: torch.Tensor
    current_time: torch.Tensor  

@dataclass
class Sample:
    state: MarketState
    bid: float  



def compute_interpolated_loss(
    q_values: torch.Tensor,
    probs: torch.Tensor,
    target_bid: float,
    *,
    epsilon: float = 1e-10
) -> torch.Tensor:
    """
    Computes a more precise loss using interpolation.

    Args:
        q_values: Predicted Q-value vector (n_actions,)
        probs: Action probability distribution (n_actions,)
        target_bid: Target bid value
        epsilon: Numerical stability term

    Returns:
        Interpolated loss value
    """
    # Normalize probs
    probs = probs / (probs.sum())

    # Find the two actions closest to the target bid
    bid_diff = torch.abs(q_values - target_bid)
    top2 = torch.topk(-bid_diff, k=2)  # Get indices of the two closest

    # Calculate weights (linear interpolation based on distance)
    d1, d2 = bid_diff[top2.indices[0]], bid_diff[top2.indices[1]]
    w1 = d2 / (d1 + d2 + epsilon)
    w2 = d1 / (d1 + d2 + epsilon)

    # Interpolated probability
    interpolated_prob = w1 * probs[top2.indices[0]] + w2 * probs[top2.indices[1]]

    # Negative log-likelihood loss
    return -torch.log(interpolated_prob + epsilon)


def train_model(
    model_fn: Callable[[dict, MarketState], Tuple[torch.Tensor, torch.Tensor]],
    init_params: dict,
    train_samples: List[Sample],
    val_samples: List[Sample],
    *,
    n_epochs: int = 200,
    learning_rate: float = 0.001,
    patient: int = 10, # Renamed from patience to patient to match usage
    min_delta: float = 0.0001,
    seed: int = 42
) -> dict:
    """
    Trains the model with early stopping.
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Check for empty samples
    if len(train_samples) == 0:
        raise ValueError("Training samples list is empty")
    if len(val_samples) == 0:
        raise ValueError("Validation samples list is empty")

    for name, tensor in init_params.items():
        if not tensor.is_leaf or not tensor.requires_grad:
            raise ValueError(f"Parameter {name} is invalid: must be a trainable leaf tensor.")

    params = init_params
    optimizer = torch.optim.Adam([{"params": list(params.values())}], lr=learning_rate)

    # Early stopping variables
    best_loss = float('inf')
    best_params = {k: v.detach().clone() for k, v in params.items()}
    no_improve = 0

    start_time = time.time()
    timeout_seconds = 600  # 10 minutes

    for epoch in range(n_epochs):
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            raise TimeoutError(f"Training exceeded 10 minutes, terminated. (Ran for {elapsed/60:.2f} minutes)")

        # Training phase
        params = {k: v.requires_grad_(True) for k, v in params.items()}
        train_loss = 0.0

        shuffled_indices = torch.randperm(len(train_samples)).tolist()
        for i in shuffled_indices:
            optimizer.zero_grad()
            sample = train_samples[i]

            q_values, probs = model_fn(params, sample.state)
            loss = compute_interpolated_loss(q_values, probs, sample.bid)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params.values(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        val_loss = 0.0
        with torch.no_grad():
            for sample in val_samples:
                q_values, probs = model_fn(params, sample.state)
                val_loss += compute_interpolated_loss(q_values, probs, sample.bid).item()

        train_loss /= len(train_samples)
        val_loss /= len(val_samples)

        # Early stopping check
        if val_loss < (best_loss - min_delta):
            best_loss = val_loss
            best_params = {k: v.detach().clone() for k, v in params.items()}
            no_improve = 0
        else:
            no_improve += 1

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:03d}/{n_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"LR: {current_lr:.2e} | "
              f"Best Val: {best_loss:.4f} | "
              f"NoImprove: {no_improve}/{patient}")

        if no_improve >= patient:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return best_params


def test_model(
    model_fn: Callable[[dict, MarketState], Tuple[torch.Tensor, torch.Tensor]],
    params: dict,
    test_samples: List[Sample]
) -> List[Dict[str, Any]]:
    """
    Tests the model and computes loss and MSE for each sample.

    Args:
        model_fn: Policy calculation function.
        params: Trained parameter dictionary.
        test_samples: List of test samples.

    Returns:
        List of dictionaries containing complete sample information,
        predictions, loss, and MSE.
    """
    results = []
    with torch.no_grad():
        for sample in test_samples:
            q_values, probs = model_fn(params, sample.state)
            # Predicted bid is the Q-value of the action with the highest probability
            predicted_bid = q_values[torch.argmax(probs)].item()
            loss = compute_interpolated_loss(q_values, probs, sample.bid).item()
            true_bid = sample.bid.item() if isinstance(sample.bid, torch.Tensor) else sample.bid
            mse = (predicted_bid - true_bid) ** 2

            sample_dict = {
                **asdict(sample.state),
                "bid": sample.bid,
                "predicted_bid": predicted_bid,
                "loss": loss,
                "mse": mse
            }
            # Convert tensors to native Python types or numpy arrays for easier handling
            for k, v in sample_dict.items():
                if isinstance(v, torch.Tensor):
                    sample_dict[k] = v.cpu().numpy()
            results.append(sample_dict)
    return results


def visualize_test(
    test_results: List[Dict[str, Any]],
    params: dict,
) -> str:
    return visualize_test_results(test_results, params, n_samples=2, loss_bins=10)

def visualize_test_results(
    test_results: List[Dict[str, Any]],
    params: dict,
    n_samples: int = 2,
    loss_bins: int = 10
) -> str:
    """
    Generates a visualization report for test results.

    Args:
        test_results: List of test result dictionaries.
        params: Trained parameter dictionary (values are torch.Tensor).
        n_samples: Number of samples to show per loss bin.
        loss_bins: Number of loss intervals.

    Returns:
        Formatted report string.
    """
    if not test_results:
        return "No test results available"

    report = []

    # 1. Model Parameters
    report.append("Model Parameters:")
    for name, tensor in params.items():
        arr = tensor.detach().cpu().numpy()
        arr_str = np.array2string(arr, precision=4, separator=', ',
                                 threshold=10, edgeitems=3)
        report.append(f"  {name}: {arr_str}")

    # 2. Test Results Summary
    losses = np.array([r['loss'] for r in test_results])
    bid_diffs = np.abs(
        np.array([r['bid'] for r in test_results]) -
        np.array([r['predicted_bid'] for r in test_results])
    )
    report.append("\nTest Results Summary")
    report.append(f"Mean loss: {losses.mean():.4f} ± {losses.std():.4f}")
    report.append(f"Mean bid difference: {bid_diffs.mean():.4f} ± {bid_diffs.std():.4f}")

    # 3. Display samples binned by loss
    if len(test_results) > n_samples * loss_bins:
        bins = np.linspace(losses.min(), losses.max(), loss_bins + 1)
        report.append("\nSampled results by loss range:")
        for i in range(loss_bins):
            lower, upper = bins[i], bins[i+1]
            # Ensure the last bin includes the maximum value
            if i == loss_bins - 1:
                mask = (losses >= lower) & (losses <= upper)
            else:
                mask = (losses >= lower) & (losses < upper)
            idxs = np.where(mask)[0]
            if idxs.size == 0:
                continue
            report.append(f"\nLoss range [{lower:.4f}, {upper:.4f}] ({idxs.size} samples):")
            sampled_indices = np.random.choice(idxs, size=min(n_samples, idxs.size), replace=False)
            for j, idx in enumerate(sampled_indices, 1):
                sample = test_results[idx]
                report.append(f"\nSample {j}:")
                report.append(f"  True bid: {sample['bid']:.4f}")
                report.append(f"  Predicted bid: {sample['predicted_bid']:.4f}")
                report.append(f"  Loss: {sample['loss']:.4f}")
                report.append(f"  Bid difference: {abs(sample['bid'] - sample['predicted_bid']):.4f}")
                report.append("\n  MarketState:")
                for k, v in sample.items():
                    if k in ('bid', 'predicted_bid', 'loss', 'mse'):
                        continue
                    if isinstance(v, np.ndarray):
                        if v.ndim == 1:
                            if v.dtype == bool:
                                elems = ', '.join(str(x).lower() for x in v)
                            else:
                                elems = ', '.join(f"{x:.4f}" for x in v)
                            report.append(f"    {k}: [{elems}]")
                        else:
                            arr_str = np.array2string(v, precision=4, separator=', ')
                            report.append(f"    {k}: {arr_str}")
                    elif isinstance(v, bool):
                        report.append(f"    {k}: {str(v).lower()}")
                    else:
                        report.append(f"    {k}: {v}")
    return "\n".join(report)


def test_model_integrity(
    model_fn: Callable[[dict, MarketState], Tuple[torch.Tensor, torch.Tensor]],
    params: dict,
    test_samples: List[Sample]
) -> Tuple[bool, str]:
    """
    Performs a simplified model integrity check.
    Checks parameter types, requires_grad, forward pass, and backward pass for NaNs/Infs.
    """
    def prepare_sample_dict(sample: Sample) -> dict:
        sample_dict = {**asdict(sample.state), "bid": sample.bid}
        for k, v in sample_dict.items():
            if isinstance(v, torch.Tensor):
                sample_dict[k] = v.cpu().numpy().tolist()
        return sample_dict

    try:
        # Check parameter types and requires_grad
        for name, tensor in params.items():
            if not isinstance(tensor, torch.Tensor):
                return False, f"Parameter '{name}' type error: expected torch.Tensor, got {type(tensor)}"
            if not tensor.requires_grad: # Keep this check if parameters are expected to be trainable post-training for some reason
                return False, f"Parameter '{name}' requires_grad=False"

        # Forward pass check
        for idx, sample in enumerate(test_samples):
            try:
                with torch.no_grad():
                    q_values, probs = model_fn(params, sample.state)
                    if not (isinstance(q_values, torch.Tensor) and isinstance(probs, torch.Tensor)):
                        raise TypeError("Output must be two torch.Tensors")
                    if q_values.dim() != 1 or probs.dim() != 1:
                        raise ValueError(f"Output dimensions error (q_values:{q_values.dim()}D, probs:{probs.dim()}D)")
            except Exception as e:
                return False, (
                    f"Forward failed - Sample {idx+1}/{len(test_samples)}\n"
                    f"{type(e).__name__}: {e}\n"
                    f"Sample: {prepare_sample_dict(sample)}"
                )

        # Backward pass check (only checks for NaN/Inf in gradients)
        # This makes a mutable copy of params for the backward check
        temp_params = {k: v.clone().requires_grad_(True) for k, v in params.items()}
        for idx, sample in enumerate(test_samples[:10]): # Check a subset for backward pass
            try:
                for v_param in temp_params.values(): # Use v_param to avoid conflict with outer v
                    if v_param.grad is not None:
                        v_param.grad.zero_()

                q_values, probs = model_fn(temp_params, sample.state)
                loss = compute_interpolated_loss(q_values, probs, sample.bid)
                loss.backward()

                for name, param_tensor in temp_params.items(): # Use param_tensor to avoid conflict
                    grad = param_tensor.grad
                    if grad is not None:
                        if torch.isnan(grad).any():
                            raise FloatingPointError(f"Parameter '{name}' grad has NaN")
                        if torch.isinf(grad).any():
                            raise FloatingPointError(f"Parameter '{name}' grad has Inf")
            except Exception as e:
                return False, (
                    f"Backward failed - Sample {idx+1}/{min(len(test_samples),10)}\n"
                    f"{type(e).__name__}: {e}\n"
                    f"Sample: {prepare_sample_dict(sample)}"
                )

        return True, ""

    except Exception as e:
        return False, f"Integrity test error: {e}"


def visualize_brief(
    test_results: List[Dict[str, Any]],
    params: dict
) -> str:
    """
    Generates a brief report showing model parameters and overall loss/MSE statistics.

    Args:
        test_results: List of dictionaries, each containing 'loss' and 'mse' keys.
        params: Trained parameter dictionary (values are torch.Tensor).

    Returns:
        Brief report string.
    """
    if not test_results:
        return "No test results available"

    report = []

    # 1. Model Parameters
    report.append("Model Parameters:")
    for name, tensor in params.items():
        arr = tensor.detach().cpu().numpy()
        arr_str = np.array2string(
            arr,
            precision=4,
            separator=', ',
            threshold=10,
            edgeitems=3
        )
        report.append(f"  {name}: {arr_str}")

    # 2. Global loss / MSE statistics
    losses = np.array([r['loss'] for r in test_results], dtype=float)
    mses   = np.array([r['mse']  for r in test_results], dtype=float)

    report.append("\nTest Metrics Summary:")
    report.append(f"  Loss: mean = {losses.mean():.4f} ± {losses.std():.4f}")
    report.append(f"  MSE : mean = {mses.mean():.4f} ± {mses.std():.4f}")

    return "\n".join(report)