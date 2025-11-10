import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import traceback
from typing import Tuple

class ContinuousNLLLoss(nn.Module):
    """
    Continuous Negative Log-Likelihood Loss.
    Interpolates in the probability space and prevents log(0) after interpolation.
    """
    def __init__(self, o_grid, reduction='mean', input_is_log=False):
        super().__init__()
        # o_grid: Tensor, shape (n_bins,), represents the discretized grid for continuous values.
        self.register_buffer("o_grid", o_grid)
        self.reduction = reduction # 'mean' or 'none'
        self.input_is_log = input_is_log # If True, input is log-probabilities

    def forward(self, outputs, targets):
        batch_size = targets.size(0)
        n_bins = self.o_grid.size(0)
        if outputs.shape[-1] != n_bins or len(outputs.shape) != 2:
            # Attempt to reshape if outputs are not [batch_size, n_bins]
            outputs = outputs.reshape(batch_size, n_bins)

        # Add epsilon only to interpolated probabilities to ensure gradient and probability distribution reasonableness.
        if self.input_is_log:
            probs = outputs.exp()
        else:
            probs = outputs

        probs = probs / probs.sum(dim=-1, keepdim=True)  # Ensure normalization

        # Calculate interpolation parameters
        # Clamp targets to be within the defined grid
        targets = torch.clamp(targets, min=self.o_grid[0], max=self.o_grid[-1])
        # Calculate the relative position of targets in the grid
        ratio = (targets - self.o_grid[0]) / (self.o_grid[-1] - self.o_grid[0])
        scaled = ratio * (n_bins - 1) # Scale to bin indices
        # Find the lower bin index, clamp to avoid out-of-bounds
        lower_idx = torch.floor(scaled).long().clamp(0, n_bins - 2)
        upper_idx = lower_idx + 1 # Upper bin index

        batch_indices = torch.arange(batch_size, device=targets.device)
        # Calculate weights for interpolation
        upper_weight = scaled - lower_idx.float()
        lower_weight = 1.0 - upper_weight

        # Get probabilities for lower and upper bins
        lower_probs = probs[batch_indices, lower_idx]
        upper_probs = probs[batch_indices, upper_idx]
        # Perform linear interpolation of probabilities
        interp_probs = lower_weight * lower_probs + upper_weight * upper_probs

        # Apply epsilon correction only to the final interpolated probabilities to prevent log(0)
        losses = -torch.log(interp_probs + 1e-12)

        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'none':
            return losses # Return per-sample losses
        else:
            raise ValueError(f"Invalid reduction method: {self.reduction}")

def train_model(
    model,
    train_loader,
    val_loader,
    n_epochs=300,
    patience=10, # Early stopping patience
    device="cpu",
):
    # Initialize model and optimizer
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Assumes model has an 'o_grid' attribute for ContinuousNLLLoss
    criterion = ContinuousNLLLoss(model.o_grid)
    criterion.to(device) # Move criterion to device

    # Training state tracking
    best_val_loss = float("inf")
    best_model = None # To store the state_dict of the best model
    epochs_no_improve = 0 # Counter for early stopping
    history = {"train": [], "val": []} # To store loss history

    for epoch in range(n_epochs):
        # Training phase
        model.train() # Set model to training mode
        epoch_loss = 0.0
        with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1} Train") as pbar:
            for batch in pbar:
                inputs = batch["Total"].to(device) # Feature tensor
                targets = batch["offer"].to(device) # Target tensor

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Backward pass and optimization
                optimizer.zero_grad() # Clear previous gradients
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
                optimizer.step()

                # Update statistics
                epoch_loss += loss.item() * inputs.size(0) # Accumulate loss
                pbar.set_postfix({"loss": f"{loss.item():.4f}"}) # Update progress bar

        # Record training loss for the epoch
        history["train"].append(epoch_loss / len(train_loader.dataset))

        # Validation phase
        model.eval() # Set model to evaluation mode
        val_loss = 0.0
        with torch.no_grad(): # Disable gradient calculations
            for batch in val_loader:
                inputs = batch["Total"].to(device)
                targets = batch["offer"].to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item() * inputs.size(0)

        # Record validation loss for the epoch
        val_loss = val_loss / len(val_loader.dataset)
        history["val"].append(val_loss)

        # Print monitoring information
        print(f"Epoch {epoch+1:03d} | "
              f"Train Loss: {history['train'][-1]:.4f} | "
              f"Val Loss: {val_loss:.4f}")

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict() # Save the best model state
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load the best model found during training
    if best_model is not None:
        model.load_state_dict(best_model)
    return model, history

def test_model(model, test_loader, device="cpu"):
    """
    Test the model and return detailed information for each sample, including NLL loss.
    Args:
        model: Trained model, must have an 'o_grid' attribute.
        test_loader: DataLoader for the test set.
        device: Device to run the test on ("cpu" or "cuda").
    Returns:
        results: A list of dictionaries, each containing details for a sample.
    """
    model.to(device)
    model.eval() # Set model to evaluation mode

    # NLL loss (per-sample)
    # Assumes model has an 'o_grid' attribute
    criterion_nll = ContinuousNLLLoss(model.o_grid, reduction='none', input_is_log=False)
    criterion_nll.to(device) # Move criterion's buffers to the device

    results = []
    with torch.no_grad(): # Disable gradient calculations
        for batch in tqdm(test_loader, desc="Testing"):
            # Prepare data
            inputs  = batch["Total"].to(device)    # Assuming shape [B] or [B,...]
            targets = batch["offer"].to(device)    # Shape [B]

            # Forward pass
            outputs = model(inputs)                # Expected shape [B, n_bins]

            # 1) Per-sample NLL
            losses_nll = criterion_nll(outputs, targets)   # Shape [B]

            # Copy to CPU and convert to NumPy arrays
            # .squeeze() is used in case inputs are [B, 1]
            inp_cpu   = inputs.squeeze().cpu().numpy()
            tgt_cpu   = targets.cpu().numpy()
            nll_cpu   = losses_nll.cpu().numpy()

            # Assemble results
            # Handle the case where batch size is 1 and inp_cpu becomes a 0-dim array
            if inp_cpu.ndim == 0:
                inp_cpu_list = [inp_cpu.item()]
                tgt_cpu_list = [tgt_cpu.item()] # Assuming targets will also be scalar-like
                nll_cpu_list = [nll_cpu.item()] # Assuming nll will also be scalar-like
            else:
                inp_cpu_list = inp_cpu
                tgt_cpu_list = tgt_cpu
                nll_cpu_list = nll_cpu

            for i in range(len(inp_cpu_list)):
                results.append({
                    "Total":      float(inp_cpu_list[i]),
                    "offer":      float(tgt_cpu_list[i]),
                    "loss":       float(nll_cpu_list[i]),   # Original NLL
                })
    return results


def visualize_test(test_results, num_samples=20):
    """
    Visualize test results with summary statistics and samples.
    Args:
        test_results: Output from test_model (list of dictionaries).
        num_samples: Number of samples to display (default: 20).
    Returns:
        str: Formatted visualization report.
    """
    if not test_results:
        return "Empty test results"

    # Summary statistics
    try:
        # Ensure all dictionaries in test_results have the 'loss' key
        if not all('loss' in x for x in test_results):
             raise KeyError("'loss' key missing in one or more test result entries.")
        losses = [x['loss'] for x in test_results]
        total_samples = len(test_results)

        mean_loss = np.mean(losses)
        std_loss = np.std(losses)
        min_loss = np.min(losses)
        max_loss = np.max(losses)

        report = [
            "=== Summary Statistics ===",
            f"Total samples: {total_samples}",
            f"Mean loss: {mean_loss:.4f} ± {std_loss:.4f}", # Mean ± Standard Deviation
            f"Loss range: [{min_loss:.4f}, {max_loss:.4f}]" # Min loss, Max loss
        ]
    except Exception as e: # Catch potential errors during statistics calculation
        report = [
            "=== Summary Statistics ===",
            f"Total samples: {len(test_results)}",
            f"Could not calculate loss statistics: {str(e)}"
        ]

    # Sample selection (evenly spaced from sorted results)
    try:
        # Sort by loss (highest to lowest)
        # Ensure 'loss' key exists before sorting - already checked above, but good for robustness if called independently
        if not all('loss' in x for x in test_results):
            report.append("\n=== Sample Details ===")
            report.append("Cannot display samples: 'loss' key missing in some results.")
            return "\n".join(report)

        sorted_results = sorted(test_results, key=lambda x: x['loss'], reverse=True)

        # Cap the number of samples to display
        num_samples_to_display = min(num_samples, len(sorted_results))

        if num_samples_to_display > 0:
            report.append("\n=== Sample Details ===")

            # Calculate indices for evenly spaced samples from the sorted list
            if num_samples_to_display == 1:
                indices = [0] # Display the worst one
            else:
                # Get evenly spaced indices from the sorted list
                indices = [int(i * (len(sorted_results) - 1) / (num_samples_to_display - 1)) for i in range(num_samples_to_display)]

            # Add samples to the report
            for i, idx in enumerate(indices):
                s = sorted_results[idx]
                # Use .get for safer access to dictionary keys
                total_val = s.get('Total', 'N/A')
                offer_val = s.get('offer', 'N/A')
                loss_val = s.get('loss', 'N/A')

                # Ensure formatting works even if values are 'N/A'
                total_str = f"{total_val:.4f}" if isinstance(total_val, (int, float)) else str(total_val)
                offer_str = f"{offer_val:.4f}" if isinstance(offer_val, (int, float)) else str(offer_val)
                loss_str = f"{loss_val:.4f}" if isinstance(loss_val, (int, float)) else str(loss_val)

                report.append(
                    f"Sample {i+1} (rank {idx+1}/{total_samples}): "
                    f"Total={total_str} | Offer={offer_str} | Loss={loss_str}"
                )
    except Exception as e: # Catch potential errors during sample display
        report.append("\n=== Sample Details ===")
        report.append(f"Could not display samples: {str(e)}")

    return "\n".join(report)

def test_model_integrity(
    model,
    train_loader,
) -> Tuple[bool, str]:
    """
    Performs a forward and backward pass on all data and catches all possible errors.
    Checks if loss and gradients contain NaN or Inf values.

    Args:
        model: The model to test.
        train_loader: DataLoader for training data.

    Returns:
        Tuple[bool, str]: (test_passed, error_report)
            - If test passed: (True, "")
            - If test failed: (False, error_report)
    """
    # Save the model's original training state
    original_mode = model.training
    model.train()  # Set to training mode

    # Detect the model's device
    try:
        device = next(model.parameters()).device
    except StopIteration: # Model has no parameters
        # Fallback or raise an error if parameters are expected
        # For this function, if model has no parameters, it likely can't be trained in a standard way
        # However, we can try to proceed assuming it's a special case or let criterion creation fail
        device = torch.device("cpu") # Default device if no parameters
        # Or: return False, "Model has no parameters."

    # Create loss function
    # Assumes model has an 'o_grid' attribute
    try:
        criterion = ContinuousNLLLoss(model.o_grid)
        criterion.to(device)
    except AttributeError:
        model.train(original_mode) # Restore original mode
        return False, "Model does not have 'o_grid' attribute needed for ContinuousNLLLoss."


    batch_data_for_report = None # Store batch data for error reporting
    current_batch_idx = -1       # Store batch index for error reporting

    try:
        # Iterate through training data
        for idx, batch in enumerate(train_loader):
            current_batch_idx = idx
            batch_data_for_report = batch # Save for potential error report

            # Prepare input and target
            # Ensure keys "Total" and "offer" exist in the batch
            if "Total" not in batch or "offer" not in batch:
                raise KeyError("Batch is missing 'Total' or 'offer' key.")

            inputs = batch["Total"].to(device)
            targets = batch["offer"].to(device)

            # Clear previous gradients
            model.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Check if outputs contain NaN or Inf
            if torch.isnan(outputs).any():
                raise ValueError(f"Model outputs contain NaN values at batch index {current_batch_idx}")
            if torch.isinf(outputs).any():
                raise ValueError(f"Model outputs contain Inf values at batch index {current_batch_idx}")

            loss = criterion(outputs, targets)

            # Check if loss contains NaN or Inf
            if torch.isnan(loss).any():
                raise ValueError(f"Loss contains NaN values at batch index {current_batch_idx}")
            if torch.isinf(loss).any():
                raise ValueError(f"Loss contains Inf values at batch index {current_batch_idx}")

            # Backward pass
            loss.backward()

            # Check if gradients contain NaN or Inf
            problematic_params = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        problematic_params.append(f"{name} (NaN gradient)")
                    elif torch.isinf(param.grad).any():
                        problematic_params.append(f"{name} (Inf gradient)")

            if problematic_params:
                raise ValueError(f"Gradients contain NaN or Inf values in parameters: {', '.join(problematic_params)} at batch index {current_batch_idx}")

        # Clear gradients after successful test run
        model.zero_grad()

        # If completed successfully, restore model state and return success
        model.train(original_mode)
        return True, ""

    except Exception as e:
        # Catch exception, prepare error report
        error_type = type(e).__name__
        error_msg = str(e)
        stack_trace = traceback.format_exc()

        # Attempt to clear gradients
        try:
            model.zero_grad()
        except:
            pass  # Ignore any errors during gradient clearing in an error state

        # Limit the display of batch content
        batch_repr = "Batch data not available or error occurred before batch processing."
        if batch_data_for_report is not None:
            try:
                batch_repr = str({k: (v.shape if isinstance(v, torch.Tensor) else v) for k, v in batch_data_for_report.items()})
            except Exception as repr_e:
                batch_repr = f"Error creating batch representation: {str(repr_e)}"


        # Construct error report
        error_report = (
            f"--- Model Integrity Test Failed ---\n"
            f"Error Type: {error_type}\n"
            f"Error Message: {error_msg}\n"
            f"Failed at Batch Index: {current_batch_idx + 1 if current_batch_idx != -1 else 'N/A'} / {len(train_loader) if hasattr(train_loader, '__len__') else 'Unknown'}\n"
            f"Batch Content (Shapes/Values): {batch_repr}\n"
            f"Stack Trace:\n{stack_trace}"
        )

        # Restore model state and return failure
        model.train(original_mode)
        return False, error_report