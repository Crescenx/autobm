import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F
import json
import pandas as pd
import numpy as np

def evaluate_model(model, data_loader):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for features, labels in data_loader:
            human_hist = features["human_hist"]
            opponent_hist = features["opponent_hist"]
            timestep = features["timestep"]
            inputs = [human_hist, opponent_hist, timestep]

            probs = model(*inputs)
            loss = F.cross_entropy(probs, labels)
            test_loss += loss.item()
    return test_loss / len(data_loader)

def train_model(model, train_loader, val_loader, num_epochs=200, patience=15):
    from tqdm import tqdm
    import time
    import torch
    import pandas as pd

    # Build optimizer internally
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_checkpoint = None
    train_losses = []
    val_losses = []
    val_accs = []
    no_improvement_count = 0

    start_time = time.time()

    pbar = tqdm(range(num_epochs), desc="Training Progress")

    for epoch in pbar:

        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            human_hist = features["human_hist"]
            opponent_hist = features["opponent_hist"]
            timestep = features["timestep"]
            inputs = [human_hist, opponent_hist, timestep]

            optimizer.zero_grad()
            probs = model(*inputs)
            loss = F.cross_entropy(probs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        avg_val_loss = evaluate_model(model, val_loader)
        val_losses.append(avg_val_loss)

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for features, labels in val_loader:
                human_hist = features["human_hist"]
                opponent_hist = features["opponent_hist"]
                timestep = features["timestep"]
                inputs = [human_hist, opponent_hist, timestep]

                probs = model(*inputs)
                _, predicted = torch.max(probs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        val_accs.append(val_acc)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_acc
            best_checkpoint = model.state_dict()
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        pbar.set_postfix({
            'epoch': f'{epoch+1}/{num_epochs}',
            'train_loss': f'{avg_train_loss:.4f}',
            'val_loss': f'{avg_val_loss:.4f}',
            'val_acc': f'{val_acc:.2f}%',
            'best': f'{best_val_loss:.4f}',
            'time': time_str
        })

        if no_improvement_count >= patience:
            pbar.write(f"Early stopping at epoch {epoch + 1} due to no improvement in validation loss for {patience} epochs.")
            break

    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

    losses_df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'val_acc': val_accs
    })

    return {
        'best_checkpoint': best_checkpoint,
        'losses_df': losses_df,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc
    }

def test_model(model, test_loader):
    """
    Test the model and return detailed information for each sample.

    Args:
        model: Trained model
        test_loader: DataLoader for the test set

    Returns:
        results: A list of dictionaries, each containing details for a sample
    """
    model.eval()
    results = []

    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(test_loader):
            human_hist = features["human_hist"]
            opponent_hist = features["opponent_hist"]
            timestep = features["timestep"]
            inputs = [human_hist, opponent_hist, timestep]

            probs = model(*inputs)
            # Calculate per-sample losses
            losses = F.cross_entropy(probs, labels, reduction='none')
            _, predicted = torch.max(probs, 1)

            # Process each sample in the batch
            for i in range(labels.size(0)):
                sample_loss = losses[i].item()
                sample_correct = (predicted[i] == labels[i]).item()
                sample_accuracy = 100.0 if sample_correct else 0.0

                # Extract history information for this sample
                human_hist_sample = human_hist[i] if human_hist.dim() > 1 else human_hist
                opponent_hist_sample = opponent_hist[i] if opponent_hist.dim() > 1 else opponent_hist

                # Convert tensors to lists for better serialization
                human_hist_list = human_hist_sample.tolist()
                opponent_hist_list = opponent_hist_sample.tolist()
                timestep_val = timestep[i].item() if timestep.dim() > 0 else timestep.item()

                results.append({
                    'sample_id': batch_idx * labels.size(0) + i,
                    'loss': sample_loss,
                    'accuracy': sample_accuracy,
                    'predicted': predicted[i].item(),
                    'actual': labels[i].item(),
                    'correct': sample_correct,
                    'human_history': human_hist_list,
                    'opponent_history': opponent_hist_list,
                    'timestep': timestep_val
                })

    return results

def test_model_integrity(model, train_loader):
    model.train()

    features, labels = next(iter(train_loader))
    human_hist = features["human_hist"]
    opponent_hist = features["opponent_hist"]
    timestep = features["timestep"]
    inputs = [human_hist, opponent_hist, timestep]

    try:
        probs = model(*inputs)
    except Exception as e:
        return False, f"Error during forward pass: {e}"

    if torch.isnan(probs).any() or torch.isinf(probs).any():
        return False, "Error: Output contains NaN or inf values."


    try:
        loss = F.cross_entropy(probs, labels)
    except Exception as e:
        return False, f"Error calculating loss: {e}"

    if torch.isnan(loss).any() or torch.isinf(loss).any():
        return False, "Error: Loss is NaN or inf."

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    try:
        optimizer.zero_grad()
        loss.backward()
    except Exception as e:
        return False, f"Error during backward pass: {e}"

    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                return False, f"Error: Gradient for {name} contains NaN or inf values."

    try:
        optimizer.step()
    except Exception as e:
        return False, f"Error during parameter update: {e}"

    return True, None


def visualize_test(test_results, num_samples=3):
    """
    Visualize RPS test results with sample analysis.

    Args:
        test_results: List of dictionaries from test_model function
        num_samples: Number of high/low loss samples to display

    Returns:
        str: Formatted visualization report with sample analysis
    """
    if not test_results:
        return "No test results available"

    # Convert to numpy for easier analysis
    losses = np.array([r['loss'] for r in test_results])
    accuracies = np.array([r['accuracy'] for r in test_results])

    # Summary statistics
    mean_loss = np.mean(losses)
    std_loss = np.std(losses)
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    total_samples = len(test_results)

    # Find high and low loss samples
    sorted_indices = np.argsort(losses)
    low_loss_indices = sorted_indices[:num_samples]
    high_loss_indices = sorted_indices[-num_samples:][::-1]  # Reverse to get highest first

    # Map move indices to names
    move_names = {0: 'Rock', 1: 'Paper', 2: 'Scissors'}

    report = []
    report.append("=== RPS Model Test Results ===")
    report.append(f"Total samples: {total_samples}")
    report.append(f"Mean loss: {mean_loss:.4f} ± {std_loss:.4f}")
    report.append(f"Mean accuracy: {mean_accuracy:.2f}% ± {std_accuracy:.2f}%")
    report.append("")

    # High loss samples analysis
    report.append("=== High Loss Samples (Most Difficult to Predict) ===")
    for i, idx in enumerate(high_loss_indices):
        sample = test_results[idx]
        # Convert history to move names
        human_hist_moves = [move_names.get(int(move), str(move)) for move in sample['human_history']]
        opponent_hist_moves = [move_names.get(int(move), str(move)) for move in sample['opponent_history']]

        human_hist_str = ', '.join(human_hist_moves)
        opponent_hist_str = ', '.join(opponent_hist_moves)

        report.append(f"{i+1:2d}. Loss: {sample['loss']:.4f}")
        report.append(f"    Predicted: {move_names.get(sample['predicted'], 'Unknown')} | "
                     f"Actual: {move_names.get(sample['actual'], 'Unknown')} | "
                     f"Correct: {'Yes' if sample['correct'] else 'No'}")
        report.append(f"    Human History: [{human_hist_str}]")
        report.append(f"    Opponent History: [{opponent_hist_str}]")
        report.append(f"    Timestep: {sample['timestep']}")
        report.append("")

    # Low loss samples analysis
    report.append("=== Low Loss Samples (Easiest to Predict) ===")
    for i, idx in enumerate(low_loss_indices):
        sample = test_results[idx]
        # Convert history to move names
        human_hist_moves = [move_names.get(int(move), str(move)) for move in sample['human_history']]
        opponent_hist_moves = [move_names.get(int(move), str(move)) for move in sample['opponent_history']]

        human_hist_str = ', '.join(human_hist_moves)
        opponent_hist_str = ', '.join(opponent_hist_moves)

        report.append(f"{i+1:2d}. Loss: {sample['loss']:.4f}")
        report.append(f"    Predicted: {move_names.get(sample['predicted'], 'Unknown')} | "
                     f"Actual: {move_names.get(sample['actual'], 'Unknown')} | "
                     f"Correct: {'Yes' if sample['correct'] else 'No'}")
        report.append(f"    Human History: [{human_hist_str}]")
        report.append(f"    Opponent History: [{opponent_hist_str}]")
        report.append(f"    Timestep: {sample['timestep']}")
        report.append("")

    return "\n".join(report)
