import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Any
from dataclasses import dataclass
import json
import os
import toml
from torch.nn.utils.rnn import pad_sequence

# --- Global Configuration ---
DEFAULT_SPLIT_RATIOS = (0.7, 0.15, 0.15)
INTERNAL_SEED = 42
INTERNAL_BATCH_SIZE_ULTI_RPS = 32
ENV_CURRENT_TASK = "CURRENT_TASK" # Environment variable name (kept for compatibility)
GLOBAL_FILE_PREFIX = "./src/autobm/data/datasets/"

# Read current_task from pyproject.toml
def get_current_task():
    try:
        pyproject_data = toml.load("pyproject.toml")
        autobm_config = pyproject_data.get("tool", {}).get("autobm", {})
        return autobm_config.get("current_task", "ulti")
    except Exception as e:
        print(f"Warning: Could not read current_task from pyproject.toml: {e}")
        return "ulti"  # Default fallback

# --- Common Dataset Class ---
class DictDataset(Dataset):
    """Custom dataset class that handles dictionary-like samples."""
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]

# --- Task 1: Ultimatum Game Dataset (_ulti) ---
def create_dataset_ulti(
    csv_file: str = GLOBAL_FILE_PREFIX+'proposer.csv', # Default data path
    batch_size: int = 32, # Will receive value from router
    seed: int = 42      # Will receive value from router
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates DataLoaders for the ultimatum game task.
    Uses DEFAULT_SPLIT_RATIOS.
    """
    if not abs(sum(DEFAULT_SPLIT_RATIOS) - 1.0) < 1e-9: # Should always be true with constant
        raise ValueError("Split ratios must sum to 1.")
    if len(DEFAULT_SPLIT_RATIOS) != 3:
        raise ValueError("Split ratios must contain three values.")

    if not os.path.exists(csv_file):
        print(f"Warning: CSV file {csv_file} not found for ulti task. Returning empty DataLoaders.")
        empty_dataset = DictDataset([])
        return (
            DataLoader(empty_dataset, batch_size=batch_size),
            DataLoader(empty_dataset, batch_size=batch_size),
            DataLoader(empty_dataset, batch_size=batch_size),
        )
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV file {csv_file}: {e}. Returning empty DataLoaders.")
        empty_dataset = DictDataset([])
        return (DataLoader(empty_dataset, batch_size=batch_size), DataLoader(empty_dataset, batch_size=batch_size), DataLoader(empty_dataset, batch_size=batch_size))

    if 'Total' not in df.columns or 'offer' not in df.columns:
        raise ValueError(f"CSV file {csv_file} must contain 'Total' and 'offer' columns.")

    totals = df['Total'].values.astype(np.float32)
    offers = df['offer'].values.astype(np.float32)
    dataset_list = [{'Total': torch.tensor(t, dtype=torch.float32), 'offer': torch.tensor(o, dtype=torch.float32)} for t, o in zip(totals, offers)]

    if not dataset_list:
        print(f"Warning: No data from {csv_file}. Empty DataLoaders.")
        empty_dataset = DictDataset([])
        return (DataLoader(empty_dataset, batch_size=batch_size), DataLoader(empty_dataset, batch_size=batch_size), DataLoader(empty_dataset, batch_size=batch_size))
        
    np.random.seed(seed)
    indices = np.random.permutation(len(dataset_list))
    n_samples = len(dataset_list)
    split_train = int(n_samples * DEFAULT_SPLIT_RATIOS[0])
    split_val = split_train + int(n_samples * DEFAULT_SPLIT_RATIOS[1])
    train_indices, val_indices, test_indices = indices[:split_train], indices[split_train:split_val], indices[split_val:]
    
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(DictDataset([dataset_list[i] for i in train_indices]), batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(DictDataset([dataset_list[i] for i in val_indices]), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(DictDataset([dataset_list[i] for i in test_indices]), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# --- Task 2: Rock-Paper-Scissors Sequential Prediction Dataset (_rps) ---
def _collate_rps_features(batch_features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not batch_features: return {"human_hist": torch.empty(0,0,dtype=torch.long), "opponent_hist": torch.empty(0,0,dtype=torch.long), "timestep": torch.empty(0,dtype=torch.long)}
    return {
        "human_hist": pad_sequence([item["human_hist"] for item in batch_features], batch_first=True, padding_value=0),
        "opponent_hist": pad_sequence([item["opponent_hist"] for item in batch_features], batch_first=True, padding_value=0),
        "timestep": torch.stack([item["timestep"] for item in batch_features])
    }

def rps_collate_fn(batch: List[Dict[str, Any]]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    if not batch: return _collate_rps_features([]), torch.empty(0, dtype=torch.long)
    features_list = [item['features'] for item in batch]
    labels_list = [item['label'] for item in batch]
    batched_features = _collate_rps_features(features_list)
    batched_labels = torch.stack(labels_list).squeeze() if labels_list else torch.empty(0, dtype=torch.long)
    return batched_features, batched_labels

def _preprocess_rps_data_to_list(raw_data: List[Dict[str, list]]) -> List[Dict[str, Any]]:
    processed_list = []
    min_hist_length = 1
    for sample in raw_data:
        human_actions, opponent_actions = sample.get("human", []), sample.get("opponent", [])
        if len(human_actions) < min_hist_length + 1: continue
        for t_hist_len in range(min_hist_length, len(human_actions)):
            current_opponent_hist = opponent_actions[:t_hist_len] if len(opponent_actions) >= t_hist_len else opponent_actions[:]
            processed_list.append({
                'features': {"human_hist": torch.tensor(human_actions[:t_hist_len],dtype=torch.long), "opponent_hist": torch.tensor(current_opponent_hist,dtype=torch.long), "timestep": torch.tensor(t_hist_len,dtype=torch.long)},
                'label': torch.tensor(human_actions[t_hist_len],dtype=torch.long)
            })
    return processed_list

def create_dataset_rps(
    json_file: str = GLOBAL_FILE_PREFIX+'rps.json', # Default data path
    batch_size: int = 32, # Will receive value from router
    seed: int = 42      # Will receive value from router
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates DataLoaders for RPS sequential prediction.
    Uses DEFAULT_SPLIT_RATIOS.
    """
    if not os.path.exists(json_file):
        print(f"Warning: JSON file {json_file} not found for RPS. Empty DataLoaders.")
        empty_dataset = DictDataset([])
        return (DataLoader(empty_dataset, batch_size=batch_size, collate_fn=rps_collate_fn), DataLoader(empty_dataset, batch_size=batch_size, collate_fn=rps_collate_fn), DataLoader(empty_dataset, batch_size=batch_size, collate_fn=rps_collate_fn))
    try:
        with open(json_file, 'r') as f: raw_data_json = json.load(f)
    except Exception as e:
        print(f"Error reading JSON {json_file}: {e}. Empty DataLoaders.")
        empty_dataset = DictDataset([])
        return (DataLoader(empty_dataset, batch_size=batch_size, collate_fn=rps_collate_fn), DataLoader(empty_dataset, batch_size=batch_size, collate_fn=rps_collate_fn), DataLoader(empty_dataset, batch_size=batch_size, collate_fn=rps_collate_fn))

    all_samples_list = _preprocess_rps_data_to_list(raw_data_json)
    if not all_samples_list:
        print(f"Warning: No samples from {json_file} for RPS. Empty DataLoaders.")
        empty_dataset = DictDataset([])
        return (DataLoader(empty_dataset, batch_size=batch_size, collate_fn=rps_collate_fn), DataLoader(empty_dataset, batch_size=batch_size, collate_fn=rps_collate_fn), DataLoader(empty_dataset, batch_size=batch_size, collate_fn=rps_collate_fn))

    np.random.seed(seed)
    indices = np.random.permutation(len(all_samples_list))
    n_samples = len(all_samples_list)
    split_train = int(n_samples * DEFAULT_SPLIT_RATIOS[0])
    split_val = split_train + int(n_samples * DEFAULT_SPLIT_RATIOS[1])
    train_indices, val_indices, test_indices = indices[:split_train], indices[split_train:split_val], indices[split_val:]
    
    train_dataset, val_dataset, test_dataset = DictDataset([all_samples_list[i] for i in train_indices]), DictDataset([all_samples_list[i] for i in val_indices]), DictDataset([all_samples_list[i] for i in test_indices])
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=rps_collate_fn, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=rps_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=rps_collate_fn)
    return train_loader, val_loader, test_loader

# --- Task 3: Continuous Double Auction Market Bidding Dataset (_cda) ---
@dataclass
class MarketStateCDA:
    H_prices: torch.Tensor; H_expired: torch.Tensor; Q_prices: torch.Tensor; Q_from_current: torch.Tensor
    A_prices: torch.Tensor; P_series: torch.Tensor; current_time: torch.Tensor

def cda_collate_fn(batch: List[Dict[str, Any]]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    if not batch:
        empty_f = {"H_prices": torch.empty(0,0,dtype=torch.float32),"H_expired": torch.empty(0,0,dtype=torch.bool),"Q_prices": torch.empty(0,0,dtype=torch.float32),"Q_from_current": torch.empty(0,0,dtype=torch.bool),"A_prices": torch.empty(0,0,dtype=torch.float32),"P_series": torch.empty(0,0,dtype=torch.float32),"current_time": torch.empty(0,dtype=torch.float32)}
        return empty_f, torch.empty(0, dtype=torch.float32)

    features_list, labels_list = [item['features'] for item in batch], [item['label'] for item in batch]
    collated_features = {}
    float_keys, bool_keys = ["H_prices","Q_prices","A_prices","P_series"], ["H_expired","Q_from_current"]
    for key in float_keys:
        tensors = [f[key] for f in features_list]
        collated_features[key] = pad_sequence(tensors, batch_first=True, padding_value=0.0) if not all(t.numel()==0 for t in tensors) else torch.empty(len(tensors),0,dtype=torch.float32)
    for key in bool_keys:
        tensors = [f[key] for f in features_list]
        collated_features[key] = pad_sequence(tensors, batch_first=True, padding_value=False) if not all(t.numel()==0 for t in tensors) else torch.empty(len(tensors),0,dtype=torch.bool)
    collated_features["current_time"] = torch.stack([f["current_time"] for f in features_list])
    return collated_features, torch.stack(labels_list).squeeze()

def create_dataset_cda(
    json_file: str = GLOBAL_FILE_PREFIX+'cda.json', # Default data path
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates DataLoaders for CDA market bidding. Batch size is fixed to 1.
    Uses DEFAULT_SPLIT_RATIOS.
    """
    loader_batch_size = 1 # Hardcoded batch_size for CDA

    if not os.path.exists(json_file):
        print(f"Warning: JSON file {json_file} not found for CDA. Empty DataLoaders.")
        empty_dataset = DictDataset([])
        return (DataLoader(empty_dataset, batch_size=loader_batch_size, collate_fn=cda_collate_fn), DataLoader(empty_dataset, batch_size=loader_batch_size, collate_fn=cda_collate_fn), DataLoader(empty_dataset, batch_size=loader_batch_size, collate_fn=cda_collate_fn))
    
    torch.manual_seed(seed); np.random.seed(seed)
    try:
        with open(json_file, 'r') as f: raw_data_json = json.load(f)
    except Exception as e:
        print(f"Error reading JSON {json_file}: {e}. Empty DataLoaders.")
        empty_dataset = DictDataset([])
        return (DataLoader(empty_dataset, batch_size=loader_batch_size, collate_fn=cda_collate_fn), DataLoader(empty_dataset, batch_size=loader_batch_size, collate_fn=cda_collate_fn), DataLoader(empty_dataset, batch_size=loader_batch_size, collate_fn=cda_collate_fn))

    all_samples_list = []
    for item in raw_data_json:
        all_samples_list.append({
            'features': {
                "H_prices": torch.tensor([h.get("normalized_price",0.0) for h in item.get("H",[])],dtype=torch.float32),
                "H_expired": torch.tensor([h.get("is_expired",False) for h in item.get("H",[])],dtype=torch.bool),
                "Q_prices": torch.tensor([q.get("normalized_price",0.0) for q in item.get("Q_active",[])],dtype=torch.float32),
                "Q_from_current": torch.tensor([q.get("from_current",False) for q in item.get("Q_active",[])],dtype=torch.bool),
                "A_prices": torch.tensor([a.get("normalized_price",0.0) for a in item.get("A_active",[])],dtype=torch.float32),
                "P_series": torch.tensor(item.get("P",[]),dtype=torch.float32),
                "current_time": torch.tensor(item.get("current_time",0.0),dtype=torch.float32)},
            'label': torch.tensor(item.get("current_bid",0.0),dtype=torch.float32)
        })
    
    if not all_samples_list:
        print(f"Warning: No samples from {json_file} for CDA. Empty DataLoaders.")
        empty_dataset = DictDataset([])
        return (DataLoader(empty_dataset, batch_size=loader_batch_size, collate_fn=cda_collate_fn), DataLoader(empty_dataset, batch_size=loader_batch_size, collate_fn=cda_collate_fn), DataLoader(empty_dataset, batch_size=loader_batch_size, collate_fn=cda_collate_fn))

    n_samples = len(all_samples_list)
    shuffled_indices = torch.randperm(n_samples).tolist()
    split_train = int(n_samples * DEFAULT_SPLIT_RATIOS[0])
    split_val = split_train + int(n_samples * DEFAULT_SPLIT_RATIOS[1])
    train_list, val_list, test_list = [all_samples_list[i] for i in shuffled_indices[:split_train]], [all_samples_list[i] for i in shuffled_indices[split_train:split_val]], [all_samples_list[i] for i in shuffled_indices[split_val:]]
    
    train_dataset, val_dataset, test_dataset = DictDataset(train_list), DictDataset(val_list), DictDataset(test_list)
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=loader_batch_size, shuffle=True, collate_fn=cda_collate_fn, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=loader_batch_size, shuffle=False, collate_fn=cda_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=loader_batch_size, shuffle=False, collate_fn=cda_collate_fn)
    return train_loader, val_loader, test_loader

# --- Router Function ---
def create_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Router function to create DataLoaders based on CURRENT_TASK configuration in pyproject.toml.
    Uses fixed internal seed (42), batch size (32 for ulti/rps, 1 for cda),
    and split ratios (0.7, 0.15, 0.15).
    Individual dataset creators use their own default data file paths.
    """
    current_task = get_current_task()

    if not current_task:
        raise ValueError(
            f"Configuration 'current_task' in pyproject.toml is not set. "
            f"Expected values: 'ulti', 'rps', 'cda'."
        )

    print(f"--- Creating dataloaders for task: {current_task} ---")
    print(f"    Using fixed split ratios: {DEFAULT_SPLIT_RATIOS}")
    print(f"    Using fixed seed: {INTERNAL_SEED}")

    if current_task == "ulti":
        print(f"    Using batch size for ulti: {INTERNAL_BATCH_SIZE_ULTI_RPS}")
        return create_dataset_ulti(
            batch_size=INTERNAL_BATCH_SIZE_ULTI_RPS,
            seed=INTERNAL_SEED
            # csv_file will use its default: '/data/datasets/proposer.csv'
        )
    elif current_task == "rps":
        print(f"    Using batch size for rps: {INTERNAL_BATCH_SIZE_ULTI_RPS}")
        return create_dataset_rps(
            batch_size=INTERNAL_BATCH_SIZE_ULTI_RPS,
            seed=INTERNAL_SEED
            # json_file will use its default: './dataset/dataset.json'
        )
    elif current_task == "cda":
        print(f"    Using batch size for cda: 1 (fixed)")
        return create_dataset_cda(
            seed=INTERNAL_SEED
            # json_file will use its default: './cda.json'
        )
    else:
        raise ValueError(
            f"Unknown task: '{current_task}' from pyproject.toml configuration 'current_task'. "
            f"Supported tasks are: 'ulti', 'rps', 'cda'."
        )