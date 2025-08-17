import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

from config import project_config


def create_sequences(data, sequence_length):
        """Creates overlapping sequences from the time series data."""
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i + sequence_length])
        return np.array(sequences)

# Define the PyTorch Dataset
class SequenceDataset(Dataset):
        def __init__(self, sequences):
            self.sequences = sequences

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            # The input and target are the same for an autoencoder
            sequence = self.sequences[idx]
            return torch.tensor(sequence, dtype=torch.float32), torch.tensor(sequence, dtype=torch.float32)

def create_dataloaders(load_data=False):

    INTERIM_DATA_DIR = project_config.INTERIM_DATA_DIR
    os.makedirs(INTERIM_DATA_DIR, exist_ok=True)
    PROCESSED_DATA_DIR = project_config.PROCESSED_DATA_DIR
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    

    # --- Load the Data ---
    if load_data == True:
        print("Loading train, validation, test dataloaders...")
        train_loader = torch.load(os.path.join(PROCESSED_DATA_DIR, 'train_loader.pt'), weights_only=False)
        val_loader = torch.load(os.path.join(PROCESSED_DATA_DIR, 'val_loader.pt'), weights_only=False)
        test_loader = torch.load(os.path.join(PROCESSED_DATA_DIR, 'test_loader.pt'), weights_only=False)
        print("Data loaded successfully!")
        return train_loader, val_loader, test_loader

    

    # Step 1: Creating a Sequential Dataset
    # --- Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Preprocessed Data ---
    # DATA_DIR = "llm_ops_data"
    train_scaled = np.load(os.path.join(INTERIM_DATA_DIR, 'train_scaled.npy'))
    val_scaled = np.load(os.path.join(INTERIM_DATA_DIR, 'val_scaled.npy'))
    test_scaled = np.load(os.path.join(INTERIM_DATA_DIR, 'test_scaled.npy'))

    # --- Hyperparameters ---
    SEQUENCE_LENGTH = 12 * 6  # 6 hours of data (since each timestep is 5 mins (done this while data engineering and preprocessing))
    BATCH_SIZE = 256

    # Create the sequences
    print("Creating training, validation and testing sequences...")
    train_sequences = create_sequences(train_scaled, SEQUENCE_LENGTH)
    val_sequences = create_sequences(val_scaled, SEQUENCE_LENGTH)
    test_sequences = create_sequences(test_scaled, SEQUENCE_LENGTH)

    print(f"Shape of training sequences: {train_sequences.shape}")
    print(f"Shape of validation sequences: {val_sequences.shape}")
    print(f"Shape of testing sequences: {test_sequences.shape}")

    # Create Datasets and DataLoaders
    train_dataset = SequenceDataset(train_sequences)
    val_dataset = SequenceDataset(val_sequences)
    test_dataset = SequenceDataset(test_sequences)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of testing batches: {len(test_loader)}")

    # Save the DataLoaders
    torch.save(train_loader, os.path.join(PROCESSED_DATA_DIR, 'train_loader.pt'))
    torch.save(val_loader, os.path.join(PROCESSED_DATA_DIR, 'val_loader.pt'))
    torch.save(test_loader, os.path.join(PROCESSED_DATA_DIR, 'test_loader.pt'))
    print(f"DataLoaders saved to: {PROCESSED_DATA_DIR}")
    
    return train_loader, val_loader, test_loader

