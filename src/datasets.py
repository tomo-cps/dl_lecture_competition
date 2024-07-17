import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint

import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint

import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint

import os
import numpy as np
import torch
import torch.utils.data as data
from scipy.signal import butter, lfilter
from sklearn.preprocessing import StandardScaler

class preprocess_data_warn_ThingsMEGDataset(data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X_ica.pt")).numpy()
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt")).numpy()
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt")).numpy()
            assert len(np.unique(self.y)) == self.num_classes, "Number of classes do not match."
        
        # Apply preprocessing
        self.X = self.preprocess_data(self.X)

        # Convert back to tensor
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.subject_idxs = torch.tensor(self.subject_idxs, dtype=torch.long)
        if split in ["train", "val"]:
            self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
    
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]
    
    def preprocess_data(self, X):
        # Bandpass filter
        X = self.bandpass_filter(X, lowcut=0.5, highcut=50, fs=500)
        
        # Standardize data
        X = self.standardize(X)
        
        return X
    
    def bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        filtered_data = lfilter(b, a, data, axis=-1)
        return filtered_data

    def standardize(self, data):
        scaler = StandardScaler()
        for i in range(data.shape[0]):
            data[i] = scaler.fit_transform(data[i])
        return data

class PreprocessdThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X_ica.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]
    
class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]
    
class ThingsMEGDatasetAppliedID(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", subject_id: int = None) -> None:
        super().__init__()

        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854

        print(f"Loading data for split: {split}")
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))

        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

        if split == "train" and subject_id is not None:
            print(f"Filtering train data for subject ID: {subject_id}")
            indices = (self.subject_idxs == subject_id)
            self.X = self.X[indices]
            self.subject_idxs = self.subject_idxs[indices]
            self.y = self.y[indices]

        print(f"Loaded data for split: {split}, size: {len(self.X)}")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]

    @property
    def num_channels(self) -> int:
        return self.X.shape[1]

    @property
    def seq_len(self) -> int:
        return self.X.shape[2]

