import glob
import numpy as np
import os
import torch
from torch.utils.data import Dataset


# =========================
# TRAIN DATASET (FIXED)
# =========================
class trainset_loader(Dataset):
    def __init__(self, root):
        self.input_files = sorted(glob.glob(os.path.join(root, "input/*.npy")))
        print(f"[LOADER] {root} -> {len(self.input_files)} samples")

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_file = self.input_files[idx]

        label_file = input_file.replace("input", "label")
        proj_file = input_file.replace("input", "projection")
        geo_file = input_file.replace("input", "geometry")

        # Load data
        input_data = np.load(input_file).astype(np.float32)
        label_data = np.load(label_file).astype(np.float32)
        prj_data = np.load(proj_file).astype(np.float32)
        geometry = np.load(geo_file).astype(np.float32)

        # Convert to tensors
        input_data = torch.from_numpy(input_data).unsqueeze(0)   # (1, H, W)
        label_data = torch.from_numpy(label_data).unsqueeze(0)
        prj_data = torch.from_numpy(prj_data)                    # (views, detectors)

        geometry = torch.from_numpy(geometry)                    # (7,)

        # =========================
        # HyperFed inputs
        # =========================

        # option (used by model)
        option = geometry[:-1]   # (6,)

        # feature MUST be full 7D (IMPORTANT FIX)
        feature = geometry.clone()   # (7,)

        # =========================
        # Normalization (Paper Eqn 5)
        # =========================
        # Avoid per-sample instability
        min_val = torch.tensor([300, 1.0, 0.5, 300, 300, 300, 5e4])
        max_val = torch.tensor([600, 2.0, 3.0, 600, 600, 600, 5e5])

        feature = (feature - min_val) / (max_val - min_val + 1e-8)

        return input_data, label_data, prj_data, option, feature


# =========================
# TEST DATASET (OPTIONAL FIX)
# =========================
class testset_loader(Dataset):
    def __init__(self, root):
        self.files_A = []
        for i in range(0, 5):
            root_path = root + '_' + str(i + 1)
            path = os.path.join(root_path, 'test', 'input')
            self.files_A += sorted(glob.glob(path + '/*.npy'))

    def __len__(self):
        return len(self.files_A)

    def __getitem__(self, index):
        file_A = self.files_A[index]

        file_C = file_A.replace('input', 'projection')
        file_D = file_A.replace('input', 'geometry')

        input_data = np.load(file_A).astype(np.float32)
        prj_data = np.load(file_C).astype(np.float32)
        geometry = np.load(file_D).astype(np.float32)

        input_data = torch.from_numpy(input_data).unsqueeze(0)
        prj_data = torch.from_numpy(prj_data)
        geometry = torch.from_numpy(geometry)

        option = geometry[:-1]
        feature = geometry.clone()

        # same normalization
        min_val = torch.tensor([300, 1.0, 0.5, 300, 300, 300, 5e4])
        max_val = torch.tensor([600, 2.0, 3.0, 600, 600, 600, 5e5])

        feature = (feature - min_val) / (max_val - min_val + 1e-8)

        return input_data, prj_data, file_A, option, feature


# =========================
# TEST WITH LABEL (OPTIONAL)
# =========================
class testset_loader_w_label(Dataset):
    def __init__(self, root):
        self.files_A = []
        for i in range(0, 5):
            root_path = root + '_' + str(i + 1)
            path = os.path.join(root_path, 'test', 'input')
            self.files_A += sorted(glob.glob(path + '/*.npy'))

    def __len__(self):
        return len(self.files_A)

    def __getitem__(self, index):
        file_A = self.files_A[index]

        file_B = file_A.replace('input', 'label')
        file_C = file_A.replace('input', 'projection')
        file_D = file_A.replace('input', 'geometry')

        input_data = np.load(file_A).astype(np.float32)
        label_data = np.load(file_B).astype(np.float32)
        prj_data = np.load(file_C).astype(np.float32)
        geometry = np.load(file_D).astype(np.float32)

        input_data = torch.from_numpy(input_data).unsqueeze(0)
        label_data = torch.from_numpy(label_data).unsqueeze(0)
        prj_data = torch.from_numpy(prj_data)
        geometry = torch.from_numpy(geometry)

        option = geometry[:-1]
        feature = geometry.clone()

        # normalization
        min_val = torch.tensor([300, 1.0, 0.5, 300, 300, 300, 5e4])
        max_val = torch.tensor([600, 2.0, 3.0, 600, 600, 600, 5e5])

        feature = (feature - min_val) / (max_val - min_val + 1e-8)

        return input_data, prj_data, label_data, file_A, option, feature