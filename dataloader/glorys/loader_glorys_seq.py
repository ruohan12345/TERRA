import os
from glob import glob

import torch
import torch.utils.data as data

from utils import sort_key


class GLORYSSequentialDataset(data.Dataset):
    def __init__(
        self,
        root_full_path,
        status=0,
        norm_type="zs",
        data_precision="fp16",
        lead_time=1,
        train_end_idx=9496,
        test_days=365,
        return_sequence=False,
    ):
        self.status = status
        self.lead_time = lead_time
        self.return_sequence = return_sequence

        if (
            norm_type not in ("zs", "mm")
            or data_precision != "fp16"
        ):
            raise ValueError(
                "Only GLORYS sequential zs/mm fp16 data is supported, "
                f"but got norm_type={norm_type}, data_precision={data_precision}"
            )

        root_full_path = root_full_path

        files_full = sorted(glob(os.path.join(root_full_path, "*.pt")), key=sort_key)
        if not files_full:
            raise FileNotFoundError(f"No GLORYS .pt files found under {root_full_path}")

        self.files_full = files_full
        self.start_idx, self.end_idx = self._get_split_bounds(
            total_files=len(files_full),
            status=status,
            train_end_idx=train_end_idx,
            test_days=test_days,
            lead_time=lead_time,
        )
        self.length = max(0, self.end_idx - self.start_idx)

    def _get_split_bounds(self, total_files, status, train_end_idx, test_days, lead_time):
        train_end_idx = min(train_end_idx, total_files)
        if status == 0:
            return 0, max(0, train_end_idx - lead_time)
        if status == 1:
            test_start = train_end_idx
            test_end = min(total_files, train_end_idx + test_days)
            return test_start, max(test_start, test_end - lead_time)
        if status == 2:
            val_start = min(total_files, train_end_idx + test_days)
            val_end = min(total_files, train_end_idx + 2 * test_days)
            return val_start, max(val_start, val_end - lead_time)
        raise ValueError(f"Unsupported GLORYS dataset status: {status}")

    def __len__(self):
        return self.length

    def _load_tensor(self, index):
        tensor = torch.load(self.files_full[index], map_location="cpu")
        return torch.nan_to_num(tensor)

    def __getitem__(self, idx):
        actual_idx = self.start_idx + idx
        if self.return_sequence:
            tensors = [
                self._load_tensor(actual_idx + step)
                for step in range(self.lead_time + 1)
            ]
            return torch.stack(tensors, dim=0)

        input_tensor = self._load_tensor(actual_idx)
        label_tensor = self._load_tensor(actual_idx + self.lead_time)

        return input_tensor, label_tensor
