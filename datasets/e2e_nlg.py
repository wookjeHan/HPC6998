import os
import numpy as np
import csv
from pathlib import Path
from typing import Union
from torch.utils.data import Dataset
from retriever import download_and_unzip_dataset_from_url

class E2ENLGSample:
    def __init__(self, mr: str, ref: str):
        self.mr = mr
        self.ref = ref
    
    def __str__(self):
        return f"E2ENLGSample(mr={self.mr}, ref={self.ref})"

class E2ENLG(Dataset):
    name = "E2E_NLG"
    url = "https://github.com/tuetschek/e2e-dataset/releases/download/v1.0.0/e2e-dataset.zip"
    def __init__(
        self,
        root: Union[str, Path] = f"./{name}",
        train: bool = True,
        download: bool = False
    ) -> None:
        if isinstance(root, str):
            root = os.path.expanduser(root)
        self.root = root
        self.train = train
        if download and not self._check_exists():
            self.download()
        else:
            print("Dataset already exists or argument download=False, skipping download")

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")
        
        self.data = self._load_data()

    def _check_exists(self) -> bool:
        return os.path.isdir(self.root)

    def __len__(self) -> int:
        return self.data.size

    def __getitem__(self, index: int) -> E2ENLGSample:
        return self.data[index]

    def _load_data(self):
        TRAIN_DATA_FILE = "e2e-dataset/trainset.csv"
        TEST_DATA_FILE = "e2e-dataset/testset.csv"
        data_file = TRAIN_DATA_FILE if self.train else TEST_DATA_FILE

        with open(f"{self.root}/{data_file}", mode='r') as file:
            csv_reader = csv.reader(file)

            # Skip header
            next(csv_reader, None)

            samples = [row for row in csv_reader]
            return np.array([E2ENLGSample(*item) for item in samples])

    def download(self):
        download_and_unzip_dataset_from_url(self.url, self.root)
