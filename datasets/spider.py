import json
import os
import numpy as np
from pathlib import Path
from typing import Union, List, Dict
from torch.utils.data import Dataset
from retriever import download_and_unzip_dataset_from_google_drive

class SpiderSample:
    def __init__(
        self: str,
        db_id: str,
        query: str,
        query_toks: List[str],
        query_toks_no_value: List[str],
        question: str,
        question_toks: List[str],
        sql: Dict
    ):
        self.db_id = db_id
        self.query = query
        self.query_toks = query_toks
        self.query_toks_no_value = query_toks_no_value
        self.question = question
        self.question_toks = question_toks
        self.sql = sql

    def __str__(self):
        return f"SpiderSample(db_id={self.db_id}, question={self.question}, query={self.query})"

class Spider(Dataset):
    name = "Spider"
    google_drive_id = "1iRDVHLr4mX2wQKSgA9J8Pire73Jahh0m"
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

    def __len__(self) -> int:
        return self.data.size

    def __getitem__(self, index: int) -> SpiderSample:
        return self.data[index]

    def _check_exists(self) -> bool:
        return os.path.isdir(self.root)

    def _load_data(self) -> List[SpiderSample]:
        TRAIN_DATA_FILE = "spider/train_spider.json"
        TEST_DATA_FILE = "spider/test_data/dev.json"
        data_file = TRAIN_DATA_FILE if self.train else TEST_DATA_FILE

        with open(f"{self.root}/{data_file}", 'r') as file:
            samples = json.load(file)
        return np.array([SpiderSample(**item) for item in samples])

    def download(self):
        download_and_unzip_dataset_from_google_drive(self.google_drive_id, self.root)
