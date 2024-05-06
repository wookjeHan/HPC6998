import os
import json
import numpy as np
from pathlib import Path
from typing import Union
from torch.utils.data import Dataset
from datasets.retriever import download_github_folder_contents

class DialogSumSample:
    def __init__(self, fname: str, dialogue: str, summary: str, topic: str):
        self.fname = fname
        self.dialogue = dialogue
        self.summary = summary
        self.topic = topic

    def __str__(self):
        return f"DialogSumSample(fname={self.fname}, topic={self.topic}, summary={self.summary})"



class DialogSum(Dataset):
    name = "DialogSum"
    repo_info = ["cylnlp", "dialogsum", "DialogSum_Data"]
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

    def __getitem__(self, index: int) -> DialogSumSample:
        return self.data[index]

    def _load_data(self):
        TRAIN_DATA_FILE = "dialogsum.train.jsonl"
        # The test samples have a different format from train and dev samples.
        # The dev and test datasets are the same size so just use dev
        TEST_DATA_FILE = "dialogsum.dev.jsonl"
        data_file = TRAIN_DATA_FILE if self.train else TEST_DATA_FILE

        samples = []
        with open(f"{self.root}/{data_file}", 'r') as file:
            for line in file:
                data = json.loads(line)
                if isinstance(data['summary'], str):
                    samples += [data]
        return np.array([DialogSumSample(**item) for item in samples])


    def download(self):
        download_github_folder_contents(*self.repo_info, self.root)
