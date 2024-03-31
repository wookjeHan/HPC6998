from pathlib import Path
from typing import Union, Type
from torch.utils.data import Dataset, DataLoader

def create_dataloader(
    dataset: Type[Dataset],
    root: Union[str, Path],
    train: bool,
    batch_size: int,
    shuffle: bool,
    num_workers: int
) -> DataLoader:
    """
    Create a DataLoader for a given dataset and configuration

    :param dataset: The dataset to create a DataLoader for.
    :param root: The root directory of the data of the selected dataset. If the
        directory doesn't exist it will be created. 
    :param train: Whether to use the train dataset.
    :param batch_size: Batch size.
    :param shuffle: Shuffle.
    :param num_workers: Num workers.
    """
    return DataLoader(
        dataset(root=root, train=train, download=True),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
