import numpy as np
from typing import List, Tuple


#########################################################################################
#                                        Dataset                                        #
#########################################################################################


class Dataset:
    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class Subset(Dataset):
    def __init__(self, dataset: 'Dataset', indices: List[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index):
        if np.isscalar(index):
            return self.dataset[self.indices[index]]
        else:
            return [self.dataset[i] for i in index]

    def __len__(self) -> int:
        return len(self.indices)


def random_split(dataset: 'Dataset', lengths: List[int]) -> List['Dataset']:
    if np.sum(lengths) != len(dataset):
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    indices = np.random.permutation(len(dataset)).tolist()
    return [
        Subset(dataset, indices[offset - length : offset])
        for offset, length in zip(np.cumsum(lengths), lengths)
    ]


#########################################################################################
#                                      Data Loader                                      #
#########################################################################################


class DataLoader:
    def __init__(
        self, dataset: 'Dataset', batch_size: int, shuffle: bool = False
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_iter = len(self.dataset) // self.batch_size
        self.iter = 0
        self.reset()

    def reset(self):
        self.iter = 0
        if self.shuffle:
            self.indices = np.random.permutation(len(self.dataset))
        else:
            self.indices = np.arange(len(self.dataset))

    def __iter__(self) -> 'DataLoader':
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.iter == self.max_iter:
            self.reset()
            raise StopIteration
        else:
            self.iter += 1
            batch_indices = self.indices[
                self.iter * self.batch_size : (self.iter + 1) * self.batch_size
            ]
            batch = [self.dataset[i] for i in batch_indices]

            batch_data = np.array([x[0] for x in batch])
            batch_labels = np.array([x[1] for x in batch])

            return batch_data, batch_labels

    def __len__(self) -> int:
        return self.max_iter
