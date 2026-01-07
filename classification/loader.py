import torch
from torch.utils.data import IterableDataset
import os
from torch.utils.data import DataLoader

class ISICMonkIterableDataset(IterableDataset):
    def __init__(self, chunk_dir):
        """
        chunk_dir: folder gde su sačuvani chunk fajlovi (.pt)
        """
        super().__init__()
        self.chunk_dir = chunk_dir
        self.chunk_files = sorted([
            os.path.join(chunk_dir, f)
            for f in os.listdir(chunk_dir)
            if f.endswith(".pt")
        ])

    def __iter__(self):
        for chunk_file in self.chunk_files:
            chunk = torch.load(chunk_file)
            for sample in chunk:
                yield sample["image"], torch.tensor(sample["monk_tone"], dtype=torch.float32)


if __name__ == "__main__":

    dataset = ISICMonkIterableDataset("dataset/isic_monk_dataset")
    loader = DataLoader(dataset, batch_size=2)  # možeš i num_workers>0

    for images, monk_tones in loader:
        print(images.shape, monk_tones.shape, monk_tones)