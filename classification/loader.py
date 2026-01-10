import torch
from torch.utils.data import IterableDataset
import os
from torch.utils.data import DataLoader
import random

class ISICMonkIterableDataset(IterableDataset):
    def __init__(self, chunk_dir, shuffle_chunks=False):
        """
        chunk_dir: folder gde su sačuvani chunk fajlovi (.pt)
        """
        super().__init__()
        self.chunk_dir = chunk_dir
        self.shuffle_chunks = shuffle_chunks
        self.chunk_files = sorted([
            os.path.join(chunk_dir, f)
            for f in os.listdir(chunk_dir)
            if f.endswith(".pt")
        ])

    # def __iter__(self):
    #     for chunk_file in self.chunk_files:
    #         chunk = torch.load(chunk_file)
    #         for sample in chunk:
    #             image_tensor = sample["image"]
    #             monk_tone_tensor = torch.tensor([sample["monk_tone"]], dtype=torch.float32)
    #             target_tensor = torch.tensor([sample["target"]], dtype=torch.float32)
    #             yield image_tensor, monk_tone_tensor, target_tensor

    def __iter__(self):
        chunk_files = self.chunk_files.copy()

        if self.shuffle_chunks:
            random.shuffle(chunk_files)  # 👈 SHUFFLE CHUNK-OVA

        for chunk_file in chunk_files:
            chunk = torch.load(chunk_file)

            # opciono: shuffle unutar chunk-a
            # random.shuffle(chunk)

            for sample in chunk:
                image_tensor = sample["image"]
                monk_tone_tensor = torch.tensor(
                    [sample["monk_tone"]], dtype=torch.float32
                )
                target_tensor = torch.tensor(
                    [sample["target"]], dtype=torch.float32
                )

                yield image_tensor, monk_tone_tensor, target_tensor

if __name__ == "__main__":

    dataset = ISICMonkIterableDataset("dataset/isic_monk_dataset")
    loader = DataLoader(dataset, batch_size=2)  # možeš i num_workers>0

    for images, monk_tones, targets in loader:
        print(images.shape, monk_tones)