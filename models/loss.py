import torch
import torch.nn as nn

class BCEWithToneWeight(nn.Module):
    def __init__(self, tone_weights=None, device=None):
        """
        tone_weights: lista ili tensor veličine 11 (0-10) koji sadrži ponder za svaki monk_tone
        npr. tone_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.5, 3.0, 3.0, 3.5]
        """
        super().__init__()
        self.bce = nn.BCELoss(reduction='none')  # važno: ne redukuj odmah, da možeš množenje sa weight
        self.device = device if device is not None else torch.device("cpu")
        if tone_weights is not None:
            self.tone_weights = torch.tensor(tone_weights, dtype=torch.float32, device=self.device)
        else:
            self.tone_weights = None

    def forward(self, outputs, targets, monk_tones):
        """
        outputs: (B,1) ili (B) sa vrednostima 0-1
        targets: (B,1) ili (B) sa vrednostima 0/1
        monk_tones: (B,1) ili (B) integeri 0-10
        """
        loss = self.bce(outputs, targets)

        if self.tone_weights is not None:
            monk_tones = monk_tones.long().to(self.device)  # monk_tones na isti device
            weights = self.tone_weights[monk_tones]        # sada je safe, isti device
            loss = loss * weights

        return loss.mean()
