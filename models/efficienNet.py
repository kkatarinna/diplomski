import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNetWithFeatures(nn.Module):
    def __init__(self, num_extra_features=1, pretrained=True):
        super().__init__()

        # Load EfficientNetB0 backbone
        self.efficientnet = efficientnet_b0(
            weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # Freeze or unfreeze backbone if needed
        # for param in self.efficientnet.parameters():
        #     param.requires_grad = False

        # Output dim EfficientNetB0 je 1280
        image_feature_dim = self.efficientnet.classifier[1].in_features

        # Replace classifier with identity (izbacujemo njegov linear layer)
        self.efficientnet.classifier = nn.Identity()

        # MLP za dodatne feature-e (npr. 1 numericki feature = MST skin tone)
        self.extra_mlp = nn.Sequential(
            nn.Linear(num_extra_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        # Final classifier koji prima:
        # 1280 (slika) + 32 (numericki feature)
        self.classifier = nn.Sequential(
            nn.Linear(image_feature_dim + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),      # output = 1 (binary)
            nn.Sigmoid()
        )

    def forward(self, image, extra_features):
        """
        image: (B, 3, 224, 224)
        extra_features: (B, num_extra_features)
        """

        # 1. Features iz CNN-a
        img_feat = self.efficientnet(image)

        # 2. Features iz numerickog inputa
        extra_feat = self.extra_mlp(extra_features)

        # 3. Concatenate
        combined = torch.cat([img_feat, extra_feat], dim=1)

        # 4. Final classifier
        output = self.classifier(combined)

        return output


## # Primer korišćenja:
# model = EfficientNetWithFeatures(num_extra_features=1)

# # Primer inputa
# image = torch.randn(4, 3, 224, 224)       # batch 4 slika
# skin_tone = torch.randn(4, 1)             # 1 numerički feature

# pred = model(image, skin_tone)
# print(pred.shape)
