import torch
import torch.nn as nn
import torchvision

class EfficientNet(nn.Module):
    def __init__(self, device, num_classes):
        super(EfficientNet, self).__init__()
        torch.backends.cudnn.benchmark = True
        self.backbone = torchvision.models.efficientnet_b0(pretrained=True)     

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbone.classifier = nn.Sequential(
            nn.Linear(self.backbone.classifier[1].in_features, 2048),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, num_classes),
        )

        self.backbone.to(device)
        print("Model Initialized Successfully")


    def forward(self, x):
        return self.backbone(x)