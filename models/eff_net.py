import torch
import torch.nn as nn
import timm
    
class Baseline(nn.Module):
    """Baseline model
    We use the EfficientNets architecture that many participants in the ISIC
    competition have identified to work best.
    See here the [reference paper](https://arxiv.org/abs/1905.11946)
    Thank you to [Luke Melas-Kyriazi](https://github.com/lukemelas) for his
    [pytorch reimplementation of EfficientNets]
    (https://github.com/lukemelas/EfficientNet-PyTorch).
    """

    def __init__(self, num_classes, pretrained=True, arch_name="efficientnet_b0", weights_path = None):
        super(Baseline, self).__init__()
        self.pretrained = pretrained
        self.base_model = timm.create_model(arch_name, pretrained=pretrained)
        nftrs = self.base_model.classifier.in_features
        self.base_model.classifier = nn.Linear(nftrs, num_classes)

    def forward(self, image):
        out = self.base_model(image)
        return out


def efficient_net(num_classes, pretrained = True, arch_name="efficientnet_b0"):
    model = Baseline(num_classes, pretrained, arch_name)
     
    # # Freeze all layers
    # for param in model.parameters():
    #     param.requires_grad = False

    # # Unfreeze the final fully connected layer
    # for param in model.base_model.classifier.parameters():
    #     param.requires_grad = True
        
    return model