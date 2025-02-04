import torch
import torch.nn as nn
from efficientnet_lite_pytorch import EfficientNet
from efficientnet_lite0_pytorch_model import EfficientnetLite0ModelFile
#from efficientnet_lite2_pytorch_model import EfficientnetLite2ModelFile
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

    def __init__(self, num_classes, pretrained=True, arch_name="efficientnet-lite2", weights_path = None):
        super(Baseline, self).__init__()
        self.pretrained = pretrained
        if "lite" in arch_name:
            self.base_model = (
                EfficientNet.from_pretrained(arch_name, weights_path)
                if pretrained
                else EfficientNet.from_name(arch_name)
            )
            nftrs = self.base_model._fc.in_features
            self.base_model._fc = nn.Linear(nftrs, num_classes)
        else:
            self.base_model = timm.create_model(arch_name, pretrained=pretrained)
            nftrs = self.base_model.classifier.in_features
            self.base_model.classifier = nn.Linear(nftrs, num_classes)

    def forward(self, image):
        out = self.base_model(image)
        return out

def efficient_net_lite(num_classes, pretrained = True, arch_name="efficientnet-lite2"):
    
    weights_path = None
    if pretrained:
        weights_path = EfficientnetLite0ModelFile.get_model_file_path()
    model = Baseline(num_classes, pretrained, arch_name, weights_path)
    
    # Freeze all layers -- > Fine Tuning
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the final fully connected layer
    for param in model.base_model._fc.parameters():
        param.requires_grad = True
        
    return model


def efficient_net(num_classes, pretrained = True, arch_name="efficientnet_b0"):
    
    model = Baseline(num_classes, pretrained, arch_name)        
    return model
