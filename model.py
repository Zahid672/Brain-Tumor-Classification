import torch
from torch import nn
from torchvision.models import resnet101, ResNet101_Weights


class feature_extracter(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        # Remove the final FC layer
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
    
    def forward(self, x):
        # Get features before FC layer and flatten
        features = self.features(x)
        features = features.view(features.size(0), -1)  # Flatten: (batch_size, 2048)
        return features
    
# class finetune_resnet101(nn.Module):
#     def __init__(self, base_model, output_classes):
#         super().__init__()
#         self.base_model = base_model
#         # self.base_model.fc = nn.Sequential(
#         #     nn.Linear(in_features=2048, out_features=1024),
#         #     nn.ReLU(inplace=True),
#         #     nn.Linear(in_features=1024, out_features=output_classes)
#         # ) 
    
    
#     def forward(self, x):
#         out = self.base_model(x).fc
#         return out

# if __name__ == "__main__":
    
    

    
#     model = resnet101(weights=ResNet101_Weights.DEFAULT)
#     custom_model = finetune_resnet101(base_model=model, output_classes=2)
#     ## random input
#     x = torch.rand(1, 3, 512,512)
#     print(x.shape)
#     out = custom_model(x)
#     print(out.shape)
    