
import os 
import sys

import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms as t
from torch.utils.data import DataLoader
from torchvision.models import resnet50, resnet101, densenet121, densenet169, vgg16, vgg19, alexnet, inception_v3, resnext50_32x4d, resnext101_32x8d
from torchvision.models import shufflenet_v2_x1_0, mobilenet_v2, mnasnet0_5 

from torchvision.models import ResNet50_Weights, ResNet101_Weights, DenseNet121_Weights, DenseNet169_Weights, VGG16_Weights, VGG19_Weights
from torchvision.models import AlexNet_Weights, Inception_V3_Weights, ResNeXt50_32X4D_Weights, ResNeXt101_32X8D_Weights, ShuffleNet_V2_X1_0_Weights
from torchvision.models import  MNASNet0_5_Weights, MobileNet_V2_Weights
import timm


from Dataset import CustomDataset, plot_grid_images
from model import feature_extracter



model_dict = {
              resnet50: ResNet50_Weights.DEFAULT, 
              resnet101: ResNet101_Weights.DEFAULT,
              densenet121: DenseNet121_Weights.DEFAULT, 
              densenet169: DenseNet169_Weights.DEFAULT,
              vgg16: VGG16_Weights.DEFAULT, 
              vgg19: VGG19_Weights.DEFAULT,
              alexnet: AlexNet_Weights.DEFAULT, 
            #  inception_v3: Inception_V3_Weights.DEFAULT,
              resnext50_32x4d: ResNeXt50_32X4D_Weights.DEFAULT, 
              resnext101_32x8d: ResNeXt101_32X8D_Weights.DEFAULT,
              shufflenet_v2_x1_0: ShuffleNet_V2_X1_0_Weights.DEFAULT, 
              mobilenet_v2: MobileNet_V2_Weights.DEFAULT,
              mnasnet0_5: MNASNet0_5_Weights.DEFAULT
            }
              

vit_models = {
    "vit_base_patch16_224": timm.create_model("vit_base_patch16_224", pretrained=True, drop_rate=0.01),
    "vit_base_patch32_224": timm.create_model("vit_base_patch32_224", pretrained=True, drop_rate=0.01),
    "vit_large_patch16_224": timm.create_model("vit_large_patch16_224", pretrained=True, drop_rate=0.01),
}


if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    data_dir = 'BrainTumorDatasets', 'BT-large-2c-dataset-253im'
    

    transforms = t.Compose([t.Grayscale(num_output_channels=3), 
                           t.Resize((224,224)), 
                            t.ToTensor()])
    
    transforms_inception = t.Compose([t.Grayscale(num_output_channels=3), 
                           t.Resize((342,342)), 
                            t.ToTensor()])
    

    train_dataset = CustomDataset(data_dir, 'train', transform=transforms)
    test_dataset = CustomDataset(data_dir, 'test', transform=transforms)

    main_dir = "extracted_features_1"
    if not os.path.exists(main_dir):
        os.mkdir(main_dir)


    for key, value in model_dict.items():
        print(f'Extracting features for {key}')
        print(f"model weights: {value}")
        base_model = key(weights=value)
        model = feature_extracter(base_model=base_model).to(device)

        if key == inception_v3:
            print('Using Inception_v3')
            transforms = transforms_inception
            train_dataset = CustomDataset(data_dir, 'train', transform=transforms)
            test_dataset = CustomDataset(data_dir, 'test', transform=transforms)
    

        ## extract features for each image in the train and test dataset and save them
        train_data_array_features = []
        train_data_array_labels = []
        test_data_array_features = []
        test_data_array_labels = []

        for i in range(len(train_dataset)):
            image, label = train_dataset[i]
            image = image.unsqueeze(0)
            image = image.to(device)
            features = model(image)
            train_data_array_features.append(features.cpu().detach().numpy())
            train_data_array_labels.append(label)

        for i in range(len(test_dataset)):
            image, label = test_dataset[i]
            image = image.unsqueeze(0)
            image = image.to(device)
            features = model(image)
            test_data_array_features.append(features.cpu().detach().numpy())
            test_data_array_labels.append(label)

        train_data_array_features = np.array(train_data_array_features)
        train_data_array_labels = np.array(train_data_array_labels)
        test_data_array_features = np.array(test_data_array_features)
        test_data_array_labels = np.array(test_data_array_labels)


        sub_dir = os.path.join(main_dir, key.__name__)
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)

        np.save(os.path.join(sub_dir, 'train_data_array_features.npy'), train_data_array_features)
        np.save(os.path.join(sub_dir, 'train_data_array_labels.npy'), train_data_array_labels)
        np.save(os.path.join(sub_dir, 'test_data_array_features.npy'), test_data_array_features)
        np.save(os.path.join(sub_dir, 'test_data_array_labels.npy'), test_data_array_labels)

        print('Features extracted and saved successfully')