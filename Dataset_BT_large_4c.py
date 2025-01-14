import os 

from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as t
import numpy as np

from torchvision import transforms


# __all__ = ['MyDataset', 'plot_grid_images']

class CustomDataset(Dataset):
    def __init__(self, data_dir, data_type="Training", transform=None, target_transform=None):
        super().__init__()
        
        self.data_dir = data_dir ### self. mean yaw shi da class hissa jora ki
        self.data_type = data_type
        
        
        self.image_names, self.labels = self.__process_data() ###means da __process_data() function ba image_names, aw labels ba return
        
        self.transform = transform
        self.target_transform = target_transform
        
        
        
        
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        label = self.labels[idx]
        
        label_map = {0: 'glioma_tumor', 1: 'meningioma_tumor', 2: 'no_tumor', 3: 'pituitary_tumor'}
        
        image_path = os.path.join(self.data_dir, self.data_type, label_map[label], image_name)
        image = Image.open(image_path)
        
        if self.transform:
            image = self.transform(image)
            
            
        if self.target_transform:
            label = self.target_transform(label)
            
        return image, torch.tensor(label).long()
    
    def __process_data(self):
        glioma_tumor = os.listdir(os.path.join(self.data_dir, self.data_type, 'glioma_tumor'))
        meningioma_tumor_images = os.listdir(os.path.join(self.data_dir, self.data_type, 'meningioma_tumor'))
        no_tumor_images = os.listdir(os.path.join(self.data_dir, self.data_type, 'no_tumor'))
        pituitary_tumor_images = os.listdir(os.path.join(self.data_dir, self.data_type, 'pituitary_tumor'))
        
        
        # yes_images_end_index_train = int(len(yes_images) * 0.8)
        # no_images_end_index_train = int(len(no_images) * 0.8)
        
        # if self.data_type == 'Training':
        #     glioma_tumor = glioma_tumor[0:]
        #     meningioma_tumor_images = meningioma_tumor[0:]
        #     no_tumor_images = no_tumor_images[0:]
        #     pituitary_tumor_images = pituitary_tumor_images[0:]
        
        
            
        # if self.data_type == "test":
        #     glioma_tumor = glioma_tumor[0:]
        #     meningioma_tumor_images = meningioma_tumor[0:]
        #     no_tumor_images = no_tumor_images[0:]
        #     pituitary_tumor_images = pituitary_tumor_images[0:]
            
            
        combined_images = glioma_tumor + meningioma_tumor_images + no_tumor_images + pituitary_tumor_images
        labels = [0]*len(glioma_tumor) + [1]*len(meningioma_tumor_images) + [2]*len(no_tumor_images) + [3]*len(pituitary_tumor_images)
        
        return combined_images, labels ###confusion
    




# def plot_grid_images(x, y, batch_size):
#     rows = cols = int(batch_size**0.5)
#     fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
#     axes = axes.flatten()  # Flatten the 2D array of axes

#     for i in range(batch_size):
#         axes[i].imshow(x[i].permute(1, 2, 0))
#         axes[i].set_title(f"Label: {y[i]}")
#         axes[i].axis('off')

#     plt.tight_layout()
#     plt.show()
    
    
if __name__ == "__main__":
    data_dir = os.path.join('BrainTumorDatasets', 'BT-large-4c-dataset-3264im')
    data_type = 'Training' ####data_type??????

    # batch_size = 9


    transforms = t.Compose([transforms.Grayscale(num_output_channels=3), 
                            t.Resize((224, 224)),
                        t.ToTensor()])
        


    ## create the dataset
    # dataset = CustomDataset(data_dir, data_type='test', transform=transforms)
    train_dataset = CustomDataset(data_dir, data_type='Training', transform=transforms)
    test_dataset = CustomDataset(data_dir, data_type='Testing', transform=transforms)
    
    x, y = train_dataset[0]
    ## create dataloader
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  ###debugging: len(train_dataloader) is 23)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)   ### len(test_dataloader) is 6

    ##create the dataloader
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for x, y in train_dataset:
        print(x.shape, y.shape)
        # plot_grid_images(x,y, batch_size)
        break

        
        
    
        
        
        
        
    
