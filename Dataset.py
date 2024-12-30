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
    def __init__(self, data_dir, data_type="train", transform=None, target_transform=None):
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
        
        label_map = {0: 'no', 1: 'yes'}
        
        image_path = os.path.join(self.data_dir, label_map[label], image_name)
        image = Image.open(image_path)
        
        if self.transform:
            image = self.transform(image)
            
            
        if self.target_transform:
            label = self.target_transform(label)
            
        return image, torch.tensor(label).long()
    
    def __process_data(self):
        no_images = os.listdir(os.path.join(self.data_dir, 'no'))
        yes_images = os.listdir(os.path.join(self.data_dir, 'yes'))
        
        
        yes_images_end_index_train = int(len(yes_images) * 0.8)
        no_images_end_index_train = int(len(no_images) * 0.8)
        
        if self.data_type == 'train':
            no_images = no_images[0:no_images_end_index_train]
            yes_images = yes_images[0:yes_images_end_index_train]
            
        if self.data_type == "test":
            no_images = no_images[no_images_end_index_train:]
            yes_images = yes_images[yes_images_end_index_train:]
            
            
        combined_images = no_images + yes_images
        labels = [0]*len(no_images) + [1]*len(yes_images)
        
        return combined_images, labels
    




def plot_grid_images(x, y, batch_size):
    rows = cols = int(batch_size**0.5)
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    axes = axes.flatten()  # Flatten the 2D array of axes

    for i in range(batch_size):
        axes[i].imshow(x[i].permute(1, 2, 0))
        axes[i].set_title(f"Label: {y[i]}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
    
    
# if __name__ == "__main__":
#     data_dir = 'Images'
# # data_type = 'Images' ####data_type??????

#     batch_size = 9


#     transforms = t.Compose([transforms.Grayscale(num_output_channels=3), 
#                             t.Resize((128, 128)),
#                         t.ToTensor()])
        


#     ## create the dataset
#     # dataset = CustomDataset(data_dir, data_type='test', transform=transforms)
#     train_dataset = CustomDataset(data_dir, data_type='train', transform=transforms)
#     test_dataset = CustomDataset(data_dir, data_type='test', transform=transforms)
    
#     x, y = train_dataset[0]
#     ## create dataloader
#     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  ###debugging: len(train_dataloader) is 23)
#     test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)   ### len(test_dataloader) is 6

#     ##create the dataloader
#     # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#     for x, y in train_dataloader:
#         print(x.shape, y.shape)
#         plot_grid_images(x,y, batch_size)
#         break

        
        
    
        
        
        
        
    
