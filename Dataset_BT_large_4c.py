import os 

from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as t
import numpy as np
import cv2

from tqdm import tqdm

# __all__ = ['MyDataset', 'plot_grid_images']

class CustomDataset(Dataset):
    def __init__(self, data_dir, data_type="Training", size=(224, 224),is_augment=False, transform=None, target_transform=None):
        super().__init__()
        self.size = size
        self.is_augment = is_augment

        self.data_dir = data_dir ### self. mean yaw shi da class hissa jora ki
        self.data_type = data_type
        
        
        self.image_names, self.labels = self.__process_data() ###means da __process_data() function ba image_names, aw labels ba return
        
        self.transform = transform
        self.target_transform = target_transform
        
        
        
        
    def __len__(self):
        return len(self.image_names)
    

    def __preprocess_data(self, image, save=False, size=(224, 224)):
        """
        1. Apply theesholding to convert the image to binary image
        2. Apply morphological operations to remove noise i.e. dilation and erosion
        3. selecting the largest contour and calculating the four extreme points of the contour i,e. extreme top, bottom, left and right points
        4. Crop the image using the extreme points
        5. Resize the image to 224x224 using bicubic interpolation 
        """
        test_dir = "test_images"
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)

        # Create clean copy for processing
        clean_image = image.copy()

        if save:
            ## save the original image
            plt.imshow(image)
            plt.savefig(os.path.join(test_dir, 'original.jpg'))

        
        ## step 1: convert the image to binary image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


        if save:
            plt.imshow(thresh)
            plt.savefig(os.path.join(test_dir, 'binary.jpg'))

        ## step 2: Apply morphological operations to remove noise i.e. dilation and erosion
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        if save:
            plt.imshow(thresh)
            plt.savefig(os.path.join(test_dir, 'morphological.jpg'))

        ## step 3: selecting the largest contour and calculating the four extreme points of the contour i,e. extreme top, bottom, left and right points
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        
        ## largest contour
        if save:
            cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)
            plt.imshow(image)
            plt.savefig(os.path.join(test_dir, 'largest_contour.jpg'))


        ## step 4: Crop the image using the extreme points
        #crop = image[y:y+h, x:x+w]
        crop = clean_image[y:y+h, x:x+w]

        if save:
            plt.imshow(crop)
            plt.savefig(os.path.join(test_dir, 'cropped.jpg'))

        ## step 5: Resize the image to 224x224 using bicubic interpolation
        resized = cv2.resize(crop, size, interpolation=cv2.INTER_CUBIC)

        if save:
            plt.imshow(resized)
            plt.savefig(os.path.join(test_dir, 'resized.jpg'))


        return resized
    



    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        label = self.labels[idx]
        
        label_map = {0: 'glioma_tumor', 1: 'meningioma_tumor', 2: 'no_tumor', 3: 'pituitary_tumor'}
        
        image_path = os.path.join(self.data_dir, self.data_type, label_map[label], image_name)
        image = Image.open(image_path)
        
        resized_image = self.__preprocess_data(np.array(image), save=False, size=self.size)

        if self.transform:
            image = self.transform(resized_image)
            
            
        if self.target_transform:
            label = self.target_transform(label)

        if self.is_augment:
            ## save the augmented image 
            saved_image_path = os.path.join(self.data_dir, self.data_type, label_map[label], image_name.split('.')[0] + '_augmented.jpg')
            while os.path.exists(saved_image_path):
                random_number = np.random.randint(0, 100)
                saved_image_path = saved_image_path.split('.')[0] + f'_{random_number}.jpg'
            ## convert the tensor to numpy array
            ## apply random rotation (0, 90) degrees to PIL image
            
            image = transforms.RandomRotation(degrees=(0, 90))(image)
            image = transforms.RandomHorizontalFlip(p=1)(image)
            image.save(saved_image_path)

            
            
        return image, torch.tensor(label).long()
    
    def __process_data(self):
        if isinstance(self.data_dir, tuple):
            self.data_dir = os.path.join(*self.data_dir)
            
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


        # label_map = {0: 'glioma_tumor', 1: 'meningioma_tumor', 2: 'no_tumor', 3: 'pituitary_tumor'}
        # for index, image_name in enumerate(combined_images):
        #     if "augmented" in image_name:
        #         ## remove the augmented image from the directory
        #         full_pathh = os.path.join(self.data_dir, self.data_type, label_map[labels[index]], image_name)
        #         os.remove(full_pathh)

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


    transforms = t.Compose([
        t.ToPILImage(),
        t.Grayscale(num_output_channels=3), 
                             t.ToTensor()])
        


    ## create the dataset
    # dataset = CustomDataset(data_dir, data_type='test', transform=transforms)
    train_dataset = CustomDataset(data_dir, data_type='Training', is_augment=False, transform=transforms)
    test_dataset = CustomDataset(data_dir, data_type='Testing', transform=transforms)
    
    # x, y = train_dataset[0]
    ## create dataloader
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  ###debugging: len(train_dataloader) is 23)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)   ### len(test_dataloader) is 6

    ##create the dataloader
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # for index in range(len(train_dataset)):
    #     x, y = train_dataset[index]
    #     print(f"Image shape: {x.shape}, Label: {y}")
        
    #     ## save the image as test.jpg
    #     plt.imshow(x.permute(1, 2, 0))
    #     plt.savefig('test.jpg')
    #     break
        

    ##important commands in Debug Console 1-x.shape 2-y.shape 3- train_dataset.labels 4- test_dataset.labels 5- np.unique_counts(train_dataset) 5- np.unique_counts(test_dataset)
    
    ## 6- np.unique_counts(train_dataset.labels) 7- np.unique_counts(test_dataset.labels) 8- len(train_dataset) 9- len(test_dataset) 10- len(train_dataset + test_dataset)

        
        
    
        
        
        
        
    
