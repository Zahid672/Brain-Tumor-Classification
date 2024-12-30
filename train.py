import torch
import torch.nn as nn
from torchvision import transforms as t
from torch.utils.data import DataLoader
from torchvision.models import resnet101, ResNet101_Weights

import sys


from Dataset import CustomDataset, plot_grid_images
from model import finetune_resnet101


def evaluate(model, test_dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    accuracy = 0.0
    with torch.no_grad():
        for x, y in test_dataloader:
            x = x.to(device)
            y = y.to(device)
            
            y_pred = model(x)
            loss = criterion(y_pred, y)
            
            running_loss += loss.item()
            
            batch_accuracy = find_accuracy(y_pred, y)
            accuracy += batch_accuracy.item()
            
            
    loss = running_loss/len(test_dataloader)
    epoch_accuracy = accuracy/len(test_dataloader)
    return loss, epoch_accuracy



def find_accuracy(y_pred, y_true):
    y_pred = torch.softmax(y_pred, dim=1)
    y_pred = torch.argmax(y_pred, dim=1)
    correct = (y_pred == y_true).float()
    accuracy = correct.sum()/len(y_pred)
    return accuracy


if __name__ == "__main__":
    data_dir = 'Images'
    batch_size = 256
    input_size = 128 * 128 * 3
    hidden_size = 120
    num_classes = 2
    num_epochs = 30
    learning_rate = 0.0001
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = resnet101(weights=ResNet101_Weights.DEFAULT)
    model = finetune_resnet101(base_model=model, output_classes=2).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    transforms = t.Compose([t.Grayscale(num_output_channels=3), 
                           t.Resize((224,224)), 
                            t.ToTensor()])
    
    
    train_dataset = CustomDataset(data_dir, 'train', transform=transforms)
    test_dataset = CustomDataset(data_dir, 'test', transform=transforms)
    
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    
    total_step = len(train_dataloader)
    for epoch in range(num_epochs):
        train_loss_per_epoch = 0
        train_accuracy_per_epoch = 0
        for i, (images, labels) in enumerate(train_dataloader):
            
            
            
            images = images.to(device)
            labels = labels.to(device)
            
            
            outputs = model(images)
            
            
            loss = criterion(outputs, labels)
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            
            
            
            
            Accuracy = find_accuracy(outputs, labels)
            
            
            
            train_accuracy_per_epoch += Accuracy.detach().cpu()
            train_loss_per_epoch += loss.detach().cpu()
            
        average_train_accuracy = train_accuracy_per_epoch / len(train_dataloader)
        average_train_loss = train_accuracy_per_epoch / len(train_dataloader)
        
        
        
        test_loss, test_accuracy = evaluate(model, test_dataloader, criterion, device)
        
        
        print(f"Epoch {epoch}'s Results: ")
        print(f"Train Loss: {average_train_loss}, Test Loss: {test_loss}")
        print(f"Train Accuracy: {average_train_accuracy}, Test Accuracy: {test_accuracy}")
        print("**************\n")
            