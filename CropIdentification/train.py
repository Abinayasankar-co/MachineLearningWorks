import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import time 
import copy
import os
from model import CropClassifier

class ModelTraining():
    def __init__(self, data_dir : str, batch_size : int , num_epochs : int , learning_rate: float, num_classes: int, device : str):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.data_transform = {
            "train" : transforms.Compose([
                transforms.Resize((224,224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.3,contrast=0.3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.299,0.224,0.225])        
            ]),
            "val" : transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],
                                     [0.229,0.224,0.225])
            ])
        }
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    
    def _initialize_dataset_process(self):
        image_datasets = {
            x : datasets.ImageFolder(os.path.join(self.data_dir, x), self.data_transform[x])
            for x in ["train","val"]
        }
        dataloaders = {
            x : DataLoader(image_datasets[x], batch_size=self.batch_size,
                           shuffle=(x == "train"), num_workers=4)
            for x in ["train","test"]
        }
        dataset_sizes = {x : len(image_datasets[x]) for x in ["train", "val"]}
        class_name = image_datasets["train"].classes
        return class_name, dataset_sizes , dataloaders
    

    def _initialize_models(self):
        classifer = CropClassifier.from_pretrained(self.num_classes, self.device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(classifer.model.paramters(), lr= self.learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs)

        return classifer, criterion, optimizer, scheduler
    

    def _train_and_validate(self):
        try:
            model, criterion, optimizer, scheduler = self._initialize_models()
            classes, dataset_sizes, dataloaders = self._initialize_dataset_process()
            best_model_wts = copy.deepcopy(model.state_dict())
            best_acc = 0.0
            train_acc_history , val_acc_history = [] , []

            for epoch in range(self.num_epochs):
                for phase in ["train","test"]:
                    model.train() if phase == "train" else model.eval()
                    running_loss , running_corrects = 0.0, 0

                    for input, labels in dataloaders[phase]:
                        inputs, labels = input.to(self.device), labels.to(self.device)
                        optimizer.zero_grad()

                        with torch.set_grad_enabled(phase == "train"):
                            outputs = model(inputs) if phase == "train" else CropClassifier.tta_inference(model,inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)
                            
                            if phase == "train":
                                loss.backward()
                                optimizer.step()
                        
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                    
                    if phase == "train":
                        scheduler.step()
                    

                    epoch_loss = running_loss / dataset_sizes[phase]
                    epoch_acc = running_corrects.double() / dataset_sizes[phase]

                    print(f"{phase.capitalize()} Loss : {epoch_loss:.4f} Accuracy : {epoch_acc:.4f}")

                    if phase == "train":
                        train_acc_history.append(epoch_acc)
                    
                    else:
                        val_acc_history.append(epoch_acc)
                        
                        if epoch_acc > best_acc:
                            best_acc = epoch_acc
                            best_model_wts = copy.deepcopy(model.state_dict())
                            torch.save(model.state_dict())
            
            model.load_state_dict(best_model_wts)
            return model, train_acc_history, val_acc_history
        
        except Exception as e:
           print(f"{e}")

    
    
                    
                    

        
    
    
    

