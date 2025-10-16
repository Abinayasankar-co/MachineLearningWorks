import torch
import torch.nn as nn
from torchvision import datasets, models
import numpy as np

#the Shufflenet isused to make the classifer more Feasible with the Mobile phone 
class CropClassifier:
    def __init__(self, num_classes, device):
        self.device = device
        self.model = None
        self.num_classes  = num_classes
    
    @classmethod
    def from_pretrained(cls, num_classes, device):
        instance = cls(num_classes, device)
        model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        num_fltrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_fltrs, num_classes)
        )
        instance.model = model.to(device)
        return instance
    
    @staticmethod
    def compute_accuracy(preds, labels):
        try:
            correct = torch.sum(preds == labels.data)
            return correct.double() / len(labels)
        
        except Exception as e:
            raise Exception(f"Error while Analyzing the Data Accuracy : {e}")
    
    @staticmethod
    def tta_inference(model, inputs, n=3):
        model.eval()
        outputs = []
        with torch.no_grad():
            for _ in range(n):
                augmented_inputs = torch.flip(inputs, dims=[3]) if np.random.rand() > 0.5 else inputs
                outputs.append(model(augmented_inputs))
        
        return torch.mean(torch.stack(outputs), dim=0)
