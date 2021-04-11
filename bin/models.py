import torch
from torch.nn import Module
from torch.nn import Module, Conv2d, MaxPool2d, Linear
import torch.nn.functional as F
import numpy as np

class HandNNModel(Module):
    def __init__(self):
        super().__init__()

        # input shape = (32, 256, 256) - (batch_size, w, h) from dataloader
        self.conv1 = Conv2d(1, 32, kernel_size=5)  # output shape: (252, 252, 32)
        self.pool1 = MaxPool2d(2)  # output shape: (121, 121, 32)
        self.conv2 = Conv2d(32, 64, kernel_size=3)  # output shape: (119, 119, 64)
        self.pool2 = MaxPool2d(2)  # output shape: (59, 59, 64) - torch uses floor by default
        self.conv3 = Conv2d(64, 64, kernel_size=3)  # output shape: (57, 57, 64)
        self.pool3 = MaxPool2d(2)  # output shape: (28, 28, 64)
        self.fc1 = Linear(30 * 30 * 64, 128)  # output shape: (28*28*64, 128)
        self.fc2 = Linear(128, 10)
        self.activation = torch.nn.ReLU()

    def forward(self, X):
        X = self.activation(self.conv1(X))
        X = self.pool1(X)
        X = self.activation(self.conv2(X))
        X = self.pool2(X)
        X = self.activation(self.conv3(X))
        X = self.pool3(X)
        X = torch.flatten(X, 1)  # flatten with start_dim = 1
        X = self.fc1(X)
        X = self.fc2(X)
        output = F.softmax(X)
        return output

def prediction_to_class_str(pred):
    classes = {0 : "palm", 1 : "L", 2 : "fist", 3 : "fist_moved", 4 : "thumb", 5 : "index", 6 : "ok", 7 : "palm_moved", 8 : "c", 9 : "down"}
    return classes[pred]

def classify_arbitrary_image(model, img):
    device = torch.device('cuda')
    img_type = type(img)
    if img_type == torch.Tensor:
        #print("tensor")
        img = img.float()
        img = img.unsqueeze(1)
    elif img_type == np.ndarray:
        #print("np array")
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, 1)
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
    else:
        print("error: something other than a torch.Tensor or an np.ndarray was passed as img")
    img = img.to(device=device)
    prediction = model(img)
    prediction = prediction.cpu().data.numpy()
    y_hat = np.argmax(prediction, axis=1)
    return prediction_to_class_str(y_hat[0])

def classify_many_images(model, imgs):
    # for now, assuming imgs is a list of images that are either np.ndarrays or torch.Tensors
    # labels will be given back in order images were given
    predictions = []
    for img in imgs:
        predictions.append(classify_arbitrary_image(model, img))
    return predictions