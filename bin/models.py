import torch
from torch.nn import Module
from torch.nn import Module, Conv2d, MaxPool2d, Linear
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import csv, time



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


class AslNNModel(Module):
    # same structure as HandNNModel, need to change dimensions
    def __init__(self):
        super().__init__()

        # input shape = (64, 400, 400) - (batch_size, w, h) from dataloader
        self.conv1 = Conv2d(1, 32, kernel_size=5)  # output shape: (496, 496, 32)
        self.pool1 = MaxPool2d(2)  # output shape: (198, 198, 32)
        self.conv2 = Conv2d(32, 64, kernel_size=3)  # output shape: (196, 196, 64)
        self.pool2 = MaxPool2d(2)  # output shape: (98, 98, 64) - torch uses floor by default
        self.conv3 = Conv2d(64, 64, kernel_size=3)  # output shape: (96, 96, 64)
        self.pool3 = MaxPool2d(2)  # output shape: (48, 48, 64)
        self.fc1 = Linear(48 * 48 * 64, 128)  # output shape: (48*48*64, 128)
        self.fc2 = Linear(128,
                          26)  # 24 possible output classes, but it goes up to idx 26: CUDA screams otherwise, so here we are
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

def asl_prediction_to_class_str(pred):
    # this may be wrong
    classes = {0: 'a', 1 : 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l', 12: 'm',
               13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z'}
    return classes[pred]

def classify_arbitrary_image(model, img, transpose=False, pred_to_str_fn=prediction_to_class_str):
    device = torch.device('cuda')
    img_type = type(img)
    if img_type == torch.Tensor:
        #print("tensor")
        img = img.float()
        img = img.unsqueeze(1)
    elif img_type == np.ndarray:
        #print("np array")
        # in some cases, we want to swap the order of the channels. In other cases, we don't
        if transpose:
            img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, 1)
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
    else:
        print("error: something other than a torch.Tensor or an np.ndarray was passed as img")
    img = img.to(device=device)
    with torch.no_grad():
        prediction = model(img)
    prediction = prediction.cpu().data.numpy()
    y_hat = np.argmax(prediction, axis=1)
    return pred_to_str_fn(y_hat[0])

def classify_many_images(model, imgs):
    # for now, assuming imgs is a list of images that are either np.ndarrays or torch.Tensors
    # labels will be given back in order images were given
    predictions = []
    for img in imgs:
        predictions.append(classify_arbitrary_image(model, img))
    return predictions

def make_asl_count_df(y):
    counts = np.zeros(26)
    y = np.array(y)
    for i in range(26):
        counts[i] = np.sum(np.where(y == i, 1, 0))
    idx = [chr(i) for i in range(97, 123)]
    columns=["Count"]
    df = pd.DataFrame(counts, index=idx, columns=columns)
    return df

def y_as_np_arr(dataset):
    return np.array([sample['y'] for sample in dataset])


def make_asl_train_test_split(dataset, counts_df, split_ratio=0.75, train_csv="./data/asl/train_asl.csv", test_csv="./data/asl/test_asl.csv"):
    header = ["path_to_file", "GT"]
    train = []
    test = []
    np.random.seed(0)
    train_counts, test_counts = [int(np.ceil(split_ratio*counts_df.iloc[idx, 0])) for idx in range(counts_df.shape[0])], [int(np.floor((1-split_ratio)*counts_df.iloc[idx, 0])) for idx in range(counts_df.shape[0])]
    y = y_as_np_arr(dataset)
    for idx in range(counts_df.shape[0]):
        #curr_train_selections = []
        #curr_test_selections = []
        only_class_locs = np.where(y==idx)[0]
        train_idxes = np.random.choice(only_class_locs, size=train_counts[idx], replace=False)
        for data_idx in only_class_locs:
            sample = dataset[data_idx]
            if data_idx in train_idxes:
                #curr_train_selections.append(sample['fname'])
                train.append(sample['fname'])
            else:
                #curr_test_selections.append(sample['fname'])
                test.append(sample['fname'])
        #train.append(curr_train_selections)
        #test.append(curr_test_selections)
    train_lasts = [fname.split('/')[-1] for fname in train]
    train_gt = [fname.split('_')[0] for fname in train_lasts]
    test_lasts = [fname.split('/')[-1] for fname in test]
    test_gt = [fname.split('_')[0] for fname in test_lasts]
    with open(train_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for i, fpath in enumerate(train):
            writer.writerow([fpath, train_gt[i]])
    with open(test_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for i, fpath in enumerate(test):
            writer.writerow([fpath, test_gt[i]])
    return train, test


def train_model(model, data_loader, max_epochs, use_cuda, save_path='trained_model.pkl', save=True):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        params = {'batch_size': 48, 'shuffle': True, 'num_workers': 8, 'pin_memory': True}
    else:
        params = {'batch_size': 48, 'shuffle': True, 'num_workers': 8}
    if use_cuda:
        model.cuda()

    print(model.eval())
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    start = time.time()
    for epoch in range(max_epochs):
        print(f"start epoch {epoch}")
        running_loss = 0.0
        epoch_start = time.time()
        for i, data in enumerate(data_loader):
            imgs, labels = data['image'], data['y']
            # move to GPU
            imgs, labels = imgs.cuda(), labels.cuda()
            imgs = imgs.float()
            optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 0:
                print(f"epoch {epoch}: running loss = {running_loss}")
        epoch_end = time.time()
        print(f"epoch {epoch} runtime = {epoch_end - epoch_start}")
    end = time.time()
    print(f"Total training time = {end - start}")

    # make sure to save the model so we don't need to train again
    if save:
        torch.save(model.state_dict(), save_path)