import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import glob
from PIL import Image
from PIL import ImageFile
from maxvit import MaxViT

ImageFile.LOAD_TRUNCATED_IMAGES = True
from demo import ViT
from torchvision import transforms
from torch.utils.data import Dataset, Subset, DataLoader
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
import os

torch.manual_seed(2022)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Current Device : {device}")


class CatDataset(Dataset):
    def __init__(self, dataset_path, transform_fn, enhance_path=None):
        self.dataset_path = dataset_path
        self.transform = transform_fn
        self.label_idx2name = {}
        self.img_path = []
        if dataset_path:
            file_list = os.listdir(dataset_path)
            file_list = file_list[0:20]
            self.label_idx2name = np.array(file_list)
            self.label_name2idx = {}
            self.img2label = {}
            for i in range(len(file_list)):
                self.label_name2idx[self.label_idx2name[i]] = i
                lst = glob.glob(f"{dataset_path}/{file_list[i]}/*.jpg")
                if len(lst) >= 200:
                    lst = lst[0:200]
                self.img_path.extend(lst)
                for j in range(len(lst)):
                    self.img2label[lst[j]] = i
        if enhance_path:
            file_list = os.listdir(dataset_path)
            file_list = file_list[0:20]
            for i in range(len(file_list)):
                lst = glob.glob(f"{dataset_path}/{file_list[i]}/*.jpg")
                self.img_path.extend(lst)
                for j in range(len(lst)):
                    self.img2label[lst[j]] = i

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        img = self.img_path[index]
        label = self.img2label[img]
        img = Image.open(img).convert("RGB")
        img = self.transform(img)
        return (img, label)


channel_mean = torch.Tensor([0.485, 0.456, 0.406])
channel_std = torch.Tensor([0.229, 0.224, 0.225])
transformations_list = [
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=(30)),
    transforms.RandomPerspective(distortion_scale=0.5),
    transforms.RandomAdjustSharpness(sharpness_factor=20),
]

vit_train_transform_fn = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomApply(transformations_list, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=channel_mean, std=channel_std),
    ]
)

vit_valid_transform_fn = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=channel_mean, std=channel_std),
    ]
)

train_dataset = CatDataset(dataset_path="./train", transform_fn=vit_train_transform_fn)
valid_dataset = CatDataset(dataset_path="./valid", transform_fn=vit_train_transform_fn)
valid_dataset.transform = vit_valid_transform_fn
print(f"训练集图片的个数为：{len(train_dataset)}")
print(f"测试集图片的个数为：{len(valid_dataset)}")

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

valid_dataloader = DataLoader(valid_dataset, batch_size=4, shuffle=True)


class PretrainViT(nn.Module):

    def __init__(self):
        super(PretrainViT, self).__init__()
        # model = MaxViT(
        #     depths=(2, 2, 5, 2),
        #     channels=(96, 128, 256, 512),
        #     embed_dim=64,
        #     num_classes=20,
        # )
        model = ViT(n_classes=20)
        self.model = model

    def forward(self, x):
        output, attn_weight = self.model(x)
        return output, attn_weight


net = PretrainViT()
# net.load_state_dict(torch.load("d:/net.pt"))
print(
    f"number of paramaters: {sum([param.numel() for param in net.parameters() if param.requires_grad])}"
)

criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(), lr=0.015)
optimizer = optim.SGD(net.parameters(), lr=0.015)
# optimizer = optim.RMSprop(net.parameters(), lr=0.009)


def get_accuracy(output, label):
    output = output.to("cpu")
    label = label.to("cpu")

    sm = F.softmax(output, dim=1)
    _, index = torch.max(sm, dim=1)
    return torch.sum((label == index)) / label.size()[0]


def train(model, dataloader):
    model.train()
    running_loss = 0.0
    total_loss = 0.0
    running_acc = 0.0
    total_acc = 0.0

    for batch_idx, (batch_img, batch_label) in enumerate(dataloader):

        batch_img = batch_img.to(device)
        batch_label = batch_label.to(device)

        optimizer.zero_grad()
        output, attn_weights = model(batch_img)
        loss = criterion(output, batch_label)
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_loss += loss.item()

        acc = get_accuracy(output, batch_label)
        running_acc += acc
        total_acc += acc

        if batch_idx % 100 == 0 and batch_idx != 0:
            print(
                f"[step: {batch_idx:4d}/{len(dataloader)}] loss: {running_loss / 100:.3f}"
            )
            running_loss = 0.0
            running_acc = 0.0

    return total_loss / len(dataloader), total_acc / len(dataloader)


def validate(model, dataloader):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0

    for batch_idx, (batch_img, batch_label) in enumerate(dataloader):

        batch_img = batch_img.to(device)
        batch_label = batch_label.to(device)

        # optimizer.zero_grad()
        output = model(batch_img)

        loss = criterion(output, batch_label)
        # loss.backward()
        # optimizer.step()

        total_loss += loss.item()
        acc = get_accuracy(output, batch_label)
        total_acc += acc

    return total_loss / len(dataloader), total_acc / len(dataloader)


net.to(device)
train_loss_history = []
valid_loss_history = []
train_acc_history = []
valid_acc_history = []
EPOCHS = 20
for epoch in range(EPOCHS):
    train_loss, train_acc = train(net, train_dataloader)
    valid_loss, valid_acc = validate(net, valid_dataloader)
    print(
        f"Epoch: {epoch:2d}, training loss: {train_loss:.3f}, training acc: {train_acc:.3f} validation loss: {valid_loss:.3f}, validation acc: {valid_acc:.3f}"
    )

    train_loss_history.append(train_loss)
    valid_loss_history.append(valid_loss)

    train_acc_history.append(train_acc)
    valid_acc_history.append(valid_acc)

    if valid_loss <= min(valid_loss_history):
        torch.save(net.state_dict(), "net.pt")
