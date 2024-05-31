import sys
import os
from tkinter import YES
from unittest import result
import numpy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import cv2
from PIL import Image
from torchvision import transforms
from maxvit import MaxViT


class PretrainViT(nn.Module):

    def __init__(self):
        super(PretrainViT, self).__init__()
        model = MaxViT(
            depths=(2, 2, 5, 2),
            channels=(96, 128, 256, 512),
            embed_dim=64,
            num_classes=73,
        )
        self.model = model

    def forward(self, x):
        return self.model(x)


channel_mean = torch.Tensor([0.485, 0.456, 0.406])
channel_std = torch.Tensor([0.229, 0.224, 0.225])
transformations_list = [
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=(30)),
    transforms.RandomPerspective(distortion_scale=0.5),
    transforms.RandomAdjustSharpness(sharpness_factor=20),
]

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomApply(transformations_list, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=channel_mean, std=channel_std),
    ]
)


def get_attention_map(img, get_mask=False):
    x = transform(img)
    x.size()
    logits, att_mat = model(x.unsqueeze(0))
    att = torch.stack(att_mat).squeeze(1)
    att = torch.mean(att, dim=1)
    att = att[1]
    # residual_att = torch.eye(att.size(1))
    # aug_att_mat = att + residual_att
    aug_att_mat = att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
    joint_attentions = torch.zeros(aug_att_mat.size())
    # joint_attentions[0] = aug_att_mat[0]

    # for n in range(1, aug_att_mat.size(0)):
    #     joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])
    # joint_attentions[n] += joint_attentions[n - 1]

    # v = joint_attentions[-1]
    v = aug_att_mat
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    # print(v.shape)
    mask = v[0, :].reshape(grid_size, grid_size).detach().numpy()
    # print(mask.shape)
    if get_mask:
        result = cv2.resize(mask / mask.max(), img.size)
    else:
        mask = cv2.resize(mask / mask.max(), img.size)[..., np.newaxis]
        result = (mask * img).astype("uint8")

    return result


def plot_attention_map(original_img, att_map):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
    ax1.set_title("Original")
    ax2.set_title("Attention Map Last Layer")
    _ = ax1.imshow(original_img)
    _ = ax2.imshow(att_map)


model = PretrainViT()
model.load_state_dict(torch.load("d:/net.pt"))

img = Image.open(
    "D:\\Github\\Altria-repository\\transformer\\birds\\train\\ABBOTTS BOOBY\\041.jpg"
)

result = get_attention_map(img)
cv2.imwrite(f"D:/attnmap.jpg", result)
print("Yes")
