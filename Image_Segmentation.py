#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2;
import numpy as np;


def fill_boundary(img_path):
    
    im_in = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE); 
    th, im_th = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY_INV);
  
    im_floodfill = im_th.copy()
  
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
  
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    cv2.imwrite("R.png",im_floodfill)

    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
  
    im_out = im_th | im_floodfill_inv
    return np.abs(mask-1)


# In[3]:


import numpy as np
import os
import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image

class BiomedDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".png", "_Annotation.png"))
        image = np.array(Image.open(img_path))
        mask = fill_boundary(mask_path)#, dtype=np.float32)
        mask[mask == 255.0] = 1

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


# In[4]:


from torch.utils.data import random_split
from torch.utils.data import DataLoader

def get_loaders(
    input_dir,
    output_dir,
    train_size,
    batch_size,
    train_transform,
    val_transform,
    num_workers,
    pin_memory=True,
):
    
    ds = BiomedDataset(input_dir,output_dir,transform=train_transform)

    train_ds, val_ds = random_split(ds, [train_size, len(ds)-train_size], generator=torch.Generator().manual_seed(42))
    
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_dl = DataLoader(val_ds, batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    
    
    return train_dl, val_dl


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"D score: {dice_score/len(loader)}")
    model.train()
    
    
    
def check_accuracy1(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"D score: {dice_score/len(loader)}")
    model.train()
    return dice_score
    


# In[5]:


import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[32, 64, 128, 256, 512, 1024],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)


# In[7]:


import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 15
NUM_WORKERS = 0 

IMAGE_HEIGHT = 240
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "X_train/"
TRAIN_MASK_DIR = "Y_train/"

train_size = 799



def train_fn(loader, model, optimizer, loss_fn, scaler, loss_arr):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        
        loss_arr.append(loss.item())
        
    return loss_arr
        


# In[8]:


train_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=0.0,
            std=1.0,
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)

val_transforms = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0],
            std=[1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)


# In[7]:


model = UNET(in_channels=1, out_channels=1).to(DEVICE)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_loader, val_loader = get_loaders(
    TRAIN_IMG_DIR,
    TRAIN_MASK_DIR,
    train_size,
    BATCH_SIZE,
    train_transform,
    val_transforms,
    NUM_WORKERS,
    PIN_MEMORY,
)


print('Correct till here1')

dice_score_val = []
dc = check_accuracy1(val_loader, model, device=DEVICE)
dice_score_val.append(dc)

scaler = torch.cuda.amp.GradScaler()
    
print('Correct till here3')


loss_arr = []


for epoch in range(NUM_EPOCHS):
    loss_arr = train_fn(train_loader, model, optimizer, loss_fn, scaler, loss_arr)

    print('Correct till here4')

        # check accuracy
    dc = check_accuracy1(val_loader, model, device=DEVICE)
    dice_score_val.append(dc)

        # print some examples to a folder
#     save_predictions_as_imgs(
#         val_loader, model, folder="saved_images/", device=DEVICE
#     )


# In[ ]:





# In[8]:


import matplotlib.pyplot as plt
plt.plot(range(len(loss_arr)), loss_arr)


# In[9]:


k = []
for x in dice_score_val:
    k.append(torch.Tensor.item(x)/8)


# In[10]:


plt.plot(range(len(dice_score_val)), k)


# In[ ]:





# **After 4 epochs the dice score is not changing much**

# In[ ]:





# My cusdtom encoder is working better than UNET as I have included more layers into it and changed the kernel size to 3.
# But if we remove skip connections there is slight dectrease in dice score

# In[59]:


def return_predictions(img_path):
    
    test_image = np.array(Image.open(img_path).convert("L"))
    augmentations = val_transforms(image=test_image)
    image = augmentations["image"].resize(1,1,IMAGE_HEIGHT,IMAGE_WIDTH)
    pred  = model(image.to(DEVICE)).detach().cpu().numpy()
    pred = pred.reshape(IMAGE_HEIGHT,IMAGE_WIDTH)
    pred = torch.sigmoid(torch.from_numpy(pred))
    pred = (pred>0.5).float()
    pred = pred.numpy()
    img = np.uint8(pred) 
    edge = np.array(cv2.Canny(img, 0, 1))
    img1 = image.reshape(IMAGE_HEIGHT,IMAGE_WIDTH).numpy()
    img2 = (cv2.merge([img1,img1,img1]))
    img2[:,:,1][edge>0.5] = 0
    img2[:,:,0][edge>0.5] = 1
    img2[:,:,2][edge>0.5] = 0
    plt.imshow(img2)


# In[62]:


count = 0
for img_path in os.listdir('test_set/'):
    count += 1 
    plt.figure()
    if img_path[-2:] != 'db':
        return_predictions('test_set/' + img_path)
        plt.savefig('pred' + img_path)
    plt.show()
    if count == 10:
        break
    


# In[ ]:





# In[ ]:




