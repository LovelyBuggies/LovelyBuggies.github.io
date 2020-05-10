---
layout:     post
title:      Training Issues
subtitle:   深度学习笔记（十五）
date:       2019-06-01
author:     Nino Lau
header-img: img/Snip20190312_63.png
catalog: true
tags:
    - 深度学习

---


In this part, we would spend time revising common skills used in training model.

In short version, easy but useful.

In long version, we would introduce content in **Outline** step by step. 

We mainly reference from [sklearn](https://scikit-learn.org) and [Pytorch](https://pytorch.org)

Edited by Jiaxin Zhuang, Yang yang, Jiabin Cai.

# Outline

1. Common setup
    1. Commonly required module
    1. Random seed setting for reproducibility
1. Data split and Cross Validatioin
    1. Using sklearn's method to do Five-fold split
    2. Calculating Mean and Std for training dataset
    3. Data augmentation
2. classificatioon network 
    1. Original Resnet18
    2. Modified Resnet18 for out train dataset
3. Training 
    1. Including that define a model, loss function, metric, save model
    2. Pre-set hyper-parameters
    3. Initialize model parameters
    4. repeat over certain number of epochs
        1. Shuffle whole training data
        2. For each mini-batch data
            1. load mini-batch data
            2. compute gradient of loss over parameters
            3. update parameters with gradient descent
4. Transfer learning
5. Ensemble


```python
%load_ext autoreload
%autoreload 2
```

# 1.1 Commonly required module

[numpy](https://docs.scipy.org/doc/numpy-1.13.0/user/whatisnumpy.html): NumPy is the fundamental package for scientific computing in Python.

[pytorch](https://pytorch.org/docs/stable/index.html): End-to-end deep learning platform.

[torchvision](https://pytorch.org/docs/stable/torchvision/index.html): This package consists of popular datasets, model architectures, and common image transformations for computer vision.

[tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard): A suite of visualization tools to make training easier to understand, debug, and optimize TensorFlow programs.

[tensorboardX](https://tensorboardx.readthedocs.io/en/latest/tensorboard.html): Tensorboard for Pytorch.


```python
'''step 1'''
# Load all necessary modules here, for clearness
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torchvision.datasets import MNIST
import torchvision
from torchvision import transforms
from torch.optim import lr_scheduler
# from tensorboardX import SummaryWriter
from collections import OrderedDict
from matplotlib import pyplot as plt
```


```python
'''step 2'''
# Whether to put data in GPU according to GPU is available or not 
cuda = torch.cuda.is_available() 
device = torch.device("cuda:0") 
#  In case the default gpu does not have enough space, you can choose which device to use
torch.cuda.set_device(device) # device: id

# Since gpu in lab is not enough for your guys, we prefer to cpu computation
#device = torch.device("cuda:0") 
```

# 1.2 Random seed setting for reproducibility
In order to make computations deterministic on your specific problem on one specific platform and PyTorch release, there are a couple of steps to take.


```python
'''step 3'''
# However, in the same exp, seed for torch and numpy doesn't be the same.
SEED = 47

# Sets the seed for generating random numbers, including GPU and CPU
torch.manual_seed(SEED)

# Deterministic algorithm for convolutional ops
torch.backends.cudnn.deterministic = True
# Deterministic alogorithm for cudnn, otherwise, cuddn would choose the fastest algorithm for every 
# iteration ops, which cause variability and time consuming if input changes frequently.
torch.backends.cudnn.benchmark = False

# Seed the generator for Numpy
np.random.seed(SEED)
```

# 2. Data split and Cross Validatioin
We would split **Cifar10** into 5-fold and do cross validation.

The Cifar10 database (Modified National Institute of Standards and Technology database) s a collection of images that are commonly used to train machine learning and computer vision algorithms.

The CIFAR-10 dataset contains 60,000 32x32 color images in 10 different classes. The 10 different classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. There are 6,000 images of each class

And they look like images below.

![cifar2.jpg](attachment:cifar2.jpg)

## 2.1 Using sklearn's method to do Five-fold split

TODO

![grid_search_cross_validation.png](attachment:grid_search_cross_validation.png)


```python
# Example to use kFold
from sklearn.model_selection import KFold
import numpy as np

train_transform = transforms.Compose([
    
])

dataset = torchvision.datasets.CIFAR10(root='./data',
                                      train=True, transform=train_transform,download=True)

data = dataset.train_data

# dataset.train_labels gets list object, we should transform to numpy for convinience
label = np.array(dataset.train_labels)

# set numpy random seed, we can get a determinate k-fold dataset
# np.random.seed(1)

kf = KFold(n_splits=5,shuffle=True)

for train_index, test_index in kf.split(data):
    print('train_index', train_index, 'test_index', test_index)
    train_data, train_label = data[train_index], label[train_index]
    test_data, test_label = data[test_index], label[test_index]

# here we use the last fold to be our trainset
dataset.train_data = train_data
dataset.train_labels = list(train_label)
```

    Files already downloaded and verified
    train_index [    0     1     2 ..., 49996 49998 49999] test_index [    4     7     9 ..., 49993 49995 49997]
    train_index [    0     2     3 ..., 49997 49998 49999] test_index [    1    13    18 ..., 49976 49987 49991]
    train_index [    1     2     4 ..., 49997 49998 49999] test_index [    0     3     5 ..., 49986 49988 49994]
    train_index [    0     1     3 ..., 49997 49998 49999] test_index [    2    16    17 ..., 49982 49992 49996]
    train_index [    0     1     2 ..., 49995 49996 49997] test_index [    6     8    10 ..., 49989 49998 49999]


# 2.2 Calculating Mean and Std for training dataset


```python
def get_mean_std(dataset, ratio=0.01):
    """Get mean and std by sample ratio
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=int(len(dataset)*ratio), 
                                             shuffle=True, num_workers=10)
    train = iter(dataloader).next()[0]
    mean = np.mean(train.numpy(), axis=(0,2,3))
    std = np.std(train.numpy(), axis=(0,2,3))
    return mean, std
```


```python
# cifar10
train_dataset = torchvision.datasets.CIFAR10('./data', 
                                             train=True, download=False, 
                                             transform=transforms.ToTensor())
test_dataset = torchvision.datasets.CIFAR10('./data', 
                                           train=False, download=False, 
                                            transform=transforms.ToTensor())

train_mean, train_std = get_mean_std(train_dataset)

test_mean, test_std = get_mean_std(test_dataset)

print(train_mean, train_std)
print(test_mean,test_std)
```

    [ 0.4931373   0.48048604  0.44251624] [ 0.24307655  0.23882599  0.25874203]
    [ 0.48723358  0.48176482  0.45129994] [ 0.24020454  0.23832673  0.25824794]


## 2.3 Hyper parameter


```python
'''step 4'''
# set hyper parameter 
batch_size = 32
n_epochs = 50
learning_rate = 1e-3
```

## 2.4 Data augmentation

Using suitable data augmentation usually can get a better model. However, not all augmentation function are effective to all dataset It is advisable to choose favorable function for our dataset.  
transform document:[https://pytorch.org/docs/stable/torchvision/transforms.html]  

here are the most commonly used functions you may interest  


- torchvision.transforms.Compose()  
- torchvision.transforms.CenterCrop()
- torchvision.transforms.Pad()
- torchvision.transforms.RandomCrop()
- torchvision.transforms.RandomHorizontalFlip()
- torchvision.transforms.Resize()
- torchvision.transforms.Normalize()
- torchvision.transforms.ToPILImage()
- torchvision.transforms.ToTensor()
- torchvision.transforms.RandomRotation()

There is not standard answer for how to choose a suitable augmentation for your dataset, but we try to teach you what may be useful.  

We use cifar10 as our dataset in this class. so, here are some suggestion when you use it.  

The size of images in cifar10 are 32×32, it maybe not suitable to choose **rotation operation** for the images, becuase rotation will bring black pixels, those black pixels may exist after **randomcrop operation**.

we suggest that you can consider to use horizontal flip in nature dataset but not use vertical flip. becuase you know it is very rare for an object to be inverted.


```python
%matplotlib inline
from PIL import Image

# rotate 30°
transform_rotate = transforms.RandomRotation((30,30))

transform_horizontalflip = transforms.RandomHorizontalFlip(p=1)

transform_verticalflip = transforms.RandomVerticalFlip(p=1)

```


```python
# the first image in cifar10 trainset
img = Image.open('./img/example.jpg')
plt.imshow(img)
plt.axis('off')
plt.show()
```


![](http://ww2.sinaimg.cn/large/006tNc79ly1g4qz2yo9mhj3073070glh.jpg)



```python
# a cat image
img1 = Image.open('./img/cat.jpeg')
plt.imshow(img1)
plt.axis('off')
plt.show()
```


![](http://ww2.sinaimg.cn/large/006tNc79ly1g4qz30yfibj30a0070adp.jpg)



```python
# rotate the image
img2 = transform_rotate(img1)
plt.imshow(img2)
plt.axis('off')
plt.show()
```


![](http://ww1.sinaimg.cn/large/006tNc79ly1g4qz3u3o26j30a0070juj.jpg)



```python
# horizontal flip the image
img3 = transform_horizontalflip(img1)
plt.imshow(img3)
plt.axis('off')
plt.show()
```


![](http://ww1.sinaimg.cn/large/006tNc79ly1g4qz3x5p1mj30a00700wd.jpg)


```python
# vertical flip the image
img4 = transform_verticalflip(img1)
plt.imshow(img4)
plt.axis('off')
plt.show()
```


!![](http://ww2.sinaimg.cn/large/006tNc79ly1g4qz41ea3qj30a0070adp.jpg)



```python
'''step 5'''
'''
    the mean and variance below are from get_mean_std() function, every time you run the above function may get
    defferent value, because we use sampling
'''

'''
    notice: we usually will not use the dataset above, because its transform function has been change the numpy
    data to tensor data, but those transformations such as filp, rotation, crop, pad should be done before the
    data transforms to tensor.
'''

#transform1
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(10),
    transforms.ToTensor(), # Convert a PIL Image or numpy.ndarray to tensor.
    # Normalize a tensor image with mean 0.1307 and standard deviation 0.3081
    transforms.Normalize((0.4924044, 0.47831464, 0.44143882), (0.25063434, 0.2492162,  0.26660094))
])

# transform2
# train_transform = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])


test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.49053448, 0.47128814, 0.43724576), (0.24659537, 0.24846372, 0.26557055))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', 
                            train=True, 
                            transform=train_transform,
                            download=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data', 
                           train=False, 
                           transform=test_transform,
                           download=False)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)
```

    Files already downloaded and verified


# 3. classificatioon network

## 3.1 Original Original Resnet18


```python
resnet18 = torchvision.models.resnet18()
print(resnet18)
```

    ResNet(
      (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
      (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      (layer1): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (layer2): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (downsample): Sequential(
            (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (layer3): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (downsample): Sequential(
            (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (layer4): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (downsample): Sequential(
            (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (avgpool): AvgPool2d(kernel_size=7, stride=1, padding=0)
      (fc): Linear(in_features=512, out_features=1000, bias=True)
    )


## 3.2 Modified Resnet18 for out train dataset

As we can see, the resnet model we import from **torchvison.models** downsamples five times, It means that the size of feature maps in avgpooling layer is only 1×1, we may lose too much information during the convolution layers. It's worth noting that before enter the blocks, the input images have been downsapled tow times, we lose lots of information. so, fixing the first two downsaple layer may be a good choise.


```python
'''step 6'''
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18,self).__init__()
        original_model = torchvision.models.resnet18()
        original_model.conv1.stride = 1
        self.feature_extractor = nn.Sequential(
            *(list(original_model.children())[0:3]),
            *(list(original_model.children())[4:-2]),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(512,10)
        
    def forward(self, x):
        out1 = self.feature_extractor(x)
        out1 = out1.view(out1.size(0),-1)
        out2 = self.fc(out1)
        return out2
```

# 4. Training 
We would define training function here. Additionally, hyper-parameters, loss function, metric would be included here too. 

## 4.1 Pre-set hyper-parameters
setting hyperparameters like below

hyper paprameters include following part

* learning rate: usually we start from a quite bigger lr like 1e-1, 1e-2, 1e-3, and slow lr as epoch moves.
* n_epochs: training epoch must set large so model has enough time to converge. Usually, we will set a quite big epoch at the first training time.  
* batch_size: usually, bigger batch size mean's better usage of GPU and model would need less epoches to converge. And the exponent of 2 is used, eg. 2, 4, 8, 16, 32, 64, 128. 256.  


```python
'''step 7'''
# create a model object
# model = torchvision.models.resnet18()
# model.avgpool = nn.AdaptiveAvgPool2d(1)
# model.fc = nn.Linear(512,10)
model = ResNet18()
model.to(device)
# Cross entropy
loss_fn = torch.nn.CrossEntropyLoss()
# l2_norm can be done in SGD
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9) 
```

## 4.2 Initialize model parameters
Pytorch provide default initialization (**uniform intialization**) for linear layer. But there is still some useful intialization method.

Read more about initialization from this [link](https://pytorch.org/docs/stable/_modules/torch/nn/init.html)

```
    torch.nn.init.normal_
    torch.nn.init.uniform_
    torch.nn.init.constant_
    torch.nn.init.eye_
    torch.nn.init.xavier_uniform_
    torch.nn.init.xavier_normal_
    torch.nn.init.kaiming_uniform_
```

## 4.3 Repeat over certain numbers of epoch

* Shuffle whole training data 

```shuffle
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
```
* For each mini-batch data

    * load mini-batch data
    
    ```
    for batch_idx, (data, target) in enumerate(train_loader): \
        ...
    ```
    
    * compute gradient of loss over parameters
    
    ```
     output = net(data) # make prediction
     loss = loss_fn(output, target)  # compute loss 
     loss.backward() # compute gradient of loss over parameters 
    ```
    
    * update parameters with gradient descent
    
    ```
    optimzer.step() # update parameters with gradient descent 
    ```



```python
'''step 8'''
def train(train_loader, model, loss_fn, optimizer,device):
    """train model using loss_fn and optimizer. When thid function is called, model trains for one epoch.
    Args:
        train_loader: train data
        model: prediction model
        loss_fn: loss function to judge the distance between target and outputs
        optimizer: optimize the loss function
    Returns:
        total_loss: loss
    """
    
    # set the module in training model, affecting module e.g., Dropout, BatchNorm, etc.
    model.train()
    
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # clear gradients of all optimized torch.Tensors'
        outputs = model(data) # make predictions 
        loss = loss_fn(outputs, target) # compute loss 
        total_loss += loss.item() # accumulate every batch loss in a epoch
        loss.backward() # compute gradient of loss over parameters 
            
        optimizer.step() # update parameters with gradient descent 
            
    average_loss = total_loss / batch_idx # average loss in this epoch
    
    return average_loss
```


```python
'''step 9'''
def evaluate(loader, model, loss_fn, device):
    """test model's prediction performance on loader.  
    When thid function is called, model is evaluated.
    Args:
        loader: data for evaluation
        model: prediction model
        loss_fn: loss function to judge the distance between target and outputs
    Returns:
        total_loss
        accuracy
    """
    
    # context-manager that disabled gradient computation
    with torch.no_grad():
        
        # set the module in evaluation mode
        model.eval()
        
        correct = 0.0 # account correct amount of data
        total_loss = 0  # account loss
        
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            outputs = model(data) # make predictions 
            # return the maximum value of each row of the input tensor in the 
            # given dimension dim, the second return vale is the index location
            # of each maxium value found(argmax)
            _, predicted = torch.max(outputs, 1)
            # Detach: Returns a new Tensor, detached from the current graph.
            #The result will never require gradient.
            correct += (predicted == target).cpu().sum().detach().numpy()
            loss = loss_fn(outputs, target)  # compute loss 
            total_loss += loss.item() # accumulate every batch loss in a epoch
        
        accuracy = correct*100.0 / len(loader.dataset) # accuracy in a epoch
        average_loss = total_loss / len(loader)
        
    return average_loss, accuracy
```

Define function fit and use train_epoch and test_epoch

In this section we will produce tow method to change learning rate during the period of training  

- use **optimizer.param_groups** to change the learning rate in the optimizer at any epoch you want

- use **optimizer.lr_scheduler.StepLR()** to change the learning rate in the optimizer every several of epoch


when you find training loss and accuracy tend to be gentle, it maybe usful to decay the learning rate, but it is not always effective, you should do more experience to choose a suitable hyper parameters for your model


```python
'''step 10'''
def fit(train_loader, val_loader, model, loss_fn, optimizer, n_epochs, device):
    """train and val model here, we use train_epoch to train model and 
    val_epoch to val model prediction performance
    Args: 
        train_loader: train data
        val_loader: validation data
        model: prediction model
        loss_fn: loss function to judge the distance between target and outputs
        optimizer: optimize the loss function
        n_epochs: training epochs
    Returns:
        train_accs: accuracy of train n_epochs, a list
        train_losses: loss of n_epochs, a list
    """
    
    
    train_accs = [] # save train accuracy every epoch
    train_losses = [] # save train loss every epoch
    
    test_accs = []
    test_losses = []
    
    scheduler = lr_scheduler.StepLR(optimizer,step_size=6,gamma=0.1)
    
    for epoch in range(n_epochs): # train for n_epochs 
        # train model on training datasets, optimize loss function and update model parameters
        
#         # change the learning rate at any epoch you want 
#         if n_epochs % 6 == 0 and n_epochs != 0:
#             lr = lr * 0.1
#             for param_group in optimizer.param_groups:
#                 param_groups['lr'] = lr
        
        train_loss= train(train_loader, model, loss_fn, optimizer, device=device)
        
        # evaluate model performance on train dataset
        _, train_accuracy = evaluate(train_loader, model, loss_fn, device=device)
        
        # change the learning rate by scheduler
        scheduler.step()
        
        
        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(epoch+1, \
                                                                n_epochs, train_loss, train_accuracy)
        print(message)
    
        # save loss, accuracy
        train_accs.append(train_accuracy)
        train_losses.append(train_loss)
        show_curve(train_accs,'tranin_accs')
        show_curve(train_losses,'train_losses')
    
        # evaluate model performance on val dataset
        val_loss, val_accuracy = evaluate(val_loader, model, loss_fn, device=device)
        
        
        test_accs.append(val_accuracy)
        test_losses.append(val_loss)
        show_curve(test_accs,'test_accs')
        show_curve(test_losses,'test_losses')
        
        
        message = 'Epoch: {}/{}. Validation set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(epoch+1, \
                                                                n_epochs, val_loss, val_accuracy)
        print(message)
            
    return train_accs, train_losses
```


```python
'''step 10'''
def show_curve(ys, title):
    """plot curlve for Loss and Accuacy
    
    !!YOU CAN READ THIS LATER, if you are interested
    
    Args:
        ys: loss or acc list
        title: Loss or Accuracy
    """
    x = np.array(range(len(ys)))
    y = np.array(ys)
    plt.plot(x, y, c='b')
    plt.axis()
    plt.title('{} Curve:'.format(title))
    plt.xlabel('Epoch')
    plt.ylabel('{} Value'.format(title))
    plt.show()
```


```python
'''step 12: without tuning lr'''
train_accs, train_losses = fit(train_loader, test_loader, model, loss_fn, optimizer, n_epochs, device=device)
```

    Epoch: 1/50. Train set: Average loss: 1.6115, Accuracy: 52.0100



![](http://ww4.sinaimg.cn/large/006tNc79ly1g4qz4emsk1j30ar07qt8l.jpg)



![](http://ww2.sinaimg.cn/large/006tNc79ly1g4qz4hk3uyj30b007qmx3.jpg)



![](http://ww1.sinaimg.cn/large/006tNc79ly1g4qz4keaj0j30ar07qt8l.jpg)



![](http://ww3.sinaimg.cn/large/006tNc79ly1g4qz4nj3upj30b007qa9z.jpg)


    Epoch: 1/50. Validation set: Average loss: 1.3504, Accuracy: 50.2000
    Epoch: 2/50. Train set: Average loss: 1.2543, Accuracy: 62.1840



![](http://ww2.sinaimg.cn/large/006tNc79ly1g4qz4qoohaj30ar07q3yi.jpg)



![](http://ww2.sinaimg.cn/large/006tNc79ly1g4qz4vcx9uj30b007qaa3.jpg)



![](http://ww3.sinaimg.cn/large/006tNc79ly1g4qz4z6225j30ar07q0sq.jpg)



![](http://ww2.sinaimg.cn/large/006tNc79ly1g4qz524cewj30b007qmx6.jpg)


    Epoch: 2/50. Validation set: Average loss: 1.1162, Accuracy: 59.6400
    Epoch: 3/50. Train set: Average loss: 1.0419, Accuracy: 68.9220



![](http://ww4.sinaimg.cn/large/006tNc79ly1g4qz58gudqj30b007qaa3.jpg)



![](http://ww3.sinaimg.cn/large/006tNc79ly1g4qz5bt1dbj30au07qt8q.jpg)



![](http://ww3.sinaimg.cn/large/006tNc79ly1g4qz5ev9pyj30ar07qt8q.jpg)



![](http://ww4.sinaimg.cn/large/006tNc79ly1g4qz5hvnc1j30b007qjrf.jpg)


    Epoch: 3/50. Validation set: Average loss: 0.9808, Accuracy: 64.7800
    Epoch: 4/50. Train set: Average loss: 0.8807, Accuracy: 74.1900


![](http://ww4.sinaimg.cn/large/006tNc79ly1g4qz5kw93wj30ar07qt8p.jpg)



![](http://ww3.sinaimg.cn/large/006tNc79ly1g4qz5pfwezj30au07qglm.jpg)



![](http://ww2.sinaimg.cn/large/006tNc79ly1g4qz5t1iy3j30b007q74b.jpg)



![](http://ww4.sinaimg.cn/large/006tNc79ly1g4qz5wqwrnj30au07q3yi.jpg)


    Epoch: 4/50. Validation set: Average loss: 0.8958, Accuracy: 68.2700
    Epoch: 5/50. Train set: Average loss: 0.7469, Accuracy: 78.1360



![](http://ww1.sinaimg.cn/large/006tNc79ly1g4qz60getmj30ar07qmx5.jpg)



![](http://ww2.sinaimg.cn/large/006tNc79ly1g4qz65lk3hj30au07qglm.jpg)



![](http://ww1.sinaimg.cn/large/006tNc79ly1g4qz68hry2j30b007qdfv.jpg)



![](http://ww2.sinaimg.cn/large/006tNc79ly1g4qz6bfewmj30au07qgll.jpg)


    Epoch: 5/50. Validation set: Average loss: 0.8581, Accuracy: 70.0200
    Epoch: 6/50. Train set: Average loss: 0.6315, Accuracy: 81.8460



![](http://ww3.sinaimg.cn/large/006tNc79ly1g4qz6ftu2gj30ar07qjrd.jpg)



![](http://ww4.sinaimg.cn/large/006tNc79ly1g4qz6imyw7j30au07qweh.jpg)



![](http://ww2.sinaimg.cn/large/006tNc79ly1g4qz6lynvnj30ar07qt8o.jpg)



![](http://ww2.sinaimg.cn/large/006tNc79ly1g4qz6owxp5j30au07qgll.jpg)


    Epoch: 6/50. Validation set: Average loss: 0.8236, Accuracy: 72.1000
    Epoch: 7/50. Train set: Average loss: 0.5260, Accuracy: 85.4920



![](http://ww1.sinaimg.cn/large/006tNc79ly1g4qz6s7bszj30ar07q3yi.jpg)



![](http://ww3.sinaimg.cn/large/006tNc79ly1g4qz6vzpiwj30au07qdfu.jpg)



![](http://ww2.sinaimg.cn/large/006tNc79ly1g4qz6ys9maj30ar07qgll.jpg)



![](http://ww2.sinaimg.cn/large/006tNc79ly1g4qz720u2cj30au07q749.jpg)


    Epoch: 7/50. Validation set: Average loss: 0.8027, Accuracy: 73.2500
    Epoch: 8/50. Train set: Average loss: 0.3511, Accuracy: 90.0380



![](http://ww4.sinaimg.cn/large/006tNc79ly1g4qz75olrtj30ar07qglm.jpg)



![](http://ww2.sinaimg.cn/large/006tNc79ly1g4qz7903hbj30au07qmx6.jpg)



![](http://ww2.sinaimg.cn/large/006tNc79ly1g4qz7d0g1zj30ar07qq2x.jpg)



![](http://ww3.sinaimg.cn/large/006tNc79ly1g4qz7hhwf9j30au07q74a.jpg)


    Epoch: 8/50. Validation set: Average loss: 0.7302, Accuracy: 75.7000
    Epoch: 9/50. Train set: Average loss: 0.3008, Accuracy: 91.1640



![](http://ww4.sinaimg.cn/large/006tNc79ly1g4qz7kz1bzj30ar07qjre.jpg)



![](http://ww4.sinaimg.cn/large/006tNc79ly1g4qz7olocgj30au07qjre.jpg)



![](http://ww1.sinaimg.cn/large/006tNc79ly1g4qz7rlv1lj30ar07qq2x.jpg)



![](http://ww2.sinaimg.cn/large/006tNc79ly1g4qz7xs8eij30au07q74a.jpg)


    Epoch: 9/50. Validation set: Average loss: 0.7309, Accuracy: 75.7900
    Epoch: 10/50. Train set: Average loss: 0.2683, Accuracy: 92.0980



![](http://ww4.sinaimg.cn/large/006tNc79ly1g4qz80lzocj30ar07q74a.jpg)



![](http://ww3.sinaimg.cn/large/006tNc79ly1g4qz8b7zlnj30au07qdfu.jpg)



![](http://ww3.sinaimg.cn/large/006tNc79ly1g4qz86wicfj30ar07q749.jpg)



![](http://ww1.sinaimg.cn/large/006tNc79ly1g4qz8eux6zj30au07qweh.jpg)


    Epoch: 10/50. Validation set: Average loss: 0.7380, Accuracy: 75.7700
    Epoch: 11/50. Train set: Average loss: 0.2399, Accuracy: 92.7860



![](http://ww4.sinaimg.cn/large/006tNc79ly1g4qz8q7hvvj30ar07q74a.jpg)



![](http://ww1.sinaimg.cn/large/006tNc79ly1g4qz8sscnqj30au07qaa2.jpg)



![](http://ww4.sinaimg.cn/large/006tNc79ly1g4qz8vopsdj30ar07qaa1.jpg)



![](http://ww1.sinaimg.cn/large/006tNc79ly1g4qz8yz478j30au07qweh.jpg)


    Epoch: 11/50. Validation set: Average loss: 0.7476, Accuracy: 75.6500
    Epoch: 12/50. Train set: Average loss: 0.2169, Accuracy: 93.6720



![](http://ww3.sinaimg.cn/large/006tNc79ly1g4qz93rivrj30ar07q749.jpg)



![](http://ww1.sinaimg.cn/large/006tNc79ly1g4qz97ekvlj30au07qglm.jpg)



![](http://ww4.sinaimg.cn/large/006tNc79ly1g4qz9aazxoj30ar07q749.jpg)


![](http://ww2.sinaimg.cn/large/006tNc79ly1g4qz9dctahj30au07qweh.jpg)


    Epoch: 12/50. Validation set: Average loss: 0.7563, Accuracy: 75.7800
    Epoch: 13/50. Train set: Average loss: 0.1961, Accuracy: 94.1720



![](http://ww3.sinaimg.cn/large/006tNc79ly1g4qz9gz9xjj30ar07qjrd.jpg)



![](http://ww2.sinaimg.cn/large/006tNc79ly1g4qz9pp5hkj30au07qjre.jpg)



![](http://ww1.sinaimg.cn/large/006tNc79ly1g4qz9vh9soj30ar07q749.jpg)



![](http://ww1.sinaimg.cn/large/006tNc79ly1g4qz9zf3r4j30au07q3yi.jpg)


    Epoch: 13/50. Validation set: Average loss: 0.7695, Accuracy: 75.6800
    Epoch: 14/50. Train set: Average loss: 0.1752, Accuracy: 94.5560



![](http://ww2.sinaimg.cn/large/006tNc79ly1g4qza5n2h9j30ar07qgll.jpg)



![](http://ww4.sinaimg.cn/large/006tNc79ly1g4qza9n9xkj30au07qglm.jpg)



![](http://ww3.sinaimg.cn/large/006tNc79ly1g4qzad4swrj30ar07qaa1.jpg)



![](http://ww1.sinaimg.cn/large/006tNc79ly1g4qzagh9ofj30au07q3yi.jpg)


    Epoch: 14/50. Validation set: Average loss: 0.7627, Accuracy: 75.9100
    Epoch: 15/50. Train set: Average loss: 0.1686, Accuracy: 94.6880



![](http://ww3.sinaimg.cn/large/006tNc79ly1g4qzak26p8j30ar07qdft.jpg)



![](http://ww4.sinaimg.cn/large/006tNc79ly1g4qzb2ja7qj30au07qdfu.jpg)



![](http://ww1.sinaimg.cn/large/006tNc79ly1g4qzb6mk63j30ar07q0sp.jpg)



![](http://ww2.sinaimg.cn/large/006tNc79ly1g4qzba3nrkj30au07q3yi.jpg)


    Epoch: 15/50. Validation set: Average loss: 0.7632, Accuracy: 75.9400
    Epoch: 16/50. Train set: Average loss: 0.1647, Accuracy: 94.7860



![](http://ww4.sinaimg.cn/large/006tNc79ly1g4qzbdh4dvj30ar07qdft.jpg)



![](http://ww3.sinaimg.cn/large/006tNc79ly1g4qzbg9ncxj30au07qglm.jpg)



![](http://ww1.sinaimg.cn/large/006tNc79ly1g4qzbnw01sj30ar07q749.jpg)



![](http://ww4.sinaimg.cn/large/006tNc79ly1g4qzbqy6k7j30au07qweh.jpg)


    Epoch: 16/50. Validation set: Average loss: 0.7669, Accuracy: 75.9400
    Epoch: 17/50. Train set: Average loss: 0.1633, Accuracy: 94.7500



![](http://ww2.sinaimg.cn/large/006tNc79ly1g4qzbultguj30ar07qgll.jpg)



![](http://ww3.sinaimg.cn/large/006tNc79ly1g4qzbxzoa9j30au07qjre.jpg)



![](http://ww1.sinaimg.cn/large/006tNc79ly1g4qzc0umnvj30ar07q749.jpg)



![](http://ww2.sinaimg.cn/large/006tNc79ly1g4qzc5p6qdj30au07q3yi.jpg)


    Epoch: 17/50. Validation set: Average loss: 0.7673, Accuracy: 75.9700
    Epoch: 18/50. Train set: Average loss: 0.1600, Accuracy: 94.8080



![](http://ww3.sinaimg.cn/large/006tNc79ly1g4qzc8v22oj30as07qjrd.jpg)



![](http://ww4.sinaimg.cn/large/006tNc79ly1g4qzccf4g8j30aw07qmx6.jpg)



![](http://ww1.sinaimg.cn/large/006tNc79ly1g4qzcffvl0j30as07qaa1.jpg)



![](http://ww2.sinaimg.cn/large/006tNc79ly1g4qzcikaikj30aw07q74a.jpg)


    Epoch: 18/50. Validation set: Average loss: 0.7676, Accuracy: 75.9000
    Epoch: 19/50. Train set: Average loss: 0.1577, Accuracy: 94.8620



![](http://ww2.sinaimg.cn/large/006tNc79ly1g4qzcliv0mj30ar07qgll.jpg)



![](http://ww2.sinaimg.cn/large/006tNc79ly1g4qzcovsj5j30au07qjre.jpg)



![](http://ww1.sinaimg.cn/large/006tNc79ly1g4qzcs00s6j30ar07q3yh.jpg)



![](http://ww2.sinaimg.cn/large/006tNc79ly1g4qzcvf3tdj30au07q3yi.jpg)


    Epoch: 19/50. Validation set: Average loss: 0.7686, Accuracy: 75.9700
    Epoch: 20/50. Train set: Average loss: 0.1583, Accuracy: 94.9320



![](http://ww4.sinaimg.cn/large/006tNc79ly1g4qzczwh20j30ar07qgll.jpg)



![](http://ww4.sinaimg.cn/large/006tNc79ly1g4qzd4aq5ij30au07qglm.jpg)



![](http://ww1.sinaimg.cn/large/006tNc79ly1g4qzd7wfbuj30ar07q749.jpg)



![](http://ww3.sinaimg.cn/large/006tNc79ly1g4qzdb1kykj30au07qweh.jpg)


    Epoch: 20/50. Validation set: Average loss: 0.7703, Accuracy: 75.9100
    Epoch: 21/50. Train set: Average loss: 0.1551, Accuracy: 94.9160



![](http://ww3.sinaimg.cn/large/006tNc79ly1g4qzdoazjdj30ar07qgll.jpg)



![](http://ww3.sinaimg.cn/large/006tNc79ly1g4qzdraxjyj30au07qjre.jpg)



![](http://ww3.sinaimg.cn/large/006tNc79ly1g4qzdvj4iwj30ar07q749.jpg)



![](http://ww1.sinaimg.cn/large/006tNc79ly1g4qzdyzc6nj30au07q0sq.jpg)


    Epoch: 21/50. Validation set: Average loss: 0.7702, Accuracy: 75.9800
    Epoch: 22/50. Train set: Average loss: 0.1563, Accuracy: 94.9640



![](http://ww4.sinaimg.cn/large/006tNc79ly1g4qzegbh90j30ar07qt8o.jpg)



![](http://ww3.sinaimg.cn/large/006tNc79ly1g4qzelul2jj30au07qq2x.jpg)



![](http://ww2.sinaimg.cn/large/006tNc79ly1g4qzepp5fuj30ar07qglk.jpg)



![](http://ww4.sinaimg.cn/large/006tNc79ly1g4qzesw49sj30au07qdft.jpg)


    Epoch: 22/50. Validation set: Average loss: 0.7691, Accuracy: 76.0000
    Epoch: 23/50. Train set: Average loss: 0.1547, Accuracy: 94.9320



![](http://ww2.sinaimg.cn/large/006tNc79ly1g4qzexrfr9j30ar07qt8o.jpg)



![](http://ww2.sinaimg.cn/large/006tNc79ly1g4qzf1a4hsj30au07qq2x.jpg)



![](http://ww4.sinaimg.cn/large/006tNc79ly1g4qzf4jrs9j30ar07qjrc.jpg)



![](http://ww2.sinaimg.cn/large/006tNc79ly1g4qzf7jt5ej30au07qdft.jpg)


    Epoch: 23/50. Validation set: Average loss: 0.7691, Accuracy: 75.9400
    Epoch: 24/50. Train set: Average loss: 0.1547, Accuracy: 94.8580



![](http://ww3.sinaimg.cn/large/006tNc79ly1g4qzfakx7wj30ar07qq2w.jpg)



![](http://ww4.sinaimg.cn/large/006tNc79ly1g4qzfe0v20j30au07qmx5.jpg)



![](http://ww4.sinaimg.cn/large/006tNc79ly1g4qzfi7qtzj30ar07qglk.jpg)



![](http://ww3.sinaimg.cn/large/006tNc79ly1g4qzfo8ydvj30au07qaa1.jpg)


    Epoch: 24/50. Validation set: Average loss: 0.7696, Accuracy: 75.9500
    Epoch: 25/50. Train set: Average loss: 0.1530, Accuracy: 95.0520



![](http://ww2.sinaimg.cn/large/006tNc79ly1g4qzfro8mgj30ar07qt8o.jpg)



![](http://ww1.sinaimg.cn/large/006tNc79ly1g4qzfw2glgj30au07qt8p.jpg)



![](http://ww2.sinaimg.cn/large/006tNc79ly1g4qzg593m0j30ar07qjrc.jpg)



![](http://ww2.sinaimg.cn/large/006tNc79ly1g4qzg81f7vj30au07q749.jpg)


    Epoch: 25/50. Validation set: Average loss: 0.7703, Accuracy: 75.9200
    Epoch: 26/50. Train set: Average loss: 0.1528, Accuracy: 95.0200



![](http://ww2.sinaimg.cn/large/006tNc79ly1g4qzgea9ttj30ar07qq2w.jpg)



![](http://ww4.sinaimg.cn/large/006tNc79ly1g4qzgh4fhvj30au07qmx5.jpg)



![](http://ww3.sinaimg.cn/large/006tNc79ly1g4qzgjzzu5j30ar07qjrc.jpg)



![](http://ww1.sinaimg.cn/large/006tNc79ly1g4qzgolgyaj30au07qaa1.jpg)


    Epoch: 26/50. Validation set: Average loss: 0.7698, Accuracy: 76.0900
    Epoch: 27/50. Train set: Average loss: 0.1522, Accuracy: 95.1080



![](http://ww4.sinaimg.cn/large/006tNc79ly1g4qzgta3ufj30ar07qq2w.jpg)



![](http://ww2.sinaimg.cn/large/006tNc79ly1g4qzgw9wbhj30au07qq2x.jpg)



![](http://ww1.sinaimg.cn/large/006tNc79ly1g4qzgyswd5j30ar07qmx4.jpg)



![](http://ww1.sinaimg.cn/large/006tNc79ly1g4qzh21r3xj30au07qaa1.jpg)


    Epoch: 27/50. Validation set: Average loss: 0.7687, Accuracy: 75.9700
    Epoch: 28/50. Train set: Average loss: 0.1537, Accuracy: 94.9220



![](http://ww4.sinaimg.cn/large/006tNc79ly1g4qzh4yqb4j30ar07qq2w.jpg)



![](http://ww1.sinaimg.cn/large/006tNc79ly1g4qzh8e348j30au07qmx5.jpg)



![](http://ww1.sinaimg.cn/large/006tNc79ly1g4qzhazy3lj30ar07qjrc.jpg)



![](http://ww3.sinaimg.cn/large/006tNc79ly1g4qzhe1sg8j30au07q749.jpg)


    Epoch: 28/50. Validation set: Average loss: 0.7698, Accuracy: 75.9600
    Epoch: 29/50. Train set: Average loss: 0.1538, Accuracy: 94.9700



![](http://ww3.sinaimg.cn/large/006tNc79ly1g4qzhgr7w9j30ar07qt8o.jpg)



![](http://ww3.sinaimg.cn/large/006tNc79ly1g4qzhjm1uwj30au07qmx5.jpg)



![](http://ww3.sinaimg.cn/large/006tNc79ly1g4qzhm95arj30ar07qmx4.jpg)



![](http://ww1.sinaimg.cn/large/006tNc79ly1g4qzhsoylrj30au07q749.jpg)


    Epoch: 29/50. Validation set: Average loss: 0.7707, Accuracy: 76.0000
    Epoch: 30/50. Train set: Average loss: 0.1539, Accuracy: 94.9140



![](http://ww4.sinaimg.cn/large/006tNc79ly1g4qzhvyfuhj30ar07q0sp.jpg)



![](http://ww1.sinaimg.cn/large/006tNc79ly1g4qzhys2o4j30au07qt8p.jpg)



![](http://ww3.sinaimg.cn/large/006tNc79ly1g4qzi2tnybj30ar07qq2w.jpg)



![](http://ww4.sinaimg.cn/large/006tNc79ly1g4qzi60br8j30au07qdft.jpg)


    Epoch: 30/50. Validation set: Average loss: 0.7700, Accuracy: 75.9300
    Epoch: 31/50. Train set: Average loss: 0.1546, Accuracy: 95.0280



![](http://ww4.sinaimg.cn/large/006tNc79ly1g4qzib66emj30ar07q0sp.jpg)



![](http://ww2.sinaimg.cn/large/006tNc79ly1g4qzie0hztj30au07qq2x.jpg)



![](http://ww3.sinaimg.cn/large/006tNc79ly1g4qzigvns0j30ar07qq2w.jpg)



![](http://ww4.sinaimg.cn/large/006tNc79ly1g4qzik77i4j30au07q749.jpg)


   ......华丽的省略号。。。。。。

    Epoch: 49/50. Validation set: Average loss: 0.7696, Accuracy: 76.0200
    Epoch: 50/50. Train set: Average loss: 0.1530, Accuracy: 95.0000



![](http://ww1.sinaimg.cn/large/006tNc79gy1g4qzjnqrhpj30ar07qmx4.jpg)



![](http://ww2.sinaimg.cn/large/006tNc79gy1g4qzjrsdmqj30au07qjrd.jpg)



![](http://ww3.sinaimg.cn/large/006tNc79gy1g4qzjuordtj30ar07qjrc.jpg)



![](http://ww3.sinaimg.cn/large/006tNc79gy1g4qzjxyalfj30au07q3yh.jpg)


    Epoch: 50/50. Validation set: Average loss: 0.7701, Accuracy: 75.9100


If you want design a model, in its convolution layers, the original learning rate is 0.1, while the original learning rate of linear layer is 0.01, and you want to nine times smaller its learning rate every 10 epoch. Please write down your solution by pseudo code.

```python
#         # change the learning rate at any epoch you want 
#         if n_epochs % 6 == 0 and n_epochs != 0:
#             lr = lr * 0.1
#             for param_group in optimizer.param_groups:
#                 param_groups['lr'] = lr
```

### 4.4 save model 
Pytorch provide two kinds of method to save model. We recommmend the method which only saves parameters. Because it's more feasible and dont' rely on fixed model. 

When saving parameters, we not only save **learnable parameters in model**, but also **learnable parameters in optimizer**. 

A common PyTorch convention is to save models using either a .pt or .pth file extension.

Read more abount save load from this [link](https://pytorch.org/tutorials/beginner/saving_loading_models.html)

Generally speaking, to save our disk space and training time, we usually save the best model in Verification set and the last model for the last epoch.

the next cell will show you how to save the parameters of model and models.


```python
# to save the parameters of a model
torch.save(model.state_dict(), './params/resnet18_params.pt')

# to save the model
torch.save(model, './params/resnet18.pt')
```

    /opt/conda/lib/python3.6/site-packages/torch/serialization.py:250: UserWarning: Couldn't retrieve source code for container of type ResNet18. It won't be checked for correctness upon loading.
      "type " + obj.__name__ + ". It won't be checked "


You can get your the performance of your model at test set in every epoch. so, you can save the best model during the training period. Please add this function into your **train()** function

# 5. Transfer Learning

In this section, you will learn how to load the parameters you trained to your new model


```python
model1 = torchvision.models.resnet18()
print(model1.state_dict())

# generate a parameters file of model1
torch.save(model1.state_dict(),'./params/resnet18_params.pt')
```
