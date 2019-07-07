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

    OrderedDict([('conv1.weight', tensor([[[[ 1.2353e-02, -3.5713e-02, -4.6419e-02,  ..., -5.4487e-04,
               -2.5580e-02,  9.5039e-03],
              [-1.4113e-02, -2.1687e-02, -7.3655e-02,  ..., -2.4545e-02,
                3.7708e-02,  2.4177e-02],
              [-2.6923e-02, -2.2991e-03,  6.8104e-03,  ...,  3.0040e-02,
                2.5846e-02,  1.0737e-03],
              ...,
              [ 1.7352e-02, -2.8815e-02, -3.1425e-02,  ..., -1.4524e-02,
               -1.5380e-02,  1.5295e-02],
              [ 4.0697e-02, -6.9606e-03,  3.0921e-02,  ...,  4.7491e-03,
                2.9805e-03, -2.2234e-02],
              [-3.6224e-02, -1.5144e-02,  1.1833e-02,  ...,  5.6690e-03,
               -6.7935e-03,  7.5249e-03]],
    
             [[ 1.0802e-02, -1.3089e-02, -1.0400e-02,  ...,  5.3511e-03,
                1.5235e-02, -2.3676e-02],
              [ 5.4103e-03,  1.6151e-02,  4.3026e-02,  ...,  3.7497e-02,
                4.3525e-02, -2.6875e-02],
              [ 3.1831e-02,  5.4097e-02, -1.4268e-02,  ..., -3.4951e-02,
                3.2749e-02, -1.7871e-02],
              ...,
              [-5.7626e-02, -1.8986e-02,  6.0225e-03,  ...,  2.7881e-02,
               -7.7839e-03, -6.0318e-03],
              [ 1.9256e-02,  3.0078e-02,  3.5863e-02,  ...,  3.4886e-02,
                2.4019e-02,  8.8592e-03],
              [ 3.7452e-02,  5.7697e-03,  1.0191e-03,  ..., -2.0031e-02,
               -4.0363e-03, -1.6401e-02]],
    
             [[ 2.5891e-02,  1.7239e-02,  3.0213e-02,  ...,  1.0534e-02,
                5.1747e-02, -7.5586e-03],
              [-1.0796e-03,  1.6125e-02, -1.9219e-02,  ..., -2.1511e-02,
                3.4048e-02, -3.2933e-02],
              [ 7.1547e-03,  4.2137e-02, -9.0451e-03,  ...,  6.6442e-03,
                2.5878e-02,  2.4297e-02],
              ...,
              [-2.2563e-03,  7.9977e-04,  2.3270e-02,  ..., -3.2405e-02,
                6.0747e-04, -1.4864e-02],
              [-1.2239e-02, -1.9019e-02,  2.0364e-02,  ..., -4.4587e-02,
               -1.4965e-03, -7.7182e-03],
              [-4.3396e-02,  1.7862e-02, -1.6113e-03,  ...,  2.7045e-02,
                7.1188e-03,  1.3907e-02]]],


​    
            [[[ 5.5103e-03,  1.3440e-02, -5.4135e-02,  ...,  1.5246e-02,
                7.0073e-03, -6.8188e-03],
              [ 2.5464e-02,  1.9965e-03,  1.1401e-02,  ...,  2.2967e-02,
               -2.3158e-02, -4.8028e-02],
              [-3.0861e-02,  4.3261e-02,  2.2991e-04,  ...,  7.2141e-03,
               -2.5534e-03, -1.6390e-02],
              ...,
              [-2.2476e-03,  1.5484e-02, -1.8076e-02,  ...,  4.2129e-02,
               -2.6390e-02,  1.1068e-02],
              [-1.4368e-02, -2.3337e-02, -3.3074e-02,  ..., -4.1228e-02,
               -2.1975e-02,  1.5222e-02],
              [ 1.0343e-03,  1.2913e-02,  1.4365e-02,  ..., -1.7734e-02,
               -2.8202e-02,  1.5239e-02]],
    
             [[-1.9221e-02, -4.1880e-02, -3.4249e-02,  ..., -2.2087e-02,
               -9.4479e-03, -4.2507e-02],
              [ 1.5276e-02, -1.7290e-02, -5.0062e-03,  ..., -3.3725e-02,
                8.4538e-02,  7.4596e-02],
              [ 1.5014e-02,  1.3399e-02, -1.5247e-02,  ...,  6.1760e-03,
                1.0639e-03, -3.7298e-03],
              ...,
              [-1.7920e-02,  4.6874e-03, -1.3490e-02,  ...,  8.3387e-03,
                1.7101e-02, -2.1144e-02],
              [ 1.6083e-02, -2.8346e-02, -1.0181e-02,  ...,  8.7736e-03,
                1.7608e-02,  1.3936e-03],
              [-1.3562e-02, -1.6548e-02, -1.0019e-02,  ..., -3.1214e-03,
               -4.5485e-02,  1.8684e-02]],
    
             [[ 1.6536e-03, -6.2918e-02,  3.2885e-02,  ..., -1.2632e-02,
               -1.9542e-02, -4.5302e-02],
              [-2.3802e-02, -3.7273e-03,  8.8764e-03,  ...,  1.1835e-03,
                3.5745e-02,  8.3474e-03],
              [-1.8022e-02,  1.2632e-02,  4.6297e-02,  ..., -1.1841e-02,
                1.9780e-02, -9.1604e-03],
              ...,
              [ 1.4568e-02, -6.0503e-02,  3.2855e-02,  ..., -1.6141e-02,
               -3.4471e-03,  3.1018e-02],
              [-1.8240e-02, -2.3850e-02, -1.0398e-02,  ..., -3.8220e-02,
               -1.7804e-02, -2.1852e-03],
              [ 1.8869e-03,  3.8349e-02, -6.0193e-03,  ..., -3.2453e-02,
                3.5725e-02,  3.7736e-02]]],


​    
            [[[-3.7067e-02,  2.5998e-02, -1.5807e-03,  ..., -4.3575e-03,
                8.9018e-03, -2.6813e-02],
              [-6.3004e-02,  2.5725e-02,  2.2938e-02,  ..., -6.4212e-03,
               -4.9227e-02,  8.6667e-03],
              [-1.0451e-02,  7.7711e-03,  1.2346e-02,  ...,  2.1020e-03,
                3.6730e-02,  8.9196e-03],
              ...,
              [-1.3248e-02, -1.3267e-02, -1.6068e-02,  ...,  1.0950e-02,
               -5.2188e-03, -4.5454e-04],
              [-1.7372e-02, -1.7221e-02, -3.5878e-02,  ..., -1.0645e-02,
               -5.3494e-03, -2.7517e-02],
              [ 3.0780e-02, -2.9385e-02, -1.5382e-03,  ..., -2.7368e-02,
               -5.5117e-02,  9.2951e-04]],
    
             [[-2.9642e-02,  3.0652e-02, -2.0592e-02,  ...,  3.3554e-03,
               -5.0659e-03,  2.5336e-02],
              [-1.7673e-02,  6.2349e-03,  1.8002e-03,  ...,  4.5638e-02,
                1.4157e-02,  6.8525e-03],
              [ 2.9224e-02, -1.0199e-02,  5.6825e-03,  ..., -4.1673e-03,
                1.3161e-02,  6.3385e-02],
              ...,
              [ 1.3108e-02, -1.1734e-02, -7.6878e-03,  ...,  2.2743e-02,
               -3.8229e-02,  1.6747e-02],
              [ 2.5327e-03,  2.6047e-03,  5.3567e-02,  ..., -2.6054e-03,
               -1.4169e-02,  9.2331e-04],
              [ 1.1231e-03,  4.1346e-02,  7.1592e-03,  ..., -1.8857e-02,
               -2.1284e-02,  3.1787e-03]],
    
             [[-2.2749e-02,  2.8438e-02,  1.7338e-02,  ..., -1.4904e-02,
                8.4677e-03,  2.4564e-02],
              [ 1.9473e-02, -2.8390e-02,  1.3051e-02,  ...,  1.4458e-03,
                6.6112e-03,  2.8940e-02],
              [ 1.2877e-02, -6.3633e-03, -1.5618e-02,  ...,  7.4578e-03,
               -1.8088e-02,  3.8192e-02],
              ...,
              [-6.9419e-03, -2.7709e-02,  4.2880e-02,  ..., -8.6494e-03,
                7.7893e-03,  1.4865e-02],
              [ 3.1384e-03,  3.7284e-03, -1.2248e-02,  ...,  4.7131e-02,
                9.8473e-03,  2.4719e-02],
              [ 9.0937e-03, -9.5125e-03,  3.4045e-02,  ...,  3.5318e-02,
                8.1867e-03, -1.7129e-02]]],


​    
            ...,


​    
            [[[ 1.0484e-04,  4.1647e-02, -4.4440e-02,  ..., -2.5187e-02,
               -2.6889e-02, -2.6403e-03],
              [-1.4718e-02,  3.9153e-02,  2.1888e-02,  ...,  4.7834e-02,
               -4.0136e-02,  1.2690e-02],
              [-2.0801e-02, -4.4061e-03,  9.3861e-03,  ...,  4.2037e-02,
                9.2128e-03, -1.1401e-02],
              ...,
              [ 1.7732e-02,  2.5918e-02,  7.7917e-03,  ..., -5.1466e-02,
               -4.4485e-02,  8.7730e-03],
              [-4.3991e-03,  1.5231e-02,  6.1977e-03,  ..., -2.1098e-02,
               -7.3863e-03,  7.3081e-02],
              [ 2.5606e-02,  2.2408e-02,  1.3843e-02,  ..., -2.2067e-02,
                1.4076e-03,  3.6694e-03]],
    
             [[-4.1789e-02, -1.2482e-02, -6.4251e-03,  ..., -4.3878e-03,
               -7.4479e-03, -1.3693e-02],
              [ 1.2541e-02,  1.0336e-02, -2.9981e-02,  ...,  1.6250e-02,
                1.5679e-02, -2.3492e-03],
              [-1.8501e-02,  3.0069e-03, -2.4815e-02,  ...,  4.0587e-02,
               -4.5073e-02,  1.6156e-02],
              ...,
              [ 1.5642e-02, -5.8200e-04,  5.1297e-03,  ...,  2.7395e-02,
                6.1172e-03,  2.5805e-02],
              [-2.5227e-02,  1.1054e-02, -4.7216e-02,  ..., -2.8864e-02,
                3.1125e-02,  7.3301e-03],
              [ 1.4667e-02, -3.2257e-02,  2.9765e-02,  ...,  1.9084e-02,
               -6.4017e-03,  2.5315e-02]],
    
             [[ 1.7699e-02, -9.3588e-03, -6.6574e-03,  ..., -3.5212e-02,
               -3.1355e-02, -3.9393e-02],
              [ 1.0191e-02,  3.0520e-02, -6.5790e-03,  ...,  1.8725e-02,
                3.6009e-03, -1.1696e-02],
              [-3.4537e-03,  1.8168e-02, -4.9313e-03,  ...,  2.8893e-02,
                1.4810e-02,  1.2410e-02],
              ...,
              [-3.0489e-02,  2.8490e-02, -2.0475e-02,  ...,  2.5340e-02,
               -1.2772e-02,  4.1910e-03],
              [ 1.2047e-02, -1.3051e-02, -1.0325e-02,  ..., -7.2207e-03,
                1.3946e-02, -3.5824e-02],
              [-3.5725e-02,  5.6465e-03, -4.8942e-02,  ...,  2.7280e-02,
               -2.2560e-02,  2.2423e-02]]],


​    
            [[[ 2.5215e-02, -2.6037e-02, -7.1189e-03,  ...,  1.2421e-02,
               -2.3827e-02,  1.2885e-02],
              [ 2.8025e-02,  2.2198e-02, -1.9905e-02,  ..., -2.6249e-02,
                3.4476e-02,  1.1111e-03],
              [ 2.0601e-02,  1.3386e-02,  9.3798e-03,  ...,  1.8897e-02,
                4.5602e-02,  5.2975e-03],
              ...,
              [ 1.6043e-02, -2.6220e-02,  2.8817e-02,  ...,  3.2787e-02,
               -2.0426e-02, -4.5844e-03],
              [ 2.8145e-03,  9.9836e-03,  1.7517e-02,  ...,  9.4083e-03,
               -3.1759e-02,  6.0568e-02],
              [-9.7894e-04,  3.2081e-02,  1.0885e-02,  ..., -2.7470e-02,
                7.8087e-03, -8.2926e-03]],
    
             [[ 1.2586e-02,  1.4858e-02,  2.2889e-02,  ..., -2.5693e-02,
                6.6764e-04, -2.6042e-02],
              [-1.7399e-02, -4.7715e-03,  7.1666e-03,  ...,  4.3989e-02,
               -4.7763e-02, -1.5933e-03],
              [-3.5319e-03,  1.7150e-02, -3.7306e-02,  ...,  2.9311e-02,
                2.1062e-02,  1.7357e-02],
              ...,
              [-1.5538e-02,  3.4753e-03,  1.9574e-02,  ...,  3.0604e-02,
                9.5223e-03,  4.1064e-02],
              [ 1.7160e-02, -1.9599e-02,  3.3547e-04,  ...,  5.6494e-02,
               -3.5979e-02,  8.9146e-03],
              [-1.0057e-03,  7.0559e-03,  5.7961e-03,  ...,  1.1160e-02,
               -2.6502e-02,  8.7206e-03]],
    
             [[-1.1577e-03, -2.7506e-04, -2.8923e-03,  ..., -1.4175e-03,
               -2.3723e-02, -2.3104e-02],
              [ 8.9705e-03,  2.4740e-02, -2.5172e-02,  ..., -8.8374e-03,
               -4.6386e-02, -2.0492e-02],
              [-1.4191e-02, -1.8026e-02, -3.3038e-02,  ..., -2.7241e-02,
                1.5081e-02,  1.5494e-02],
              ...,
              [ 8.4246e-03,  3.8451e-03, -1.7167e-02,  ...,  6.2993e-02,
               -3.7641e-04,  1.7011e-02],
              [-4.9899e-02,  3.7582e-02,  1.6764e-02,  ...,  8.1563e-03,
                5.6094e-03, -2.6289e-03],
              [-1.7065e-02, -8.2094e-03,  2.0225e-02,  ...,  1.5519e-02,
                3.7813e-02,  3.7859e-02]]],


​    
            [[[ 1.1671e-02,  8.9922e-05,  1.3341e-02,  ...,  3.7564e-02,
               -1.7323e-02,  1.5196e-02],
              [-2.3953e-02,  2.9051e-03, -5.9366e-02,  ..., -1.5524e-02,
               -1.0648e-02, -8.8292e-03],
              [ 4.2525e-03, -7.0732e-03, -5.1372e-04,  ..., -1.2803e-02,
                1.0091e-03, -3.1203e-02],
              ...,
              [ 8.3300e-03, -1.3929e-02,  4.4173e-02,  ..., -4.3772e-03,
                4.5299e-02,  6.0889e-03],
              [-2.7856e-02, -1.4683e-02, -1.5261e-02,  ..., -2.5578e-02,
               -1.2986e-02,  8.5385e-03],
              [ 1.8987e-02,  1.7251e-02,  2.2474e-03,  ...,  5.2491e-03,
                2.7538e-02,  1.9099e-02]],
    
             [[-2.1429e-02, -2.1452e-02, -1.8188e-03,  ..., -1.9078e-02,
               -2.6875e-02,  1.6612e-02],
              [-1.4193e-02,  2.2775e-02, -7.5320e-03,  ...,  4.0329e-02,
                2.3638e-03,  1.8719e-02],
              [ 1.2235e-02,  7.1510e-03, -2.9710e-02,  ..., -3.6395e-02,
               -1.5017e-02, -3.0130e-02],
              ...,
              [-4.3853e-03,  1.2023e-02,  1.9581e-02,  ...,  1.8423e-02,
               -1.5625e-02, -3.9582e-02],
              [ 2.1389e-02, -1.0032e-02,  3.3228e-02,  ...,  5.2692e-02,
                1.8387e-02,  1.1849e-02],
              [-5.2513e-02,  3.9669e-02, -3.7646e-02,  ..., -3.4407e-02,
               -3.2430e-02, -5.8756e-03]],
    
             [[-1.5025e-02,  5.4491e-03,  3.3839e-03,  ..., -4.3292e-02,
               -4.1655e-02, -6.3504e-02],
              [-2.1221e-02, -1.5059e-03,  6.1776e-04,  ...,  4.8302e-03,
                2.3602e-02,  2.6395e-02],
              [-2.4673e-02, -7.1872e-03, -1.2602e-02,  ...,  3.5135e-02,
               -4.5383e-03,  2.2941e-03],
              ...,
              [-2.0872e-02,  1.5452e-02, -3.1187e-02,  ...,  4.5221e-03,
               -2.5530e-02, -1.4707e-02],
              [ 4.1107e-02, -3.1206e-02,  2.8682e-02,  ..., -7.0993e-03,
                3.0871e-02, -1.2732e-02],
              [-7.9013e-03,  2.0054e-02, -6.2115e-03,  ...,  3.4874e-02,
                2.2789e-03,  1.9972e-02]]]])), ('bn1.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])), ('bn1.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('bn1.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('bn1.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])), ('bn1.num_batches_tracked', tensor(0)), ('layer1.0.conv1.weight', tensor([[[[ 3.0539e-02,  1.9263e-02, -9.5153e-03],
              [-1.6868e-02,  8.1200e-02, -5.0737e-02],
              [-1.4941e-02,  9.8156e-02,  2.4128e-02]],
    
             [[-3.5668e-03, -1.7678e-03,  1.0533e-02],
              [-1.2527e-01, -4.3889e-02, -1.9012e-02],
              [-1.1522e-02, -5.6890e-02,  3.5555e-02]],
    
             [[-1.3885e-02, -4.6435e-02,  8.2544e-02],
              [ 4.4329e-02,  9.3898e-02,  2.1418e-02],
              [-2.4573e-02, -2.0268e-03,  4.4004e-02]],
    
             ...,
    
             [[-1.6858e-02,  1.8185e-02,  9.6267e-02],
              [ 2.6345e-02, -5.6629e-02,  5.5235e-03],
              [-3.3296e-02,  9.4378e-02, -6.3945e-02]],
    
             [[ 2.9833e-02,  8.4714e-02,  3.4327e-02],
              [ 8.7805e-02,  1.6598e-03,  1.1578e-01],
              [-1.2608e-02, -7.5151e-03,  5.4985e-02]],
    
             [[-1.2946e-01, -3.8235e-02,  2.0858e-02],
              [ 3.6629e-02, -8.0922e-03,  4.9266e-02],
              [ 5.9833e-02, -1.4829e-01, -7.2516e-03]]],


​    
            [[[ 3.1191e-02, -4.3731e-02, -2.6382e-02],
              [ 1.4370e-01, -1.5643e-02,  3.0076e-02],
              [-3.9352e-02,  3.6075e-02,  1.3779e-03]],
    
             [[-9.7976e-02, -4.2582e-02,  4.7527e-02],
              [-6.0998e-02,  1.4464e-02, -4.8669e-02],
              [-5.6894e-02,  8.1457e-02, -2.4365e-02]],
    
             [[ 1.6716e-02,  1.4763e-04, -6.4492e-02],
              [-4.4191e-02,  1.8873e-01,  8.8353e-02],
              [ 2.9974e-02,  1.6756e-02,  2.0861e-02]],
    
             ...,
    
             [[ 1.2659e-01,  5.4396e-02, -3.1809e-02],
              [-2.4254e-02, -2.8308e-02, -5.3568e-02],
              [ 7.3916e-02,  1.2409e-01,  1.6906e-02]],
    
             [[ 8.7342e-02,  4.5410e-02, -1.6811e-02],
              [-1.1112e-02, -3.1272e-02,  5.4239e-02],
              [-3.2455e-02,  4.1273e-02,  5.1288e-03]],
    
             [[-5.1213e-02,  6.2314e-02, -1.4645e-02],
              [ 4.9925e-02, -3.3413e-02, -1.7102e-02],
              [-3.2561e-02,  2.9704e-02,  1.0166e-01]]],


​    
            [[[ 1.7685e-02, -4.6276e-02, -6.1931e-02],
              [-1.6014e-02, -1.3426e-01,  1.3155e-01],
              [-1.0861e-02,  1.0231e-01,  5.3349e-02]],
    
             [[ 4.5723e-02,  4.2911e-02, -8.4300e-03],
              [ 1.5883e-02,  9.9476e-02, -2.7056e-02],
              [ 1.6891e-02, -1.5752e-04, -9.7016e-02]],
    
             [[-1.2807e-01, -5.4600e-02, -2.2462e-02],
              [-6.2221e-03, -1.8678e-02, -9.5671e-02],
              [-7.4823e-02,  2.1588e-02, -4.0575e-02]],
    
             ...,
    
             [[ 5.5803e-03, -1.7282e-02,  3.0226e-02],
              [ 5.1910e-02,  6.6598e-02,  1.0962e-01],
              [-8.4714e-02,  7.3650e-02, -2.5640e-02]],
    
             [[ 8.5122e-02,  4.1985e-02, -1.2679e-03],
              [-1.4886e-01,  4.8878e-02, -2.1148e-02],
              [-8.1039e-02, -7.3524e-02, -3.2161e-02]],
    
             [[-8.3176e-02, -5.2273e-02,  7.0170e-02],
              [-9.2101e-02, -1.6115e-03, -8.0622e-02],
              [ 6.2005e-02,  1.0547e-01, -5.1940e-02]]],


​    
            ...,


​    
            [[[ 4.6587e-02,  3.1479e-02, -3.1340e-02],
              [ 6.5531e-02, -5.0680e-02,  5.0297e-02],
              [-3.1221e-02,  4.4492e-02, -4.1228e-02]],
    
             [[-3.2066e-02, -3.0490e-02,  6.1165e-02],
              [-7.8898e-03, -8.2159e-02, -2.2750e-02],
              [ 3.4592e-02,  1.3200e-01,  8.1046e-02]],
    
             [[-5.5585e-02, -5.0348e-03, -1.4225e-02],
              [ 2.0807e-02,  2.7146e-02, -9.3803e-02],
              [ 4.9110e-02, -1.8202e-02, -6.5716e-02]],
    
             ...,
    
             [[ 6.8095e-02, -8.5798e-02,  5.9580e-02],
              [-2.2132e-03, -6.3908e-02, -3.7729e-02],
              [-3.3713e-02,  4.3529e-02,  2.3565e-02]],
    
             [[-1.2734e-02,  4.0865e-02, -2.0637e-02],
              [ 2.6333e-02,  3.1064e-02, -1.1274e-01],
              [ 4.4891e-02, -5.4836e-02, -1.0355e-01]],
    
             [[ 1.7384e-02,  8.2354e-02,  3.6510e-02],
              [-7.5621e-03,  5.1190e-02, -1.0679e-01],
              [-2.5223e-02,  1.5098e-02, -8.4929e-02]]],


​    
            [[[-2.3767e-02, -1.9151e-02, -1.0226e-01],
              [ 8.5812e-02, -6.7412e-02, -6.5305e-02],
              [-3.5956e-02,  2.2744e-02, -1.7631e-02]],
    
             [[-9.5714e-02, -7.8447e-02, -3.2280e-02],
              [ 7.4174e-02, -4.2331e-02, -9.6931e-02],
              [-1.7788e-02, -1.7947e-02, -2.3771e-02]],
    
             [[-1.7588e-02, -1.0841e-01, -1.7130e-02],
              [-1.1424e-01, -2.8594e-02,  2.8611e-02],
              [ 3.2761e-02, -6.3870e-02,  5.6570e-02]],
    
             ...,
    
             [[-5.4793e-02, -8.1073e-02, -3.8775e-02],
              [ 5.4089e-02,  8.1337e-02, -1.2124e-01],
              [ 9.2851e-02,  1.0174e-02,  1.1949e-03]],
    
             [[ 1.2050e-02,  8.1603e-02, -1.9838e-02],
              [-2.0533e-02,  4.8026e-02,  1.2331e-02],
              [-3.0673e-03,  1.5969e-02, -7.5717e-02]],
    
             [[ 2.3808e-03, -4.0620e-02,  5.3948e-02],
              [-1.8374e-02,  7.5871e-03,  5.1573e-02],
              [-4.2799e-02, -2.3825e-02,  9.0677e-03]]],


​    
            [[[-1.4914e-01,  6.0750e-02, -1.5681e-02],
              [ 8.3451e-02, -1.2266e-02,  1.3369e-02],
              [-1.2846e-01, -3.8551e-02,  9.5243e-03]],
    
             [[ 6.4532e-02, -9.8366e-03,  5.3057e-02],
              [ 8.9660e-02,  1.6758e-03,  3.0947e-02],
              [ 1.0565e-01, -1.0911e-01, -3.9327e-02]],
    
             [[-3.4817e-02,  1.1678e-01, -1.0543e-02],
              [ 4.7430e-02,  2.3337e-02,  6.9885e-02],
              [ 2.7513e-02, -8.1519e-03, -7.1742e-02]],
    
             ...,
    
             [[ 3.2491e-02, -7.0260e-02, -3.3710e-02],
              [ 1.0237e-02,  9.9978e-02,  2.9672e-02],
              [-9.3573e-03,  1.0182e-01,  1.6146e-02]],
    
             [[ 1.6679e-01,  2.2240e-02, -8.0446e-02],
              [-7.6930e-02, -4.4717e-02,  3.0263e-02],
              [ 6.5176e-02,  1.7422e-03, -1.6081e-02]],
    
             [[ 5.1144e-02, -6.7699e-02, -7.4803e-02],
              [-6.8858e-03,  8.3474e-02, -1.4611e-02],
              [ 8.9938e-03, -5.9545e-02,  6.8011e-02]]]])), ('layer1.0.bn1.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])), ('layer1.0.bn1.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer1.0.bn1.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer1.0.bn1.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])), ('layer1.0.bn1.num_batches_tracked', tensor(0)), ('layer1.0.conv2.weight', tensor([[[[ 0.0083, -0.0330,  0.0239],
              [ 0.1444,  0.0189,  0.0167],
              [-0.0436,  0.0294,  0.0107]],
    
             [[-0.1209, -0.0254, -0.0941],
              [-0.0487, -0.0630, -0.0906],
              [ 0.0006,  0.0342,  0.0982]],
    
             [[-0.0041,  0.0920, -0.0070],
              [-0.0273,  0.0331, -0.0145],
              [-0.1147, -0.0142,  0.0629]],
    
             ...,
    
             [[-0.0096, -0.0182, -0.0159],
              [ 0.1239, -0.0338,  0.0627],
              [-0.0305, -0.0864,  0.0739]],
    
             [[-0.0162, -0.0785, -0.1316],
              [ 0.0478,  0.0196,  0.0629],
              [ 0.0621, -0.0325,  0.0875]],
    
             [[ 0.0043, -0.1119,  0.0232],
              [ 0.0625,  0.1038, -0.0756],
              [ 0.0425, -0.0416,  0.0315]]],


​    
            [[[-0.0910, -0.0469,  0.0288],
              [ 0.0337, -0.0080, -0.0733],
              [ 0.0270, -0.0115, -0.0522]],
    
             [[-0.0318,  0.0210,  0.0072],
              [ 0.0030,  0.0356,  0.0582],
              [-0.0695,  0.0288,  0.0182]],
    
             [[-0.0327,  0.0090, -0.0537],
              [-0.0192,  0.0546, -0.0595],
              [-0.0541,  0.0084,  0.0343]],
    
             ...,
    
             [[ 0.0542,  0.0615, -0.0681],
              [-0.0219,  0.1173, -0.0402],
              [ 0.0097, -0.0537,  0.0559]],
    
             [[-0.0379,  0.0716, -0.0035],
              [ 0.0098,  0.0187,  0.0190],
              [-0.0257, -0.0989, -0.0778]],
    
             [[ 0.0420,  0.0235,  0.0480],
              [-0.0723, -0.0717, -0.0880],
              [ 0.0378, -0.0219,  0.0515]]],


​    
            [[[-0.0492,  0.0226,  0.0223],
              [-0.0254, -0.0417,  0.0176],
              [-0.0005, -0.0251, -0.0185]],
    
             [[-0.0328, -0.0242, -0.0487],
              [-0.0183,  0.0093, -0.0453],
              [ 0.0281, -0.0482,  0.0656]],
    
             [[-0.0157, -0.0337,  0.1204],
              [ 0.0717,  0.0375,  0.0046],
              [ 0.1011, -0.0440,  0.0141]],
    
             ...,
    
             [[-0.0673, -0.0259, -0.0246],
              [-0.0129,  0.0625, -0.0236],
              [ 0.0643,  0.0036, -0.0228]],
    
             [[-0.0136,  0.0921, -0.0378],
              [-0.0193, -0.0240, -0.0241],
              [-0.0187, -0.0062, -0.0296]],
    
             [[ 0.1154, -0.0782,  0.0653],
              [ 0.0156, -0.0046,  0.0296],
              [ 0.1194,  0.0948, -0.0432]]],


​    
            ...,


​    
            [[[ 0.0365,  0.0108,  0.0052],
              [-0.0331, -0.0711,  0.0663],
              [ 0.0771,  0.0402,  0.0853]],
    
             [[-0.0210,  0.0714,  0.0206],
              [ 0.0606, -0.0247, -0.0115],
              [-0.0478, -0.0496,  0.1077]],
    
             [[-0.0271,  0.0521,  0.0589],
              [-0.0165,  0.0817, -0.0201],
              [-0.0580, -0.0378, -0.0737]],
    
             ...,
    
             [[ 0.0976,  0.0284,  0.0424],
              [-0.0407, -0.0614, -0.0880],
              [-0.0502, -0.0829,  0.0614]],
    
             [[-0.0403,  0.0362, -0.0232],
              [-0.0203, -0.0789,  0.0276],
              [-0.0240,  0.0607,  0.0470]],
    
             [[-0.0507, -0.0425, -0.0552],
              [ 0.0509,  0.0233, -0.0280],
              [-0.1407, -0.0146, -0.0659]]],


​    
            [[[-0.0501,  0.0019,  0.0723],
              [-0.0173,  0.0494, -0.0947],
              [ 0.0874, -0.0464, -0.0561]],
    
             [[ 0.1013,  0.0606, -0.0128],
              [ 0.0755, -0.0198,  0.0805],
              [-0.0607,  0.0667,  0.0699]],
    
             [[-0.0165,  0.0337, -0.0096],
              [-0.0110, -0.0783,  0.0059],
              [-0.0598, -0.0082, -0.0286]],
    
             ...,
    
             [[ 0.0175,  0.0038,  0.0052],
              [-0.0550, -0.0232, -0.0211],
              [ 0.0114,  0.0307,  0.0737]],
    
             [[-0.0087, -0.0385, -0.0230],
              [-0.0099,  0.0482,  0.0191],
              [-0.0498,  0.0061,  0.0107]],
    
             [[-0.0619, -0.0375, -0.0460],
              [-0.0748,  0.0587,  0.0155],
              [-0.0382, -0.1296,  0.0401]]],


​    
            [[[ 0.1032,  0.0737, -0.0121],
              [-0.0740,  0.0474,  0.1038],
              [ 0.0264,  0.0181, -0.0661]],
    
             [[ 0.0893, -0.1562,  0.0106],
              [-0.0010, -0.0407,  0.0568],
              [-0.0355, -0.0063, -0.0501]],
    
             [[ 0.0838, -0.0432, -0.1053],
              [-0.0758, -0.0197,  0.0002],
              [-0.0392,  0.0217, -0.0483]],
    
             ...,
    
             [[ 0.0440,  0.1094,  0.0016],
              [-0.0285,  0.0497,  0.0423],
              [-0.0162, -0.0153, -0.0864]],
    
             [[ 0.0398,  0.0847,  0.0057],
              [-0.0832, -0.0192,  0.0490],
              [ 0.0811,  0.0741, -0.0071]],
    
             [[-0.0179, -0.1659, -0.0107],
              [ 0.0372,  0.0517,  0.0248],
              [-0.0744,  0.0986, -0.0061]]]])), ('layer1.0.bn2.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])), ('layer1.0.bn2.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer1.0.bn2.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer1.0.bn2.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])), ('layer1.0.bn2.num_batches_tracked', tensor(0)), ('layer1.1.conv1.weight', tensor([[[[-0.1661, -0.0325,  0.0481],
              [ 0.0180,  0.0693,  0.0220],
              [-0.0003, -0.0421,  0.1044]],
    
             [[-0.0187,  0.0235, -0.0151],
              [-0.0813,  0.0226, -0.0949],
              [ 0.0097,  0.0214,  0.0410]],
    
             [[-0.0135,  0.0218,  0.0028],
              [ 0.0019, -0.0467, -0.0480],
              [ 0.0054, -0.0323, -0.0525]],
    
             ...,
    
             [[-0.1421,  0.0737,  0.0375],
              [ 0.0137,  0.0635, -0.0040],
              [ 0.0347, -0.0009, -0.0568]],
    
             [[ 0.0090,  0.0043,  0.0387],
              [-0.0073, -0.0276,  0.0165],
              [ 0.0033, -0.1153, -0.0409]],
    
             [[-0.0409, -0.0183,  0.0113],
              [-0.0867,  0.0243, -0.0858],
              [-0.0934,  0.1038, -0.1234]]],


​    
            [[[-0.0077,  0.0054, -0.0260],
              [ 0.0113,  0.0551,  0.0002],
              [-0.0482, -0.0336,  0.0117]],
    
             [[ 0.0295, -0.0811,  0.0316],
              [-0.0098, -0.0331, -0.0203],
              [-0.0209,  0.0733, -0.0355]],
    
             [[-0.0141,  0.0304,  0.0505],
              [-0.0175,  0.0566, -0.0006],
              [-0.0098,  0.0682,  0.0371]],
    
             ...,
    
             [[ 0.0124, -0.0007, -0.0031],
              [-0.0143,  0.1253, -0.0289],
              [-0.0202, -0.0201, -0.0772]],
    
             [[-0.0043,  0.0408, -0.0973],
              [-0.0419, -0.0009,  0.0605],
              [ 0.0404, -0.0502,  0.0407]],
    
             [[ 0.0026,  0.0231, -0.0167],
              [ 0.0552, -0.0043,  0.0487],
              [ 0.0675,  0.0704, -0.0067]]],


​    
            [[[-0.0854,  0.1115,  0.0739],
              [ 0.0499, -0.1231,  0.0368],
              [-0.0748,  0.0232, -0.0357]],
    
             [[ 0.0052,  0.0502, -0.0465],
              [ 0.0301,  0.0489,  0.0320],
              [ 0.0301, -0.0610,  0.0776]],
    
             [[ 0.0165,  0.0352,  0.0138],
              [ 0.0055,  0.0054, -0.0178],
              [ 0.0606,  0.0947,  0.0279]],
    
             ...,
    
             [[-0.0012, -0.0921,  0.0333],
              [-0.0367, -0.1523, -0.0472],
              [ 0.0398,  0.0314, -0.1023]],
    
             [[ 0.1261, -0.0244, -0.0582],
              [-0.1445, -0.0890, -0.1329],
              [-0.0689,  0.1158,  0.0307]],
    
             [[-0.0027, -0.0194,  0.0237],
              [ 0.0144, -0.0117,  0.0261],
              [-0.0502, -0.0103, -0.0098]]],


​    
            ...,


​    
            [[[ 0.1260,  0.0500, -0.0215],
              [-0.0723, -0.1556, -0.0259],
              [-0.0719, -0.0888, -0.0642]],
    
             [[ 0.0804,  0.0390, -0.0229],
              [-0.0479, -0.0119,  0.0435],
              [ 0.1310,  0.0729, -0.0791]],
    
             [[ 0.0423, -0.1180, -0.0675],
              [-0.0425, -0.1203,  0.0044],
              [ 0.0638,  0.0859,  0.0578]],
    
             ...,
    
             [[ 0.0593,  0.0736, -0.0529],
              [-0.0108, -0.0475, -0.0877],
              [-0.0467,  0.0087,  0.0352]],
    
             [[ 0.2153, -0.1379,  0.0200],
              [-0.0499,  0.0454, -0.1016],
              [ 0.0157, -0.0685,  0.1457]],
    
             [[-0.0687,  0.0630, -0.0864],
              [-0.1005,  0.0715,  0.0174],
              [-0.0171,  0.0036, -0.0891]]],


​    
            [[[-0.0446, -0.0626, -0.0922],
              [-0.0193,  0.0444,  0.0487],
              [ 0.0412,  0.0356, -0.0672]],
    
             [[-0.0293,  0.0417, -0.0478],
              [-0.0681, -0.0323, -0.0284],
              [-0.0575, -0.0768,  0.0582]],
    
             [[-0.0893, -0.0532, -0.0345],
              [-0.0412, -0.0468,  0.0752],
              [-0.0912,  0.0663,  0.0715]],
    
             ...,
    
             [[ 0.1167,  0.0335,  0.0335],
              [-0.0106, -0.1084, -0.0503],
              [ 0.0413,  0.0730, -0.0194]],
    
             [[ 0.1289,  0.0234, -0.0755],
              [ 0.0372,  0.0043,  0.0784],
              [-0.0068,  0.0358,  0.0871]],
    
             [[ 0.0393,  0.0068,  0.0128],
              [ 0.0424, -0.0368, -0.0324],
              [ 0.0648,  0.0259,  0.0306]]],


​    
            [[[ 0.0691,  0.0583,  0.0003],
              [-0.1522, -0.0230, -0.0126],
              [-0.0031,  0.0124,  0.1019]],
    
             [[ 0.0132,  0.0181, -0.0091],
              [ 0.0380, -0.1025, -0.0396],
              [ 0.1248, -0.0299,  0.0515]],
    
             [[-0.0309, -0.0312, -0.0271],
              [ 0.0667,  0.0953, -0.0649],
              [ 0.0105, -0.0209, -0.0163]],
    
             ...,
    
             [[-0.1490,  0.0701, -0.0396],
              [-0.0523,  0.0123, -0.0107],
              [ 0.1744,  0.0265,  0.0721]],
    
             [[ 0.0931,  0.0099,  0.0383],
              [ 0.0188, -0.0077, -0.0936],
              [-0.0108,  0.0174,  0.1103]],
    
             [[-0.0816, -0.0310, -0.0625],
              [ 0.0325, -0.0471,  0.0362],
              [-0.0239, -0.0185,  0.0184]]]])), ('layer1.1.bn1.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])), ('layer1.1.bn1.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer1.1.bn1.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer1.1.bn1.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])), ('layer1.1.bn1.num_batches_tracked', tensor(0)), ('layer1.1.conv2.weight', tensor([[[[-0.0418,  0.0517, -0.0367],
              [-0.1072,  0.0787, -0.0783],
              [ 0.0037, -0.0077,  0.0740]],
    
             [[-0.0156, -0.0413, -0.0214],
              [ 0.0883, -0.1097,  0.0673],
              [ 0.0476, -0.1062,  0.0265]],
    
             [[-0.0783, -0.0212, -0.0339],
              [-0.0383,  0.0644,  0.0198],
              [ 0.0287,  0.0573, -0.0919]],
    
             ...,
    
             [[ 0.0576,  0.0018, -0.0310],
              [-0.0161, -0.1002,  0.0616],
              [-0.0237,  0.0053,  0.0048]],
    
             [[-0.0185, -0.0040,  0.0640],
              [-0.0283,  0.0555,  0.0292],
              [ 0.0827, -0.0251,  0.0036]],
    
             [[ 0.0546,  0.1813, -0.0761],
              [ 0.0566, -0.0218, -0.1569],
              [ 0.1152, -0.0317,  0.0405]]],


​    
            [[[-0.1422,  0.0019, -0.1374],
              [ 0.1368, -0.0819, -0.0020],
              [-0.0233,  0.0156,  0.0488]],
    
             [[-0.0368,  0.0828, -0.0558],
              [ 0.0523, -0.0483, -0.0482],
              [ 0.1053,  0.1068,  0.0187]],
    
             [[ 0.0198, -0.0718, -0.1419],
              [-0.0948,  0.0147,  0.0190],
              [ 0.0108, -0.0117, -0.0271]],
    
             ...,
    
             [[ 0.0678,  0.0224,  0.0125],
              [ 0.0632, -0.0364,  0.0726],
              [ 0.0148, -0.1674, -0.1287]],
    
             [[ 0.0382,  0.0220, -0.0394],
              [-0.0342, -0.0371,  0.0544],
              [-0.0029, -0.0019, -0.0645]],
    
             [[-0.0156, -0.0399, -0.1027],
              [ 0.1213,  0.0268, -0.0537],
              [ 0.0584, -0.0146,  0.0157]]],


​    
            [[[ 0.0946, -0.0692,  0.1115],
              [ 0.0107,  0.0624, -0.0535],
              [-0.0596,  0.0018,  0.0396]],
    
             [[-0.0548,  0.1033,  0.0296],
              [ 0.0219, -0.0318, -0.0630],
              [-0.0180, -0.0005,  0.0548]],
    
             [[ 0.0556,  0.1429,  0.0296],
              [ 0.0171, -0.0699, -0.0222],
              [-0.1353, -0.0404,  0.0345]],
    
             ...,
    
             [[-0.0066, -0.0689, -0.1081],
              [ 0.0955,  0.0188,  0.0044],
              [-0.0013, -0.0772,  0.0168]],
    
             [[-0.0169,  0.0162, -0.0034],
              [-0.0236,  0.0275,  0.0925],
              [-0.0112,  0.0091, -0.0394]],
    
             [[ 0.0500,  0.0612, -0.0636],
              [-0.0369,  0.1176, -0.0574],
              [-0.0291, -0.0182,  0.0071]]],


​    
            ...,


​    
            [[[ 0.1107, -0.0451, -0.0485],
              [ 0.0133, -0.0131,  0.0128],
              [ 0.0743,  0.0387, -0.0319]],
    
             [[-0.0395, -0.0511,  0.0265],
              [ 0.0023,  0.0313,  0.0538],
              [ 0.0274, -0.0821,  0.0272]],
    
             [[ 0.0004,  0.0754, -0.0057],
              [ 0.0763,  0.0108, -0.0086],
              [-0.0390,  0.0788, -0.0507]],
    
             ...,
    
             [[ 0.0911,  0.0784,  0.0418],
              [ 0.0081,  0.0178, -0.0586],
              [ 0.0143,  0.0875, -0.0307]],
    
             [[ 0.1231,  0.0539,  0.0040],
              [ 0.0395, -0.0399, -0.1014],
              [ 0.0648, -0.0134,  0.0969]],
    
             [[-0.0551, -0.0911,  0.0094],
              [-0.0094, -0.1176,  0.0225],
              [ 0.0309, -0.0439, -0.0350]]],


​    
            [[[-0.0802, -0.0111, -0.0389],
              [-0.0039, -0.0396, -0.0477],
              [ 0.0213, -0.0263,  0.0047]],
    
             [[-0.0593, -0.0311, -0.0076],
              [ 0.1850,  0.0092, -0.0523],
              [-0.0179,  0.1118, -0.0099]],
    
             [[-0.0127,  0.0157,  0.0159],
              [ 0.0758, -0.0141, -0.0721],
              [ 0.0239,  0.1099, -0.0094]],
    
             ...,
    
             [[-0.0427,  0.0406,  0.0056],
              [-0.0218, -0.0121, -0.0541],
              [ 0.0533, -0.1114, -0.0181]],
    
             [[-0.0203, -0.0509, -0.0655],
              [ 0.0229,  0.0841,  0.0253],
              [ 0.0395, -0.0941, -0.0103]],
    
             [[-0.0830,  0.0291, -0.0449],
              [-0.0625,  0.0190,  0.0918],
              [-0.0615,  0.0039,  0.0896]]],


​    
            [[[-0.0533,  0.0376, -0.0035],
              [ 0.0514,  0.0254, -0.1093],
              [-0.0729, -0.0984,  0.1304]],
    
             [[-0.0579,  0.0398, -0.0262],
              [-0.0217,  0.0503, -0.0140],
              [-0.0552, -0.0712, -0.0095]],
    
             [[ 0.0142, -0.0578,  0.0958],
              [-0.0318,  0.0626,  0.0492],
              [ 0.0109, -0.0047,  0.0003]],
    
             ...,
    
             [[-0.0039,  0.0532,  0.0530],
              [ 0.0090,  0.0223,  0.0167],
              [-0.0387, -0.0130,  0.0584]],
    
             [[ 0.0535, -0.1143,  0.0704],
              [ 0.0114, -0.0757, -0.0231],
              [ 0.1362, -0.0145, -0.0142]],
    
             [[ 0.0470, -0.0066,  0.0616],
              [ 0.0179,  0.0076,  0.0384],
              [-0.0093, -0.0557, -0.0846]]]])), ('layer1.1.bn2.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])), ('layer1.1.bn2.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer1.1.bn2.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer1.1.bn2.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])), ('layer1.1.bn2.num_batches_tracked', tensor(0)), ('layer2.0.conv1.weight', tensor([[[[ 0.0060, -0.0458,  0.0395],
              [-0.0618, -0.0014, -0.0316],
              [ 0.0437,  0.0058,  0.0027]],
    
             [[-0.0855, -0.0436, -0.0019],
              [-0.0467,  0.0367, -0.0278],
              [-0.0004,  0.0849,  0.0615]],
    
             [[-0.0099,  0.0283,  0.0683],
              [ 0.0167,  0.0170,  0.0051],
              [-0.0412, -0.0289, -0.0280]],
    
             ...,
    
             [[ 0.0478, -0.0383,  0.0187],
              [ 0.0094,  0.0047,  0.0491],
              [ 0.0179,  0.0175, -0.0291]],
    
             [[-0.0653, -0.0411, -0.0138],
              [ 0.1275,  0.0323,  0.0157],
              [-0.0130,  0.0325,  0.0376]],
    
             [[-0.0172, -0.0395,  0.0027],
              [ 0.0210,  0.0518,  0.0195],
              [-0.0436,  0.0678,  0.0457]]],


​    
            [[[-0.0013, -0.0328, -0.0262],
              [-0.0115,  0.0324, -0.0278],
              [-0.0248, -0.0294, -0.0380]],
    
             [[ 0.0403, -0.0017,  0.0553],
              [ 0.0593, -0.0345, -0.0149],
              [ 0.0094,  0.0113,  0.0617]],
    
             [[ 0.0438,  0.0013,  0.0569],
              [ 0.0134,  0.0698,  0.0032],
              [-0.0487,  0.0060, -0.0422]],
    
             ...,
    
             [[-0.0056,  0.0620, -0.0209],
              [-0.0107,  0.0245,  0.0321],
              [-0.0604,  0.0308, -0.0498]],
    
             [[-0.0384,  0.0313,  0.0267],
              [-0.0731,  0.0370,  0.0448],
              [ 0.0489,  0.0586, -0.0123]],
    
             [[-0.0310,  0.0247,  0.0184],
              [ 0.0207, -0.0285, -0.0191],
              [ 0.0201, -0.0094, -0.0130]]],


​    
            [[[-0.0183, -0.0379, -0.0875],
              [-0.0086, -0.0389, -0.0356],
              [ 0.0400, -0.0403,  0.1065]],
    
             [[-0.0492,  0.0258,  0.0319],
              [ 0.0183,  0.0280, -0.0278],
              [-0.0338, -0.1121, -0.0628]],
    
             [[-0.0242, -0.0331, -0.0384],
              [-0.0234, -0.0100, -0.0630],
              [ 0.0317,  0.0313, -0.0515]],
    
             ...,
    
             [[-0.0236, -0.0411,  0.0166],
              [ 0.0699,  0.0918,  0.0101],
              [-0.0005, -0.0006, -0.0425]],
    
             [[-0.0410,  0.0628, -0.0840],
              [ 0.0098,  0.0228, -0.0583],
              [-0.0094,  0.0215, -0.0637]],
    
             [[ 0.0215,  0.0117, -0.0682],
              [-0.0111,  0.0199,  0.0780],
              [ 0.0050,  0.0571,  0.0253]]],


​    
            ...,


​    
            [[[-0.0746, -0.0486, -0.0010],
              [ 0.0341,  0.0851, -0.0946],
              [ 0.0124,  0.0472, -0.0573]],
    
             [[-0.0189,  0.0290, -0.0303],
              [-0.0232, -0.0205, -0.0168],
              [-0.0034,  0.0630,  0.0066]],
    
             [[-0.0389, -0.0413, -0.0489],
              [-0.0304, -0.0109, -0.0292],
              [ 0.0476,  0.0005,  0.0348]],
    
             ...,
    
             [[ 0.0478,  0.0152,  0.0667],
              [ 0.0524, -0.0323,  0.0056],
              [-0.0133, -0.0292,  0.0614]],
    
             [[ 0.0556, -0.0114,  0.0356],
              [-0.0693,  0.0634, -0.0174],
              [ 0.0692,  0.0518, -0.0460]],
    
             [[-0.0132,  0.0179, -0.0121],
              [-0.0056,  0.0573, -0.0743],
              [-0.0128, -0.0058, -0.0049]]],


​    
            [[[ 0.0172,  0.0307,  0.0437],
              [-0.0358, -0.0098,  0.0533],
              [-0.0702, -0.0728,  0.0780]],
    
             [[ 0.0749, -0.0362, -0.0053],
              [ 0.0096, -0.0204, -0.0239],
              [-0.0154, -0.0101, -0.0086]],
    
             [[ 0.0047,  0.0374, -0.0289],
              [-0.0600,  0.0487, -0.0130],
              [-0.0032, -0.0242,  0.0271]],
    
             ...,
    
             [[-0.0029,  0.0010, -0.0515],
              [ 0.0176, -0.0491, -0.0399],
              [-0.0052,  0.0752, -0.0279]],
    
             [[ 0.0449,  0.0155, -0.0454],
              [ 0.0128,  0.0712,  0.0472],
              [-0.0417,  0.0190,  0.0454]],
    
             [[-0.0674,  0.0464,  0.0473],
              [ 0.0133, -0.0986, -0.0194],
              [ 0.0300,  0.0219, -0.0223]]],


​    
            [[[ 0.0609, -0.0621,  0.0276],
              [ 0.0091, -0.0020, -0.0011],
              [ 0.0309, -0.0084, -0.0435]],
    
             [[ 0.0111,  0.0236,  0.0367],
              [ 0.0792,  0.0743, -0.0432],
              [-0.0540,  0.0395,  0.0420]],
    
             [[-0.0225, -0.0245, -0.0029],
              [-0.0392,  0.0383,  0.0899],
              [-0.0118,  0.0049, -0.0263]],
    
             ...,
    
             [[ 0.1031,  0.0167, -0.0020],
              [-0.0125, -0.0907,  0.0373],
              [ 0.0090, -0.0008,  0.0524]],
    
             [[ 0.0812,  0.0085, -0.0226],
              [ 0.0177, -0.0148, -0.0286],
              [-0.0171, -0.0206,  0.0571]],
    
             [[-0.0742,  0.0241,  0.0427],
              [-0.0483, -0.0376, -0.0237],
              [-0.0554, -0.0395, -0.0414]]]])), ('layer2.0.bn1.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1.])), ('layer2.0.bn1.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer2.0.bn1.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer2.0.bn1.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1.])), ('layer2.0.bn1.num_batches_tracked', tensor(0)), ('layer2.0.conv2.weight', tensor([[[[ 0.0540,  0.0808, -0.0557],
              [-0.0042, -0.0145,  0.0696],
              [-0.0208,  0.0225, -0.0438]],
    
             [[ 0.0146,  0.0077, -0.0104],
              [ 0.0063,  0.0570, -0.0525],
              [-0.0059,  0.0452, -0.0325]],
    
             [[ 0.0307,  0.0341, -0.0237],
              [ 0.0053, -0.0322,  0.0116],
              [ 0.0380, -0.0227,  0.0056]],
    
             ...,
    
             [[ 0.0728, -0.0403,  0.0429],
              [ 0.0005,  0.0043,  0.0282],
              [-0.0084, -0.0714, -0.0208]],
    
             [[-0.0540,  0.0761,  0.0295],
              [-0.0189,  0.0028, -0.0063],
              [-0.0500,  0.0112,  0.0140]],
    
             [[ 0.0347,  0.0827,  0.0492],
              [-0.0118, -0.0020,  0.0466],
              [ 0.0434,  0.0436, -0.0186]]],


​    
            [[[ 0.0460,  0.0043,  0.0196],
              [-0.0271, -0.0468,  0.0041],
              [ 0.0331, -0.0697,  0.0376]],
    
             [[ 0.0281, -0.0401,  0.0246],
              [-0.0353, -0.0218,  0.0143],
              [ 0.0669,  0.0624,  0.0319]],
    
             [[-0.0116,  0.0075, -0.0165],
              [ 0.0110, -0.0511,  0.0491],
              [ 0.0134,  0.0530,  0.0903]],
    
             ...,
    
             [[-0.0339,  0.0166,  0.0286],
              [ 0.0027,  0.0117,  0.0407],
              [-0.0431, -0.0342,  0.0097]],
    
             [[-0.0032,  0.0125, -0.0275],
              [-0.0431,  0.0234, -0.0412],
              [ 0.0423,  0.0734, -0.0414]],
    
             [[-0.0598,  0.0072,  0.0379],
              [ 0.0426, -0.0440, -0.0191],
              [-0.0481,  0.0893,  0.0237]]],


​    
            [[[-0.0825,  0.0553,  0.0074],
              [-0.0255, -0.0539,  0.0232],
              [ 0.0644, -0.0174, -0.0372]],
    
             [[ 0.0341, -0.0136,  0.0040],
              [ 0.0033, -0.0074,  0.0289],
              [ 0.0321,  0.0334,  0.0246]],
    
             [[ 0.0643,  0.0417,  0.0225],
              [ 0.0257, -0.0056,  0.0148],
              [ 0.0348,  0.0281, -0.0416]],
    
             ...,
    
             [[ 0.0449,  0.0257, -0.0047],
              [-0.0270,  0.0014, -0.0060],
              [ 0.0515, -0.0391, -0.0946]],
    
             [[ 0.0207,  0.0787,  0.0350],
              [-0.0195,  0.0555,  0.0372],
              [ 0.0180,  0.0108, -0.0047]],
    
             [[-0.0596, -0.0661, -0.0033],
              [ 0.0371,  0.0503, -0.0218],
              [-0.0576, -0.0514,  0.0902]]],


​    
            ...,


​    
            [[[ 0.0294,  0.0230, -0.0115],
              [-0.0338, -0.0647, -0.0426],
              [-0.0279, -0.0551,  0.0729]],
    
             [[ 0.0125,  0.0363,  0.0218],
              [ 0.0022, -0.0080, -0.0459],
              [-0.0155, -0.0217, -0.0062]],
    
             [[ 0.0237, -0.0554,  0.0558],
              [-0.0203,  0.0602, -0.0062],
              [ 0.0857,  0.0023,  0.0523]],
    
             ...,
    
             [[ 0.0596, -0.0441,  0.0076],
              [-0.0520, -0.0061,  0.0128],
              [ 0.0390,  0.0791,  0.0416]],
    
             [[-0.0093, -0.0717, -0.0024],
              [-0.0657, -0.0172, -0.0540],
              [ 0.0390,  0.0569, -0.0246]],
    
             [[-0.0669, -0.0047, -0.0136],
              [-0.0264,  0.0379,  0.0256],
              [ 0.0443, -0.0414,  0.0119]]],


​    
            [[[-0.0158,  0.0465, -0.0227],
              [-0.0108, -0.0593,  0.0290],
              [-0.0309,  0.0075, -0.0199]],
    
             [[-0.0493, -0.0702, -0.0206],
              [-0.0124,  0.0799,  0.0100],
              [-0.0214,  0.0253, -0.0078]],
    
             [[-0.0163,  0.0854,  0.0402],
              [ 0.0191, -0.0416,  0.0141],
              [ 0.0074, -0.0067,  0.0804]],
    
             ...,
    
             [[ 0.0352,  0.0655, -0.0062],
              [ 0.0447,  0.0479, -0.0708],
              [-0.0972, -0.0279,  0.0688]],
    
             [[ 0.0050, -0.0125,  0.0006],
              [-0.0513,  0.0188,  0.0887],
              [-0.0286, -0.0418, -0.0104]],
    
             [[-0.0491,  0.1084,  0.0515],
              [-0.0180, -0.0015,  0.0720],
              [ 0.0138, -0.0039, -0.0229]]],


​    
            [[[-0.0086, -0.0610,  0.0271],
              [ 0.0088,  0.0534, -0.0652],
              [ 0.0101, -0.0364,  0.0920]],
    
             [[ 0.0252, -0.0443, -0.0188],
              [ 0.0025, -0.0267, -0.0080],
              [-0.0067, -0.0207, -0.0606]],
    
             [[-0.0613,  0.0134,  0.0378],
              [ 0.0246,  0.0262, -0.0212],
              [ 0.0537,  0.0398, -0.0308]],
    
             ...,
    
             [[-0.0633, -0.0193, -0.0111],
              [-0.0126,  0.0047,  0.0053],
              [-0.0018,  0.0107, -0.0034]],
    
             [[ 0.0355, -0.0341, -0.0109],
              [-0.0062,  0.0130,  0.0540],
              [-0.0594, -0.0286, -0.0381]],
    
             [[ 0.0183,  0.0292, -0.0305],
              [-0.0375,  0.0597,  0.0681],
              [-0.0246, -0.0031,  0.0534]]]])), ('layer2.0.bn2.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1.])), ('layer2.0.bn2.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer2.0.bn2.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer2.0.bn2.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1.])), ('layer2.0.bn2.num_batches_tracked', tensor(0)), ('layer2.0.downsample.0.weight', tensor([[[[ 0.1134]],
    
             [[-0.1649]],
    
             [[-0.2037]],
    
             ...,
    
             [[ 0.1360]],
    
             [[-0.0981]],
    
             [[ 0.0617]]],


​    
            [[[ 0.0900]],
    
             [[-0.1207]],
    
             [[-0.2714]],
    
             ...,
    
             [[-0.1491]],
    
             [[ 0.1718]],
    
             [[ 0.0035]]],


​    
            [[[-0.1024]],
    
             [[-0.0853]],
    
             [[ 0.1771]],
    
             ...,
    
             [[-0.0016]],
    
             [[-0.1849]],
    
             [[ 0.0911]]],


​    
            ...,


​    
            [[[-0.1319]],
    
             [[ 0.0694]],
    
             [[-0.1359]],
    
             ...,
    
             [[ 0.0161]],
    
             [[ 0.1369]],
    
             [[ 0.1154]]],


​    
            [[[-0.1115]],
    
             [[ 0.1137]],
    
             [[-0.2520]],
    
             ...,
    
             [[ 0.0064]],
    
             [[ 0.0804]],
    
             [[-0.1589]]],


​    
            [[[ 0.0434]],
    
             [[ 0.1527]],
    
             [[-0.1698]],
    
             ...,
    
             [[ 0.0994]],
    
             [[ 0.0780]],
    
             [[ 0.0740]]]])), ('layer2.0.downsample.1.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1.])), ('layer2.0.downsample.1.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer2.0.downsample.1.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer2.0.downsample.1.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1.])), ('layer2.0.downsample.1.num_batches_tracked', tensor(0)), ('layer2.1.conv1.weight', tensor([[[[ 9.4917e-03, -4.3838e-02, -1.4113e-02],
              [-1.9540e-02, -1.5401e-02,  1.4366e-02],
              [-7.2431e-02,  1.1573e-02, -1.3649e-02]],
    
             [[ 3.1076e-02,  1.4257e-02,  3.6470e-03],
              [-4.7784e-02, -5.1621e-02,  3.6865e-02],
              [-3.6935e-02, -2.3126e-02,  2.8439e-02]],
    
             [[ 1.4482e-02,  1.5604e-02, -5.7814e-03],
              [ 2.0195e-02,  3.4757e-03, -8.1251e-02],
              [ 2.8865e-03, -4.2297e-02,  6.1348e-02]],
    
             ...,
    
             [[ 6.1181e-02,  3.1304e-02,  1.2904e-02],
              [-2.4378e-02, -4.7457e-02,  2.4194e-02],
              [ 2.4862e-02, -5.0659e-02, -9.5623e-02]],
    
             [[-3.2832e-02, -4.2360e-02,  2.1370e-02],
              [-1.7944e-03, -1.2385e-01, -4.8749e-02],
              [-3.2802e-02,  8.7864e-02, -2.3640e-02]],
    
             [[ 6.9454e-02, -2.9245e-02, -4.7851e-02],
              [ 3.1639e-02,  1.2180e-02, -5.6808e-02],
              [ 1.1535e-02, -4.1574e-02, -1.1260e-02]]],


​    
            [[[-5.7193e-02,  7.4393e-03,  2.2646e-02],
              [-1.6073e-02, -6.0812e-02,  3.1450e-02],
              [ 1.1325e-02,  7.1660e-03,  1.9514e-02]],
    
             [[-3.6434e-03,  5.9549e-02, -1.9878e-02],
              [ 4.5325e-02,  1.5327e-02,  3.3561e-02],
              [-3.9024e-02,  6.6292e-02, -3.1064e-03]],
    
             [[-5.6671e-03, -1.0653e-02,  1.0467e-01],
              [ 4.3120e-02, -2.2607e-02, -7.7391e-02],
              [ 6.2994e-03, -1.5461e-02, -3.6156e-02]],
    
             ...,
    
             [[ 3.7762e-02,  2.3886e-03, -7.0734e-02],
              [-4.2752e-02,  4.1623e-02,  1.5848e-02],
              [ 1.6811e-02, -8.4648e-02, -8.8035e-03]],
    
             [[ 3.5259e-02,  5.1821e-02, -7.0861e-02],
              [-2.0294e-02,  1.6550e-02,  2.0257e-03],
              [ 6.0949e-02,  1.2421e-02,  7.3805e-02]],
    
             [[ 4.3864e-02, -2.3545e-02,  2.6641e-02],
              [-1.3562e-02, -2.0005e-02, -2.2738e-02],
              [-6.9720e-03,  4.0579e-02,  6.4031e-02]]],


​    
            [[[ 5.4271e-02, -1.2097e-02,  9.9753e-02],
              [ 7.4491e-02,  5.3236e-02,  1.0788e-02],
              [ 4.6727e-03, -1.3132e-02, -5.1397e-03]],
    
             [[ 6.5068e-02, -9.5091e-03, -4.7880e-02],
              [-1.8116e-02,  5.0310e-02, -4.3630e-03],
              [ 3.4612e-03, -4.3647e-02,  1.3044e-02]],
    
             [[ 2.4180e-03,  2.5471e-02,  3.7343e-02],
              [-1.7611e-02, -5.6464e-02, -3.4999e-02],
              [-2.7549e-02, -5.7016e-03, -4.2026e-02]],
    
             ...,
    
             [[-7.9049e-03, -3.4917e-02, -5.0150e-04],
              [-6.1644e-02,  2.9234e-02,  2.4467e-02],
              [ 6.4167e-03,  2.9870e-02,  7.5125e-02]],
    
             [[ 7.6612e-02,  1.1932e-02, -1.4564e-02],
              [-4.4840e-02,  8.0319e-03,  4.2495e-02],
              [-4.8409e-02,  4.5992e-02,  2.3031e-02]],
    
             [[ 3.2587e-02, -5.6621e-02,  6.2170e-02],
              [-3.2940e-02, -1.6148e-02, -7.8749e-03],
              [ 1.5296e-02,  6.6066e-03,  2.1501e-02]]],


​    
            ...,


​    
            [[[ 4.3392e-02,  3.1892e-02, -6.0912e-02],
              [ 3.2236e-02, -6.1438e-02, -4.4012e-02],
              [-3.4353e-02,  6.7961e-02, -5.4611e-02]],
    
             [[ 1.8713e-02, -9.7891e-02, -5.6852e-02],
              [ 2.9484e-02, -4.0038e-02,  5.6397e-02],
              [ 2.2133e-02, -3.3515e-02,  3.2406e-02]],
    
             [[-2.7721e-02,  2.2127e-02,  2.9530e-02],
              [-2.6102e-02, -3.8631e-02,  6.8731e-02],
              [ 1.9735e-02,  2.3008e-02, -2.3933e-02]],
    
             ...,
    
             [[ 4.1398e-02,  2.2786e-02,  2.7265e-03],
              [ 1.0733e-02,  3.9280e-02, -2.9558e-03],
              [-5.1938e-02, -1.9259e-02,  4.2349e-02]],
    
             [[ 7.5985e-03, -9.4925e-02,  2.1317e-02],
              [-1.9697e-02,  3.9288e-02,  1.6268e-02],
              [-8.2106e-02, -5.6089e-03,  9.8829e-02]],
    
             [[ 2.0950e-03, -2.4346e-02,  3.8180e-02],
              [-4.8120e-03,  3.7703e-03,  3.2822e-02],
              [-2.1882e-02, -8.5669e-02, -5.5339e-02]]],


​    
            [[[-3.9782e-02, -2.8178e-02,  2.1350e-02],
              [-1.5101e-02, -6.2741e-02, -4.7504e-02],
              [ 1.9134e-02, -3.2309e-02,  3.7014e-02]],
    
             [[-4.6494e-02,  5.6103e-02,  1.2124e-03],
              [ 1.2678e-02, -2.2464e-02,  3.6343e-02],
              [ 1.7750e-02,  5.7882e-02, -3.4187e-02]],
    
             [[-4.0532e-02, -4.7067e-02, -2.5017e-02],
              [ 3.1092e-02, -2.5320e-02, -4.8343e-02],
              [-7.0592e-03,  6.9279e-02,  4.1107e-03]],
    
             ...,
    
             [[-5.4115e-03, -4.6132e-02, -4.2962e-02],
              [ 2.1316e-02, -2.9461e-02,  8.0669e-02],
              [ 7.3475e-03, -6.2416e-02,  5.8797e-02]],
    
             [[ 2.8009e-02,  9.4438e-02,  2.4128e-02],
              [-5.1240e-03,  3.9849e-02, -1.9139e-05],
              [ 9.9925e-03,  2.3025e-02, -3.0954e-02]],
    
             [[ 1.6193e-02, -5.7257e-02,  4.7540e-03],
              [-5.2892e-02, -2.7952e-02, -2.2088e-02],
              [-2.2044e-02, -5.4004e-02,  5.4337e-02]]],


​    
            [[[-2.6053e-02,  9.5196e-03, -2.1971e-02],
              [ 7.5675e-02, -5.6186e-02, -7.1327e-02],
              [-7.3842e-04, -2.4744e-02,  3.8442e-02]],
    
             [[ 2.0697e-03,  4.5354e-02,  6.5955e-02],
              [ 7.3361e-03,  1.9311e-02,  2.2453e-03],
              [ 1.1895e-02,  1.2448e-02, -1.5129e-02]],
    
             [[-1.0624e-02,  4.9166e-02,  3.1875e-02],
              [ 4.2217e-02,  1.3336e-02, -2.4965e-02],
              [-1.5078e-02, -4.1329e-02,  1.7680e-03]],
    
             ...,
    
             [[ 2.1686e-02, -8.3606e-03,  3.4883e-02],
              [-2.4252e-02, -8.9345e-03,  6.1014e-02],
              [-1.0333e-02, -2.7579e-02,  3.4201e-02]],
    
             [[ 1.1051e-01,  3.1364e-02, -4.1041e-02],
              [-1.1251e-02, -5.9290e-02,  3.4159e-02],
              [-7.5320e-03,  4.0232e-02, -4.2174e-02]],
    
             [[-3.2418e-03, -8.3922e-03,  8.1281e-02],
              [-6.7691e-02,  5.3527e-02, -1.5334e-02],
              [-2.7017e-02,  1.2073e-02,  3.9451e-02]]]])), ('layer2.1.bn1.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1.])), ('layer2.1.bn1.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer2.1.bn1.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer2.1.bn1.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1.])), ('layer2.1.bn1.num_batches_tracked', tensor(0)), ('layer2.1.conv2.weight', tensor([[[[ 0.0438,  0.0212,  0.0536],
              [-0.0553, -0.0061,  0.0488],
              [ 0.0429,  0.0411,  0.0124]],
    
             [[ 0.0867,  0.0072,  0.0142],
              [ 0.0055,  0.0552, -0.0237],
              [ 0.0047,  0.0041,  0.0014]],
    
             [[-0.0873, -0.1168,  0.0350],
              [ 0.0639, -0.0410, -0.0236],
              [ 0.0454,  0.0339,  0.0153]],
    
             ...,
    
             [[ 0.0595, -0.0314,  0.0183],
              [ 0.0088,  0.0639, -0.0579],
              [ 0.0012, -0.0317,  0.0295]],
    
             [[ 0.0010, -0.0198,  0.0331],
              [-0.1408,  0.0007,  0.0637],
              [-0.0242,  0.0030,  0.0096]],
    
             [[ 0.0049, -0.0033,  0.0685],
              [ 0.0282, -0.0911,  0.0314],
              [-0.0009, -0.0623,  0.0361]]],


​    
            [[[-0.0130,  0.0253, -0.0279],
              [ 0.0479, -0.0155,  0.0235],
              [ 0.0929, -0.0080,  0.0621]],
    
             [[ 0.0703,  0.0640, -0.0015],
              [ 0.0293, -0.0201,  0.0015],
              [-0.0222, -0.0073,  0.0475]],
    
             [[ 0.0537, -0.0159,  0.0414],
              [-0.0113, -0.0737,  0.0194],
              [-0.0251, -0.0452,  0.0056]],
    
             ...,
    
             [[ 0.0374,  0.0207, -0.0172],
              [-0.0302, -0.0282, -0.0555],
              [-0.0704,  0.0335,  0.0391]],
    
             [[-0.0483,  0.0278, -0.0649],
              [-0.0218,  0.0291,  0.0120],
              [-0.0715, -0.0882, -0.0135]],
    
             [[-0.0408,  0.0279, -0.0953],
              [-0.0277, -0.0323, -0.0265],
              [-0.0082,  0.0475,  0.0367]]],


​    
            [[[ 0.0643,  0.0171, -0.0050],
              [ 0.0072,  0.0043,  0.0748],
              [-0.0254, -0.1025, -0.0675]],
    
             [[ 0.0136, -0.0239,  0.0070],
              [-0.0154, -0.0906, -0.0549],
              [ 0.0133, -0.0315, -0.0086]],
    
             [[-0.0007,  0.0256,  0.0499],
              [ 0.0102, -0.0533,  0.0108],
              [ 0.0190, -0.0124, -0.0424]],
    
             ...,
    
             [[ 0.0334,  0.0582,  0.0360],
              [ 0.0600, -0.0246,  0.0014],
              [-0.0664, -0.0340, -0.0272]],
    
             [[ 0.0595,  0.0349, -0.0132],
              [ 0.0824, -0.0058,  0.0064],
              [-0.0066,  0.0201, -0.0285]],
    
             [[ 0.0537,  0.0192,  0.0188],
              [ 0.0184,  0.0452,  0.0640],
              [-0.0817,  0.0401, -0.0109]]],


​    
            ...,


​    
            [[[-0.0428, -0.0149, -0.0246],
              [ 0.0046,  0.0200, -0.0761],
              [-0.0081,  0.0070,  0.0307]],
    
             [[-0.0494,  0.0473,  0.0065],
              [-0.0317, -0.0046,  0.0469],
              [ 0.0110, -0.0626, -0.0298]],
    
             [[ 0.0476, -0.0788, -0.0107],
              [-0.0166,  0.0018, -0.0068],
              [ 0.0084,  0.0426,  0.0553]],
    
             ...,
    
             [[ 0.0197,  0.0296, -0.0125],
              [ 0.0059, -0.0097, -0.0440],
              [-0.0721,  0.0200,  0.1105]],
    
             [[ 0.0202,  0.0191,  0.0226],
              [-0.0082, -0.0265,  0.0410],
              [-0.0283,  0.0376, -0.0068]],
    
             [[ 0.0086,  0.0258, -0.0505],
              [ 0.0324, -0.0182, -0.0452],
              [ 0.0141, -0.0192, -0.0145]]],


​    
            [[[ 0.0459, -0.0163,  0.0096],
              [ 0.0127,  0.0464,  0.0216],
              [ 0.0046,  0.0333, -0.0478]],
    
             [[ 0.0362,  0.0332,  0.0251],
              [ 0.0559,  0.0016, -0.0122],
              [-0.0081,  0.0381,  0.0250]],
    
             [[ 0.0286, -0.0459,  0.0419],
              [ 0.0129, -0.0341, -0.0141],
              [ 0.0174, -0.0138, -0.0706]],
    
             ...,
    
             [[ 0.0469, -0.0872, -0.0281],
              [-0.0472, -0.0288, -0.0116],
              [-0.0058,  0.0350, -0.0293]],
    
             [[-0.0359,  0.0015, -0.0225],
              [ 0.0304,  0.0128,  0.0108],
              [ 0.0566, -0.0777,  0.0529]],
    
             [[ 0.0267,  0.0671, -0.0195],
              [-0.0036, -0.0272,  0.0465],
              [ 0.0214, -0.0128,  0.0035]]],


​    
            [[[-0.0081,  0.0398, -0.0227],
              [-0.0413,  0.0186, -0.0158],
              [ 0.0049, -0.0042, -0.0161]],
    
             [[-0.0488, -0.0421, -0.0690],
              [ 0.0391,  0.0251,  0.0164],
              [ 0.0043,  0.0059,  0.0018]],
    
             [[ 0.0076,  0.1163,  0.0076],
              [ 0.0276, -0.0786, -0.0247],
              [ 0.0491, -0.0236,  0.0099]],
    
             ...,
    
             [[-0.0124,  0.0348, -0.0156],
              [-0.0292, -0.0776, -0.0081],
              [ 0.0320, -0.0436,  0.0371]],
    
             [[ 0.0447,  0.0725,  0.0158],
              [-0.1048, -0.0343,  0.0236],
              [-0.0035, -0.0635,  0.0495]],
    
             [[-0.0558, -0.0184,  0.0068],
              [ 0.0774,  0.0130, -0.0256],
              [ 0.0515, -0.0177, -0.0094]]]])), ('layer2.1.bn2.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1.])), ('layer2.1.bn2.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer2.1.bn2.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer2.1.bn2.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1.])), ('layer2.1.bn2.num_batches_tracked', tensor(0)), ('layer3.0.conv1.weight', tensor([[[[ 0.0349,  0.0086,  0.0096],
              [ 0.0413, -0.0102,  0.0055],
              [-0.0163, -0.0677,  0.0016]],
    
             [[ 0.0014, -0.0110,  0.0038],
              [ 0.0337,  0.0020, -0.0030],
              [-0.0236,  0.0057, -0.0509]],
    
             [[ 0.0155,  0.0119,  0.0056],
              [-0.0105, -0.0323,  0.0536],
              [-0.0747, -0.0145, -0.0404]],
    
             ...,
    
             [[ 0.0295, -0.0132,  0.0087],
              [ 0.0296,  0.0195,  0.0187],
              [-0.0496,  0.0062, -0.0463]],
    
             [[-0.0356,  0.0047, -0.0013],
              [ 0.0156, -0.0075, -0.0235],
              [-0.0105, -0.0305,  0.0782]],
    
             [[-0.0231,  0.0091,  0.0305],
              [ 0.0142,  0.0132,  0.0348],
              [ 0.0044,  0.0219,  0.0029]]],


​    
            [[[-0.0212,  0.0364, -0.0290],
              [ 0.0127,  0.0041, -0.0074],
              [-0.0006,  0.0569, -0.0181]],
    
             [[-0.0113,  0.0053, -0.0675],
              [-0.0503, -0.0165, -0.0439],
              [-0.0322, -0.0382, -0.0123]],
    
             [[ 0.0327,  0.0066, -0.0186],
              [-0.0042, -0.0269, -0.0184],
              [-0.0141,  0.0079,  0.0137]],
    
             ...,
    
             [[-0.0125, -0.0250, -0.0081],
              [-0.0542,  0.0288,  0.0271],
              [-0.0183,  0.0235,  0.0012]],
    
             [[ 0.0596, -0.0349,  0.0526],
              [ 0.0047,  0.0208, -0.0436],
              [ 0.0365,  0.0079, -0.0054]],
    
             [[ 0.0479,  0.0087, -0.0030],
              [-0.0075,  0.0429, -0.0259],
              [-0.0032, -0.0156, -0.0009]]],


​    
            [[[-0.0249,  0.0367,  0.0297],
              [ 0.0061, -0.0402, -0.0070],
              [-0.0449, -0.0183, -0.0054]],
    
             [[ 0.0308,  0.0283, -0.0199],
              [ 0.0424, -0.0101,  0.0193],
              [ 0.0449,  0.0070,  0.0582]],
    
             [[-0.0426, -0.0077, -0.0369],
              [ 0.0001, -0.0265, -0.0589],
              [-0.0601, -0.0479, -0.0013]],
    
             ...,
    
             [[-0.0179, -0.0244, -0.0579],
              [-0.0459, -0.0029, -0.0151],
              [ 0.0263, -0.0004, -0.0187]],
    
             [[ 0.0074, -0.0004,  0.0086],
              [ 0.0284,  0.0654, -0.0165],
              [ 0.0116, -0.0059,  0.0304]],
    
             [[ 0.0535, -0.0324, -0.0140],
              [-0.0323,  0.0213,  0.0131],
              [-0.0326, -0.0430,  0.0530]]],


​    
            ...,


​    
            [[[ 0.0449, -0.0052, -0.0313],
              [-0.0396,  0.0049, -0.0056],
              [-0.0410, -0.0122, -0.0070]],
    
             [[ 0.0247,  0.0044, -0.0206],
              [ 0.0302, -0.0333,  0.0366],
              [ 0.0454,  0.0860, -0.0144]],
    
             [[-0.0065, -0.0059, -0.0134],
              [-0.0098,  0.0045, -0.0063],
              [ 0.0162,  0.0272,  0.0029]],
    
             ...,
    
             [[ 0.0081, -0.0118, -0.0031],
              [ 0.0490, -0.0305,  0.0092],
              [-0.0716, -0.0051,  0.0091]],
    
             [[-0.0138,  0.0322,  0.0029],
              [-0.0223,  0.0339,  0.0149],
              [ 0.0173, -0.0205, -0.0313]],
    
             [[-0.0080, -0.0018, -0.0041],
              [ 0.0237,  0.0120, -0.0249],
              [-0.0533, -0.0087,  0.0407]]],


​    
            [[[-0.0691, -0.0210,  0.0125],
              [ 0.0003,  0.0235, -0.0084],
              [ 0.0596, -0.0081, -0.0231]],
    
             [[ 0.0002,  0.0441,  0.0161],
              [ 0.0233, -0.0274,  0.0003],
              [-0.0047,  0.0399,  0.0414]],
    
             [[ 0.0064, -0.0306,  0.0459],
              [-0.0374,  0.0177, -0.0209],
              [-0.0426, -0.0197, -0.0247]],
    
             ...,
    
             [[-0.0193,  0.0245,  0.0153],
              [-0.0099, -0.0507, -0.0386],
              [ 0.0577, -0.0096,  0.0134]],
    
             [[-0.0476,  0.0122, -0.0419],
              [ 0.0218,  0.0256,  0.0191],
              [-0.0145, -0.0224, -0.0050]],
    
             [[ 0.0110,  0.0047, -0.0037],
              [ 0.0061,  0.0668,  0.0475],
              [ 0.0066, -0.0046, -0.0025]]],


​    
            [[[ 0.0083, -0.0538, -0.0203],
              [-0.0113, -0.0147, -0.0328],
              [ 0.0206, -0.0062, -0.0044]],
    
             [[ 0.0410,  0.0431,  0.0183],
              [-0.0123,  0.0068,  0.0033],
              [-0.0031,  0.0313, -0.0670]],
    
             [[ 0.0212, -0.0293, -0.0204],
              [ 0.0063,  0.0166, -0.0234],
              [-0.0406, -0.0198, -0.0353]],
    
             ...,
    
             [[-0.0170, -0.0230, -0.0056],
              [ 0.0055,  0.0298,  0.0040],
              [-0.0243, -0.0092,  0.0778]],
    
             [[-0.0132,  0.0287,  0.0016],
              [ 0.0520,  0.0449,  0.0065],
              [ 0.0165, -0.0415,  0.0275]],
    
             [[-0.0423, -0.0387, -0.0262],
              [ 0.0171, -0.0358, -0.0033],
              [ 0.0362,  0.0118,  0.0392]]]])), ('layer3.0.bn1.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1.])), ('layer3.0.bn1.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer3.0.bn1.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer3.0.bn1.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1.])), ('layer3.0.bn1.num_batches_tracked', tensor(0)), ('layer3.0.conv2.weight', tensor([[[[-0.0204,  0.0307, -0.0113],
              [ 0.0715, -0.0123,  0.0073],
              [ 0.0461, -0.0622, -0.0495]],
    
             [[-0.0041, -0.0623, -0.0655],
              [-0.0062, -0.0174,  0.0399],
              [-0.0096,  0.0002, -0.0127]],
    
             [[ 0.0214, -0.0274,  0.0407],
              [ 0.0415, -0.0221,  0.0415],
              [ 0.0362,  0.0206,  0.0486]],
    
             ...,
    
             [[-0.0109,  0.0002,  0.0343],
              [ 0.0070,  0.0327,  0.0177],
              [ 0.0101, -0.0220,  0.0310]],
    
             [[ 0.0088,  0.0096, -0.0066],
              [-0.0532, -0.0274, -0.0142],
              [ 0.0370, -0.0454, -0.0089]],
    
             [[-0.0248, -0.0109, -0.0664],
              [ 0.0129, -0.0072,  0.0324],
              [-0.0111,  0.0036,  0.0151]]],


​    
            [[[-0.0192, -0.0985, -0.0125],
              [-0.0136, -0.0548, -0.0440],
              [-0.0898, -0.0056, -0.0413]],
    
             [[ 0.0045,  0.0264,  0.0087],
              [ 0.0075, -0.0535, -0.0234],
              [ 0.0048,  0.0244, -0.0081]],
    
             [[ 0.0031,  0.0129, -0.0103],
              [ 0.0397,  0.0222,  0.0207],
              [-0.0562, -0.1118, -0.0240]],
    
             ...,
    
             [[ 0.0356, -0.0236,  0.0706],
              [ 0.0396,  0.0216, -0.0232],
              [-0.0299, -0.0489,  0.0286]],
    
             [[-0.0415, -0.0207, -0.0064],
              [-0.0407,  0.0791,  0.0062],
              [ 0.0288,  0.0222,  0.0014]],
    
             [[ 0.0111,  0.0380, -0.0231],
              [ 0.0161,  0.0108, -0.0158],
              [-0.0293,  0.0718, -0.0129]]],


​    
            [[[ 0.0088, -0.0482, -0.0320],
              [-0.0327,  0.0047, -0.0238],
              [ 0.0105,  0.0399,  0.0064]],
    
             [[ 0.0056, -0.0405, -0.0146],
              [ 0.0072, -0.0119,  0.0366],
              [ 0.0215,  0.0121, -0.0282]],
    
             [[-0.0020, -0.0566, -0.0365],
              [ 0.0665, -0.0455,  0.0041],
              [-0.0060, -0.0327,  0.0613]],
    
             ...,
    
             [[ 0.0061, -0.0231,  0.0126],
              [-0.0126,  0.0249, -0.0173],
              [ 0.0305, -0.0202, -0.0125]],
    
             [[ 0.0108,  0.0124, -0.0241],
              [-0.0519, -0.0344,  0.0101],
              [ 0.0030,  0.0403, -0.0448]],
    
             [[-0.0054, -0.0195, -0.0558],
              [-0.0163,  0.0378,  0.0286],
              [ 0.0061,  0.0207,  0.0359]]],


​    
            ...,


​    
            [[[ 0.0380, -0.0335, -0.0105],
              [ 0.0251,  0.0047,  0.0110],
              [ 0.0437,  0.0054,  0.0125]],
    
             [[ 0.0091,  0.0064, -0.0246],
              [-0.0438,  0.0140,  0.0633],
              [ 0.0193,  0.0032, -0.0254]],
    
             [[-0.0193,  0.0379,  0.0345],
              [ 0.0015,  0.0637,  0.0273],
              [ 0.0088,  0.0133,  0.0551]],
    
             ...,
    
             [[-0.0201,  0.0015, -0.0151],
              [ 0.0344, -0.0493, -0.0246],
              [-0.0080,  0.0391, -0.0078]],
    
             [[ 0.0100, -0.0149,  0.0163],
              [-0.0002,  0.0105,  0.0341],
              [-0.0005, -0.0172,  0.0095]],
    
             [[-0.0250, -0.0026,  0.0116],
              [ 0.0039, -0.0077, -0.0106],
              [-0.0030,  0.0147,  0.0239]]],


​    
            [[[-0.0267, -0.0428,  0.0060],
              [-0.0337,  0.0093,  0.0431],
              [-0.0431, -0.0147, -0.0194]],
    
             [[-0.0112, -0.0124, -0.0457],
              [ 0.0364,  0.0053, -0.0210],
              [ 0.0062, -0.0032, -0.0576]],
    
             [[ 0.0411, -0.0081,  0.0161],
              [ 0.0104, -0.0017,  0.0217],
              [ 0.0425, -0.0259, -0.0102]],
    
             ...,
    
             [[-0.0566,  0.0281,  0.0561],
              [ 0.0386, -0.0370,  0.0405],
              [ 0.0224,  0.0461,  0.0256]],
    
             [[ 0.0308,  0.0206, -0.0410],
              [-0.0365, -0.0139, -0.0191],
              [-0.0479,  0.0091,  0.0462]],
    
             [[ 0.0405,  0.0053, -0.0278],
              [ 0.0221, -0.0220, -0.0342],
              [-0.0027,  0.0194, -0.0425]]],


​    
            [[[ 0.0046,  0.0232, -0.0319],
              [-0.0342,  0.0621, -0.0501],
              [-0.0247, -0.0112,  0.0576]],
    
             [[-0.0053,  0.0199, -0.0020],
              [-0.0033,  0.0155, -0.0357],
              [ 0.0627,  0.0041,  0.0158]],
    
             [[-0.0559, -0.0015, -0.0111],
              [-0.0504, -0.0118,  0.0309],
              [-0.0191,  0.0127,  0.0020]],
    
             ...,
    
             [[-0.0005, -0.0383, -0.0425],
              [ 0.0177,  0.0012,  0.0175],
              [ 0.0022, -0.0020, -0.0347]],
    
             [[ 0.0148,  0.0054,  0.0406],
              [-0.0098,  0.0169, -0.0666],
              [-0.0345,  0.0198, -0.0046]],
    
             [[-0.0136, -0.0206,  0.0022],
              [-0.0020,  0.0172, -0.0251],
              [-0.0197,  0.0181, -0.0603]]]])), ('layer3.0.bn2.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1.])), ('layer3.0.bn2.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer3.0.bn2.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer3.0.bn2.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1.])), ('layer3.0.bn2.num_batches_tracked', tensor(0)), ('layer3.0.downsample.0.weight', tensor([[[[ 5.1302e-02]],
    
             [[-1.2627e-01]],
    
             [[-1.3083e-01]],
    
             ...,
    
             [[-1.6266e-02]],
    
             [[ 1.1629e-01]],
    
             [[ 1.3444e-01]]],


​    
            [[[-6.1614e-02]],
    
             [[-6.8547e-02]],
    
             [[ 5.8207e-02]],
    
             ...,
    
             [[ 1.1938e-02]],
    
             [[ 2.0041e-02]],
    
             [[ 2.1884e-04]]],


​    
            [[[ 8.9777e-03]],
    
             [[-4.8123e-02]],
    
             [[ 3.0489e-02]],
    
             ...,
    
             [[-4.8910e-02]],
    
             [[ 8.1343e-02]],
    
             [[-8.4297e-03]]],


​    
            ...,


​    
            [[[ 7.5705e-02]],
    
             [[ 1.9363e-01]],
    
             [[ 8.0216e-02]],
    
             ...,
    
             [[ 9.8609e-03]],
    
             [[-2.6596e-01]],
    
             [[-5.2704e-03]]],


​    
            [[[-1.1560e-01]],
    
             [[-1.1692e-01]],
    
             [[ 4.2977e-02]],
    
             ...,
    
             [[ 4.9820e-02]],
    
             [[-1.2323e-01]],
    
             [[ 1.6390e-01]]],


​    
            [[[ 4.4047e-02]],
    
             [[ 7.3217e-02]],
    
             [[ 2.5563e-01]],
    
             ...,
    
             [[-1.6249e-03]],
    
             [[-1.0374e-02]],
    
             [[-7.1804e-03]]]])), ('layer3.0.downsample.1.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1.])), ('layer3.0.downsample.1.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer3.0.downsample.1.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer3.0.downsample.1.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1.])), ('layer3.0.downsample.1.num_batches_tracked', tensor(0)), ('layer3.1.conv1.weight', tensor([[[[-3.4275e-03, -3.2765e-02, -4.8360e-02],
              [ 2.6047e-02,  2.4244e-02,  8.3237e-03],
              [ 8.2318e-03,  2.0020e-02, -2.9594e-04]],
    
             [[-1.2054e-03,  3.2403e-02,  2.0327e-02],
              [ 3.0869e-02,  2.3991e-02, -4.9800e-03],
              [-2.4696e-02, -2.1574e-02,  3.2120e-02]],
    
             [[ 2.4407e-03,  4.5500e-02,  1.7414e-02],
              [-1.5649e-02,  1.2575e-02,  2.1246e-02],
              [-2.9785e-02,  2.3647e-03, -4.6126e-03]],
    
             ...,
    
             [[-8.2805e-05, -3.3302e-02,  1.2111e-02],
              [-4.4140e-02,  2.9691e-02,  2.3848e-02],
              [ 1.4394e-02, -2.0125e-02, -2.3710e-02]],
    
             [[ 1.1411e-02, -2.1530e-02,  4.1833e-02],
              [-3.2720e-02, -2.6466e-03,  5.9094e-02],
              [ 1.2959e-02, -4.3469e-03, -1.8603e-02]],
    
             [[-4.6260e-03,  2.6415e-02,  4.3674e-02],
              [-5.0332e-02,  1.0870e-02,  2.7126e-02],
              [ 2.2240e-02,  5.7367e-02,  1.8207e-02]]],


​    
            [[[-2.3301e-02, -4.3126e-02, -2.3901e-02],
              [-1.3154e-02, -1.4424e-02, -1.7940e-02],
              [-3.6215e-02,  4.5187e-02, -5.8700e-03]],
    
             [[ 4.2955e-03,  4.3265e-02, -2.2313e-03],
              [ 7.0994e-02, -8.8121e-03,  1.3649e-02],
              [-1.0897e-02, -2.3306e-03,  5.2530e-02]],
    
             [[-1.2553e-02,  9.9124e-04, -7.9737e-02],
              [ 4.8922e-03,  3.4057e-02,  1.0713e-02],
              [ 7.8766e-02,  2.7916e-02, -3.0844e-02]],
    
             ...,
    
             [[ 1.7683e-02,  1.9431e-03,  1.0674e-02],
              [-2.0575e-02,  9.2218e-03, -2.6168e-03],
              [ 7.8172e-03, -4.8062e-02, -2.3314e-02]],
    
             [[ 3.6147e-02,  2.6981e-02, -2.2957e-04],
              [-1.5095e-02,  3.7287e-03,  1.0717e-03],
              [-1.3981e-02, -2.9080e-02, -9.6914e-03]],
    
             [[ 1.4152e-02,  4.3945e-02,  1.7273e-02],
              [-4.5323e-02,  4.5672e-03,  6.1765e-02],
              [-3.4651e-02,  1.6901e-02,  3.5999e-03]]],


​    
            [[[ 3.2745e-02, -2.1093e-02,  2.6111e-02],
              [ 3.9376e-02, -4.1046e-03, -6.1104e-03],
              [ 1.3050e-02,  6.6005e-02, -1.5202e-02]],
    
             [[-3.6875e-02,  3.7787e-02,  1.5600e-02],
              [-1.3169e-02, -3.4448e-03,  5.3856e-03],
              [-2.4479e-03,  1.6841e-02,  1.8229e-02]],
    
             [[-1.3233e-02, -3.7768e-02, -1.8962e-02],
              [-2.0258e-02, -9.4328e-03,  2.3798e-02],
              [-4.5409e-02,  8.9546e-03,  9.7676e-03]],
    
             ...,
    
             [[-3.8426e-02,  3.6415e-02, -2.1356e-02],
              [-6.9219e-02,  5.6381e-03, -1.0655e-02],
              [-5.3993e-02, -1.0081e-02,  1.2257e-02]],
    
             [[-6.5070e-02, -3.2924e-03,  5.2459e-02],
              [-2.5407e-02, -3.0754e-02,  5.7905e-03],
              [-3.2969e-02, -4.4555e-02, -1.8686e-02]],
    
             [[-1.5556e-02,  1.0232e-02, -1.9892e-02],
              [ 3.1916e-02, -5.5386e-02, -4.2912e-02],
              [ 3.9086e-02, -1.7610e-02,  3.8135e-02]]],


​    
            ...,


​    
            [[[ 8.9523e-03,  9.0386e-03,  3.7514e-04],
              [-2.3940e-03, -1.2129e-02,  1.1425e-02],
              [ 5.1709e-02,  1.4023e-02, -1.3509e-02]],
    
             [[-2.8031e-02,  2.5869e-02,  4.1954e-03],
              [-2.0250e-03,  8.6634e-03,  3.2324e-03],
              [ 6.4992e-02, -8.3147e-03, -4.1640e-03]],
    
             [[-1.9022e-02, -5.3721e-03, -4.4217e-02],
              [-2.2197e-02, -2.5634e-02, -8.3819e-03],
              [ 3.4498e-02, -3.6383e-02,  4.7910e-03]],
    
             ...,
    
             [[-1.2622e-02, -5.1117e-02, -6.9676e-03],
              [-3.2503e-02, -7.8702e-03, -2.7234e-02],
              [-1.7722e-02, -1.9462e-03, -3.1503e-02]],
    
             [[-1.8082e-02,  1.1581e-02, -2.7600e-03],
              [-4.6376e-02, -4.0566e-03,  5.9485e-02],
              [-4.0068e-02, -1.3325e-03,  3.5468e-02]],
    
             [[-2.7588e-02, -3.6860e-03,  2.1761e-02],
              [-1.0829e-02,  1.4175e-02,  9.4780e-03],
              [ 2.0903e-02,  1.3979e-02, -4.8911e-02]]],


​    
            [[[-4.0552e-02,  2.3846e-02,  4.9954e-02],
              [-1.7661e-02,  1.4004e-02, -2.9632e-02],
              [-3.1077e-02,  6.6514e-03,  2.4366e-02]],
    
             [[ 5.1884e-02,  3.1370e-02, -7.4215e-03],
              [-1.8851e-02,  2.6021e-03, -7.0751e-03],
              [-5.2249e-02, -1.9212e-02,  1.6598e-02]],
    
             [[ 2.6262e-02, -5.4647e-04,  2.4515e-02],
              [ 3.4015e-02,  1.3750e-02, -2.9688e-02],
              [ 2.9974e-02,  3.1654e-02,  2.4101e-02]],
    
             ...,
    
             [[ 8.5331e-03, -2.7333e-02,  2.1504e-02],
              [-2.8443e-02,  2.2886e-02,  5.2746e-02],
              [-3.3169e-02,  6.6165e-02,  1.9914e-02]],
    
             [[ 1.1873e-03, -2.0247e-03,  1.8708e-02],
              [ 1.6251e-02, -1.1317e-02, -2.6039e-02],
              [ 9.7906e-03,  2.3926e-02, -5.7490e-02]],
    
             [[ 6.5744e-02, -1.4836e-02, -3.7426e-02],
              [-8.7107e-03, -2.1662e-02,  6.9513e-03],
              [ 7.7147e-04,  2.2458e-02,  4.4468e-02]]],


​    
            [[[-3.1892e-02, -2.4033e-02,  2.3010e-03],
              [ 3.7565e-02,  1.0014e-02, -2.2165e-02],
              [-2.6159e-03,  2.6453e-02, -4.9073e-02]],
    
             [[ 2.9138e-02, -2.6143e-02, -1.7554e-02],
              [-2.6148e-03,  2.1903e-02, -5.5489e-03],
              [ 5.0457e-02,  2.1847e-02, -4.6261e-02]],
    
             [[-2.3481e-02, -1.0568e-03, -3.2478e-02],
              [-1.7053e-03,  2.4769e-02,  2.6193e-02],
              [ 1.6752e-02, -1.1499e-02,  4.0311e-02]],
    
             ...,
    
             [[-1.3489e-02,  5.3589e-02, -2.1646e-04],
              [ 2.0271e-02, -1.2201e-02,  1.7955e-02],
              [ 4.1370e-02,  2.8870e-02, -3.5185e-02]],
    
             [[-1.3736e-02,  8.2726e-03, -5.6179e-02],
              [ 2.3764e-02, -3.3681e-02,  2.2471e-02],
              [ 1.2665e-02,  3.0401e-02, -2.2962e-02]],
    
             [[-1.3585e-02, -5.9026e-03, -2.0017e-02],
              [ 1.4092e-02,  3.8301e-02, -2.9398e-02],
              [-5.5344e-03,  4.3024e-02,  1.2914e-02]]]])), ('layer3.1.bn1.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1.])), ('layer3.1.bn1.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer3.1.bn1.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer3.1.bn1.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1.])), ('layer3.1.bn1.num_batches_tracked', tensor(0)), ('layer3.1.conv2.weight', tensor([[[[ 3.0450e-02, -1.4764e-02,  1.2484e-02],
              [-3.1183e-02,  3.3724e-02,  1.9898e-02],
              [ 5.9339e-02, -1.7990e-03, -1.9772e-02]],
    
             [[-1.5687e-03, -4.2487e-02, -3.7112e-02],
              [-1.4221e-02, -2.5835e-02,  3.9170e-03],
              [ 6.1574e-03, -1.5033e-02, -1.6639e-02]],
    
             [[ 1.4094e-02,  1.0711e-02,  1.0824e-02],
              [ 2.0790e-02, -2.7543e-02,  1.6675e-02],
              [-1.1795e-02,  1.4660e-02, -1.7106e-02]],
    
             ...,
    
             [[-1.4501e-02,  2.8201e-02, -7.0925e-02],
              [ 3.4783e-02,  1.3036e-02,  1.5069e-02],
              [-4.9781e-02, -1.2876e-02, -5.8367e-02]],
    
             [[ 1.3604e-02,  5.0310e-03,  1.9656e-02],
              [ 1.2169e-02, -1.0567e-02, -3.1374e-02],
              [-3.7885e-02, -3.9021e-02, -4.5678e-04]],
    
             [[-7.7496e-02, -3.9379e-02,  3.9398e-02],
              [ 1.6972e-02,  5.6611e-02, -1.7317e-03],
              [ 3.0593e-02,  6.3763e-02, -2.7644e-03]]],


​    
            [[[-1.3291e-02, -1.5206e-02,  5.2298e-03],
              [ 1.1955e-02, -8.3960e-03, -2.1701e-02],
              [ 2.8219e-03,  3.8049e-02,  6.9573e-03]],
    
             [[ 1.7531e-02, -7.3248e-03,  2.9376e-02],
              [ 2.9584e-02,  5.8657e-02, -2.7732e-02],
              [ 3.5657e-02, -5.7662e-02,  3.0640e-02]],
    
             [[-2.1569e-02,  5.9803e-02,  3.7876e-02],
              [ 3.3871e-02,  4.0264e-02,  1.2637e-02],
              [ 5.3023e-02, -1.1335e-02,  1.7939e-02]],
    
             ...,
    
             [[-1.2867e-02, -6.5142e-03, -2.3125e-02],
              [-3.9135e-02,  1.6539e-02, -3.0539e-02],
              [ 2.1629e-02, -3.8552e-02,  1.1575e-02]],
    
             [[ 3.0371e-02,  5.6315e-03,  1.2514e-04],
              [-8.9490e-03, -5.3495e-02,  1.2492e-02],
              [-3.3766e-02,  6.2749e-02, -3.1363e-03]],
    
             [[ 6.9556e-03,  4.1174e-02,  1.4969e-02],
              [-1.3804e-02,  3.0142e-02,  7.5959e-03],
              [-6.8422e-03,  3.4523e-02, -3.5308e-02]]],


​    
            [[[-1.3718e-02,  2.6032e-02, -3.5351e-02],
              [ 1.2929e-02,  1.9278e-02,  2.6253e-02],
              [-4.4458e-03, -3.0676e-02,  6.2885e-03]],
    
             [[-2.3253e-02,  5.8394e-02, -2.7177e-04],
              [ 9.8116e-05, -3.4065e-02,  8.9029e-03],
              [ 9.4137e-03, -3.1040e-02,  5.1619e-04]],
    
             [[-4.7903e-02,  1.4733e-02,  3.7089e-02],
              [-5.0217e-03,  4.9756e-02, -1.6572e-02],
              [ 3.3901e-03, -6.9980e-03,  1.0569e-02]],
    
             ...,
    
             [[ 1.0835e-02,  1.4543e-02, -2.7965e-02],
              [-2.9713e-03, -5.1880e-02, -3.5625e-03],
              [ 3.2518e-03,  1.9563e-02, -6.8342e-03]],
    
             [[-1.7425e-02,  4.1145e-02, -1.6075e-02],
              [-1.2845e-03, -4.9576e-03,  6.3727e-03],
              [ 2.9496e-04,  1.0430e-02,  9.8068e-03]],
    
             [[ 3.5511e-02,  2.3129e-02,  2.8021e-02],
              [ 1.4639e-02,  4.2938e-03,  1.4175e-02],
              [-1.7044e-03, -3.6358e-02,  4.8874e-02]]],


​    
            ...,


​    
            [[[ 8.3247e-03,  5.5425e-02,  8.3526e-03],
              [-2.4693e-02,  9.9390e-04,  3.3968e-02],
              [-4.3386e-03,  8.9296e-04, -1.1349e-02]],
    
             [[ 5.2219e-03, -3.1748e-02,  7.4649e-04],
              [-7.1650e-03,  6.8017e-03,  7.7711e-02],
              [-2.1689e-02, -2.5007e-02,  5.9812e-02]],
    
             [[-2.8304e-02,  2.6397e-02,  2.8205e-02],
              [ 8.4211e-02,  1.1275e-02,  4.8635e-03],
              [ 1.1111e-02,  2.4489e-02, -2.2332e-03]],
    
             ...,
    
             [[-2.5757e-02,  5.5498e-03, -2.1972e-02],
              [-1.3406e-02, -2.0665e-02, -2.7517e-03],
              [-2.4359e-02,  2.7043e-03,  2.5349e-02]],
    
             [[ 5.1658e-03, -2.9786e-02,  1.2704e-02],
              [-1.8020e-02,  8.5598e-02,  6.6740e-04],
              [-3.1628e-03,  2.3645e-02, -6.4903e-02]],
    
             [[ 3.9627e-03, -5.2094e-03,  1.3886e-02],
              [-3.7860e-02,  1.8379e-02,  6.1846e-02],
              [ 8.3205e-03,  2.6255e-02,  3.3783e-02]]],


​    
            [[[-2.0964e-02,  2.5370e-02, -1.7020e-02],
              [ 1.0891e-02,  8.4425e-03,  1.8108e-02],
              [-2.9098e-02,  8.4492e-03, -1.9419e-02]],
    
             [[ 1.1922e-03, -5.1480e-02,  3.8803e-03],
              [-3.6808e-02, -1.7441e-02, -3.5299e-02],
              [-1.3415e-02,  1.5315e-02,  2.6672e-02]],
    
             [[ 9.0886e-03, -2.7243e-03,  3.2336e-02],
              [-5.1367e-03,  2.2698e-02, -2.7158e-02],
              [ 1.0612e-02,  9.1343e-04,  1.0016e-02]],
    
             ...,
    
             [[ 8.2980e-03,  2.2242e-02,  1.6844e-02],
              [-4.2482e-02, -2.4660e-02,  9.3187e-03],
              [-2.8374e-02,  1.1788e-02, -1.8709e-02]],
    
             [[-5.1907e-02,  4.9372e-02, -4.9451e-02],
              [ 1.4267e-02,  3.3285e-02, -2.5228e-02],
              [-6.8552e-03, -1.6252e-03,  8.3553e-03]],
    
             [[-1.4484e-03,  2.2049e-02,  2.0003e-02],
              [-4.5934e-02, -7.7408e-03,  4.2970e-02],
              [ 2.1615e-02,  2.7941e-02, -1.1171e-02]]],


​    
            [[[ 5.4081e-02,  3.3310e-03, -1.5288e-02],
              [ 2.5752e-02,  5.8497e-03, -1.9194e-02],
              [-7.3654e-03,  3.2929e-02,  3.5923e-02]],
    
             [[-4.3991e-02,  2.6179e-02,  3.3552e-02],
              [-3.3413e-02,  2.3890e-02, -2.4189e-02],
              [ 1.3159e-02, -2.3851e-02,  1.2658e-02]],
    
             [[ 3.1942e-02,  1.4981e-03,  2.0295e-03],
              [ 4.3999e-02,  4.4064e-02,  1.1983e-03],
              [-2.9893e-02,  3.2847e-02,  3.0525e-03]],
    
             ...,
    
             [[ 3.1411e-02,  3.0203e-02, -5.8562e-02],
              [ 3.5739e-02,  2.6084e-02, -6.9919e-02],
              [ 1.5704e-02, -2.5062e-02,  4.1598e-02]],
    
             [[ 2.2958e-03,  3.9184e-02, -1.4012e-02],
              [-2.3085e-02,  5.7279e-02,  2.8511e-02],
              [ 4.4926e-02, -2.0987e-02, -4.3386e-03]],
    
             [[ 5.2276e-02, -1.1884e-03,  1.3984e-02],
              [ 2.3814e-02, -6.6492e-02, -3.9726e-02],
              [-2.3462e-03,  6.1467e-02,  5.5971e-02]]]])), ('layer3.1.bn2.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1.])), ('layer3.1.bn2.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer3.1.bn2.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer3.1.bn2.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1.])), ('layer3.1.bn2.num_batches_tracked', tensor(0)), ('layer4.0.conv1.weight', tensor([[[[-3.4426e-02, -2.0650e-02, -7.9379e-03],
              [ 1.4907e-02,  9.9145e-03,  1.6605e-02],
              [ 4.1901e-02, -3.8256e-03,  4.6165e-02]],
    
             [[-1.5916e-02, -1.4827e-02, -1.6131e-02],
              [-2.8016e-02,  2.7889e-03, -7.1107e-03],
              [-1.0538e-02, -5.5127e-02,  2.8052e-02]],
    
             [[-1.2081e-02, -2.3949e-02, -2.0703e-02],
              [ 2.6517e-04, -1.4399e-02,  2.0012e-02],
              [-1.8030e-02, -2.1231e-03,  6.6343e-03]],
    
             ...,
    
             [[-1.7246e-02,  2.4135e-02,  4.2051e-03],
              [ 2.6897e-02, -1.6369e-04, -1.1309e-02],
              [-1.5673e-02, -8.3443e-03,  4.1823e-03]],
    
             [[ 4.3984e-02, -1.7815e-02, -1.4942e-02],
              [-5.1513e-02, -8.1108e-04, -1.5165e-02],
              [-3.6811e-02,  5.4820e-03, -2.3470e-02]],
    
             [[ 1.8343e-02,  1.3291e-02,  3.3124e-03],
              [ 1.5544e-02,  8.9084e-03, -2.0378e-04],
              [-2.5889e-02, -1.6304e-03,  1.8099e-02]]],


​    
            [[[-3.4747e-02, -9.6332e-04,  3.2263e-02],
              [-1.2929e-02,  3.1741e-03,  8.1934e-04],
              [ 1.9111e-02, -2.2590e-02, -3.7585e-02]],
    
             [[-1.3702e-02, -4.8879e-02,  9.8150e-03],
              [ 2.5271e-02,  1.1344e-02,  2.6328e-02],
              [-2.1827e-02,  2.6530e-02, -3.6134e-02]],
    
             [[ 6.9475e-03,  8.4351e-03,  1.2942e-02],
              [ 1.0882e-02, -1.3784e-02,  9.6637e-04],
              [ 3.6921e-02, -3.2078e-02,  2.7090e-03]],
    
             ...,
    
             [[-5.9892e-03, -1.2388e-02,  9.6632e-04],
              [ 1.9213e-04, -7.2601e-03, -7.1639e-03],
              [-9.3838e-03, -4.8896e-03, -3.3557e-02]],
    
             [[-4.0732e-02, -1.1723e-02, -2.5068e-02],
              [ 3.5916e-02, -2.2435e-02, -3.1604e-02],
              [ 3.6318e-02,  5.6179e-03, -1.8664e-02]],
    
             [[-2.7697e-02,  2.9838e-02,  7.5057e-03],
              [ 5.4007e-03,  1.8735e-02, -2.6802e-02],
              [-2.9978e-02,  3.4745e-02,  4.1988e-02]]],


​    
            [[[-1.0869e-03,  5.6779e-03,  2.2907e-02],
              [ 1.4753e-02,  2.7519e-02,  4.3633e-03],
              [-8.5497e-04, -2.6687e-02,  1.4616e-04]],
    
             [[-4.4301e-02, -1.5542e-02, -1.8199e-02],
              [-1.7307e-02, -2.4363e-02,  1.7238e-02],
              [ 1.8318e-02, -2.4690e-03, -3.2064e-02]],
    
             [[ 2.7114e-02, -3.1437e-03, -2.7026e-02],
              [ 5.1856e-02, -2.4643e-02,  3.7309e-02],
              [ 2.6793e-02, -1.8849e-02, -1.9014e-02]],
    
             ...,
    
             [[ 6.0458e-03,  2.8492e-02, -1.6163e-03],
              [-1.1290e-02,  1.5984e-02,  2.1983e-02],
              [ 1.5031e-02,  3.2189e-02, -2.7406e-02]],
    
             [[-5.8427e-03, -4.1422e-02, -1.9051e-02],
              [-6.2273e-03,  1.3993e-02, -9.1453e-03],
              [-1.4653e-02,  2.3670e-02,  2.3815e-02]],
    
             [[-2.4436e-03,  1.2416e-02, -2.6396e-02],
              [-2.7468e-02,  9.6938e-03,  8.4615e-04],
              [ 4.1347e-04,  7.8903e-03, -1.8545e-02]]],


​    
            ...,


​    
            [[[-2.0768e-02,  1.2019e-02, -2.8248e-02],
              [ 1.9572e-02,  2.1437e-02,  1.7574e-02],
              [ 1.0294e-02,  2.8026e-02,  2.3316e-02]],
    
             [[-3.8005e-02, -2.4392e-02, -4.8353e-04],
              [-2.5244e-02,  1.8302e-02,  2.2747e-02],
              [-8.1275e-03, -2.3235e-02,  2.9404e-03]],
    
             [[-6.3621e-03,  2.6156e-02,  3.4662e-03],
              [-1.3349e-02,  1.1119e-02,  8.4367e-03],
              [-1.6682e-02, -7.1614e-03,  2.2505e-02]],
    
             ...,
    
             [[ 1.1390e-02,  2.0807e-02, -9.0026e-03],
              [-7.5694e-03, -1.2274e-03,  6.5852e-03],
              [-9.6348e-03,  7.7879e-03, -3.0931e-02]],
    
             [[-1.5925e-02,  5.9851e-04,  2.2808e-03],
              [ 3.4779e-02, -3.0730e-03, -1.7820e-02],
              [ 4.7367e-03,  3.5228e-03,  1.0464e-02]],
    
             [[-1.5196e-02, -1.9263e-02,  3.0209e-02],
              [ 4.8933e-03, -1.1437e-02,  2.8073e-02],
              [-3.4256e-03,  8.6841e-06,  1.0678e-03]]],


​    
            [[[-3.0823e-03,  1.4280e-03, -6.2772e-03],
              [ 9.2778e-04,  5.1217e-03,  9.3867e-03],
              [-1.0111e-02, -2.2206e-02, -2.5241e-03]],
    
             [[-1.1326e-02,  5.2809e-03,  2.0117e-02],
              [-3.3499e-02,  3.8196e-02,  1.9622e-02],
              [ 4.1473e-03, -2.8321e-02, -7.3684e-03]],
    
             [[-3.1827e-03, -2.1147e-02, -7.2015e-03],
              [-1.2156e-02,  3.5554e-03, -1.0348e-02],
              [-1.7716e-02, -3.9093e-02,  1.4604e-02]],
    
             ...,
    
             [[-2.9968e-02,  1.0996e-02,  1.4550e-02],
              [ 3.8767e-02,  2.4672e-02,  1.1089e-03],
              [ 7.6358e-03, -4.3723e-03,  9.2216e-03]],
    
             [[ 3.4968e-02,  2.2976e-03,  7.0608e-03],
              [-1.1412e-02,  4.5701e-03,  1.3914e-03],
              [-3.1444e-02,  4.0768e-02,  6.1653e-03]],
    
             [[ 1.7533e-02, -3.0494e-02,  1.0099e-02],
              [ 3.1681e-02, -1.6135e-02, -2.6885e-02],
              [ 1.1824e-02,  2.4516e-02,  1.0581e-02]]],


​    
            [[[-3.2708e-04,  5.9305e-03, -1.2218e-02],
              [ 1.7925e-02,  3.7259e-02,  1.1810e-02],
              [ 3.8261e-03,  2.1109e-02, -1.2404e-02]],
    
             [[-2.2040e-02, -7.9393e-03,  3.8319e-02],
              [-2.2752e-02, -7.0236e-03, -2.0490e-02],
              [-1.4765e-02,  2.1094e-04, -5.3436e-03]],
    
             [[ 1.1482e-02, -1.5475e-02,  8.2683e-03],
              [-3.4944e-03,  5.6585e-04, -9.3152e-03],
              [ 1.1439e-02,  1.9590e-02,  6.4645e-03]],
    
             ...,
    
             [[ 3.8313e-02,  2.4285e-02, -1.1131e-02],
              [-1.0242e-04,  4.0969e-02, -2.9994e-02],
              [ 7.1082e-03,  2.2107e-02,  6.5453e-03]],
    
             [[-9.8455e-03,  7.9057e-04,  4.8128e-02],
              [ 2.9453e-02, -2.8000e-02, -8.5380e-03],
              [-2.4060e-02, -2.8138e-02, -4.4874e-02]],
    
             [[-3.5345e-02, -1.9844e-02, -9.3805e-04],
              [-1.2879e-02, -1.6884e-02, -1.4061e-02],
              [ 2.4718e-02,  1.6698e-02,  5.1346e-03]]]])), ('layer4.0.bn1.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1.])), ('layer4.0.bn1.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer4.0.bn1.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer4.0.bn1.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1.])), ('layer4.0.bn1.num_batches_tracked', tensor(0)), ('layer4.0.conv2.weight', tensor([[[[ 3.9987e-02,  1.6377e-02,  1.1537e-02],
              [ 1.5429e-04,  3.3687e-03,  2.6769e-02],
              [ 4.5581e-02, -2.2978e-02,  2.2034e-02]],
    
             [[-2.7881e-02,  6.1255e-03, -1.2566e-02],
              [-1.0602e-02,  2.0413e-02,  3.6084e-02],
              [ 3.5311e-02, -1.3656e-02, -1.1612e-02]],
    
             [[ 2.2937e-02,  1.9162e-02, -5.4952e-03],
              [ 1.1285e-02,  1.8485e-02,  1.6047e-02],
              [ 2.0785e-02, -2.7041e-02, -1.5259e-02]],
    
             ...,
    
             [[ 4.8956e-03,  2.0282e-02, -5.3931e-03],
              [ 1.0222e-02,  1.8082e-02, -1.3942e-02],
              [ 2.6756e-02,  1.8785e-02, -3.1818e-02]],
    
             [[ 9.0138e-03, -2.1343e-02,  1.3420e-02],
              [ 4.6611e-02,  2.0769e-02, -1.2019e-02],
              [ 3.8152e-03,  3.2009e-02,  1.7222e-03]],
    
             [[-1.4284e-02, -1.5476e-02,  3.0132e-02],
              [ 1.4093e-03,  5.1971e-03, -2.1814e-02],
              [-6.4795e-03, -2.5254e-03,  6.3007e-05]]],


​    
            [[[ 4.5274e-03,  1.6223e-02,  2.1589e-02],
              [ 2.3792e-02,  4.5018e-02,  3.7072e-02],
              [ 1.1174e-02, -1.0785e-02, -1.5236e-03]],
    
             [[-3.0813e-03, -3.1105e-02, -8.2161e-03],
              [ 1.1115e-02,  2.0689e-02,  8.1091e-03],
              [-3.2117e-02, -1.1782e-02, -1.5681e-02]],
    
             [[ 3.8722e-04,  2.2575e-02, -2.9844e-02],
              [-1.1150e-03, -2.1851e-02, -1.7332e-02],
              [ 9.3539e-03, -2.7910e-02,  7.0806e-03]],
    
             ...,
    
             [[-7.8759e-03, -4.6700e-02,  3.6113e-02],
              [ 2.9170e-02, -1.5087e-02,  1.4162e-02],
              [-4.6622e-02,  5.1858e-03, -1.1309e-02]],
    
             [[ 4.3175e-03, -2.8738e-02, -9.7116e-03],
              [-1.7032e-02, -5.0311e-03, -3.1874e-02],
              [ 4.5853e-03, -1.2478e-02,  2.6720e-02]],
    
             [[-1.6723e-02, -2.2472e-02, -2.0471e-03],
              [ 5.5419e-04,  2.6593e-02,  2.5199e-02],
              [-1.1873e-02, -4.3603e-03,  2.6322e-03]]],


​    
            [[[-1.3595e-02, -2.6590e-02,  2.3901e-02],
              [ 4.7180e-03,  1.3594e-02,  3.0595e-02],
              [ 4.7231e-02, -4.5193e-02, -4.3234e-02]],
    
             [[ 4.5242e-02,  3.8842e-02, -1.0839e-02],
              [ 3.1156e-02, -3.6172e-02, -2.8559e-02],
              [-2.2913e-02,  1.0383e-02, -5.8493e-03]],
    
             [[ 5.0987e-03, -1.1539e-02, -3.0080e-02],
              [-1.4494e-02,  3.7111e-02,  8.3581e-03],
              [-1.6871e-02,  2.6254e-02, -4.2414e-02]],
    
             ...,
    
             [[ 1.3091e-02,  2.3709e-02, -1.7082e-02],
              [-2.3198e-02,  1.1050e-02, -9.1059e-03],
              [-1.9516e-02, -2.6130e-02,  2.1280e-02]],
    
             [[ 1.1603e-02,  1.4547e-02,  3.1231e-02],
              [ 1.0280e-02,  1.3253e-02,  1.0121e-02],
              [-1.6605e-02, -2.2807e-02, -2.4404e-02]],
    
             [[ 6.8024e-03, -1.3941e-02, -1.4979e-02],
              [-1.7372e-02,  1.2247e-02,  9.5539e-04],
              [ 1.1951e-02, -1.0422e-02,  1.7355e-02]]],


​    
            ...,


​    
            [[[ 2.9022e-02,  2.6381e-02, -1.9288e-02],
              [-4.8321e-03,  2.9191e-02,  6.9125e-04],
              [ 1.5155e-02, -3.5044e-03,  1.2359e-04]],
    
             [[ 2.5525e-02, -1.5591e-02,  1.9321e-02],
              [-2.5520e-03,  1.1416e-02,  2.1690e-03],
              [-4.6676e-03, -2.0419e-02,  9.5332e-04]],
    
             [[-2.8312e-02, -9.8543e-04, -2.9239e-02],
              [-6.1804e-05, -3.0979e-02, -7.3537e-03],
              [ 8.8831e-04, -1.3398e-02,  3.4488e-03]],
    
             ...,
    
             [[ 2.4949e-02,  2.8155e-02,  2.3277e-02],
              [ 5.2676e-03, -2.2039e-02,  2.3924e-02],
              [ 3.0675e-02, -2.8452e-03,  1.4936e-02]],
    
             [[ 1.2647e-02,  1.7826e-02,  4.9843e-02],
              [ 2.5734e-02, -2.5533e-02,  9.8916e-03],
              [ 1.9557e-02,  3.8061e-03, -2.9102e-03]],
    
             [[-8.0495e-03, -1.5350e-02, -3.7474e-02],
              [-2.5358e-02, -4.7807e-03, -3.2121e-02],
              [ 1.5920e-02, -1.0264e-02,  1.8845e-02]]],


​    
            [[[-1.4936e-02, -3.2073e-02,  1.4138e-02],
              [ 8.4281e-03, -1.8128e-02, -2.9171e-02],
              [-3.1910e-02, -7.6171e-03, -1.8151e-02]],
    
             [[-2.4884e-03, -1.3325e-02, -4.8013e-02],
              [-4.5844e-03, -5.3668e-03,  2.5114e-02],
              [-8.1003e-03, -1.3105e-02, -2.9609e-02]],
    
             [[ 2.2515e-02,  3.3793e-03,  3.8654e-04],
              [-9.1275e-03,  6.9659e-03, -4.6730e-03],
              [-2.2623e-02,  2.3079e-02, -4.0595e-03]],
    
             ...,
    
             [[ 1.3057e-02,  5.5646e-03, -1.7202e-02],
              [ 7.1149e-03,  5.9121e-03, -1.4524e-02],
              [-1.0922e-02,  4.2458e-02, -1.8767e-03]],
    
             [[ 2.2265e-02, -3.3623e-02, -1.1373e-02],
              [ 1.0413e-02,  5.7937e-03, -3.0215e-03],
              [-1.7297e-02,  2.1106e-03,  3.6974e-02]],
    
             [[-1.6561e-02,  1.1785e-02,  3.7390e-02],
              [ 1.4489e-02,  1.1884e-02,  6.2340e-03],
              [-9.3383e-03,  1.7597e-02,  1.3923e-02]]],


​    
            [[[ 3.0113e-03, -1.0260e-02,  2.6663e-02],
              [ 6.3192e-03, -1.3597e-02, -2.4875e-02],
              [-1.7034e-02, -4.9622e-03, -2.4227e-02]],
    
             [[ 2.8849e-02,  1.7336e-02,  5.2462e-03],
              [-1.4524e-02,  1.4109e-02,  2.0550e-02],
              [ 4.3476e-03,  2.5029e-02, -1.2926e-02]],
    
             [[-3.0400e-02, -2.2211e-02,  2.8638e-02],
              [ 6.5878e-03, -2.2371e-02,  1.7473e-03],
              [-1.4824e-02, -5.1557e-03, -3.2254e-02]],
    
             ...,
    
             [[-2.6154e-02,  1.6258e-02, -1.9958e-02],
              [ 1.1572e-02, -1.5528e-02,  1.4332e-02],
              [-2.0849e-02,  1.9083e-02,  2.4401e-02]],
    
             [[ 1.2214e-02, -6.1253e-03, -2.1794e-02],
              [-6.6523e-03, -1.1222e-02, -4.7902e-04],
              [-3.7102e-02, -9.2181e-03,  1.0906e-02]],
    
             [[ 3.4334e-03,  8.6663e-03, -2.1325e-02],
              [ 5.3908e-02, -9.8666e-04, -2.3395e-03],
              [-6.9018e-04, -3.1469e-02, -1.6748e-02]]]])), ('layer4.0.bn2.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1.])), ('layer4.0.bn2.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer4.0.bn2.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer4.0.bn2.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1.])), ('layer4.0.bn2.num_batches_tracked', tensor(0)), ('layer4.0.downsample.0.weight', tensor([[[[-0.1354]],
    
             [[ 0.0602]],
    
             [[ 0.0289]],
    
             ...,
    
             [[ 0.0753]],
    
             [[ 0.0123]],
    
             [[-0.0188]]],


​    
            [[[ 0.0252]],
    
             [[ 0.0575]],
    
             [[ 0.0006]],
    
             ...,
    
             [[-0.1084]],
    
             [[ 0.0502]],
    
             [[-0.0238]]],


​    
            [[[-0.0509]],
    
             [[-0.0502]],
    
             [[ 0.0187]],
    
             ...,
    
             [[-0.1324]],
    
             [[-0.0758]],
    
             [[ 0.0340]]],


​    
            ...,


​    
            [[[-0.0054]],
    
             [[ 0.0149]],
    
             [[-0.0721]],
    
             ...,
    
             [[-0.0617]],
    
             [[-0.0071]],
    
             [[-0.1242]]],


​    
            [[[-0.0233]],
    
             [[ 0.0655]],
    
             [[-0.0293]],
    
             ...,
    
             [[ 0.0699]],
    
             [[-0.0081]],
    
             [[ 0.1189]]],


​    
            [[[-0.1025]],
    
             [[ 0.0254]],
    
             [[-0.0530]],
    
             ...,
    
             [[-0.0620]],
    
             [[ 0.0768]],
    
             [[-0.0705]]]])), ('layer4.0.downsample.1.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1.])), ('layer4.0.downsample.1.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer4.0.downsample.1.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer4.0.downsample.1.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1.])), ('layer4.0.downsample.1.num_batches_tracked', tensor(0)), ('layer4.1.conv1.weight', tensor([[[[-0.0325,  0.0232,  0.0008],
              [ 0.0086,  0.0065, -0.0044],
              [ 0.0280,  0.0054,  0.0270]],
    
             [[-0.0493,  0.0004,  0.0107],
              [-0.0045,  0.0231,  0.0146],
              [-0.0070, -0.0103,  0.0178]],
    
             [[-0.0034,  0.0108, -0.0015],
              [-0.0314,  0.0683,  0.0057],
              [ 0.0080, -0.0100, -0.0121]],
    
             ...,
    
             [[ 0.0033, -0.0216,  0.0006],
              [-0.0278, -0.0087, -0.0185],
              [ 0.0188,  0.0032, -0.0023]],
    
             [[-0.0338,  0.0129, -0.0048],
              [ 0.0084,  0.0305,  0.0142],
              [-0.0139,  0.0383,  0.0406]],
    
             [[-0.0686, -0.0325,  0.0332],
              [ 0.0020, -0.0054, -0.0112],
              [-0.0077, -0.0010,  0.0078]]],


​    
            [[[-0.0243,  0.0150,  0.0318],
              [ 0.0119, -0.0112,  0.0148],
              [ 0.0383,  0.0199,  0.0068]],
    
             [[ 0.0013, -0.0315, -0.0010],
              [-0.0203,  0.0103,  0.0094],
              [-0.0151, -0.0121, -0.0203]],
    
             [[-0.0341, -0.0234, -0.0026],
              [ 0.0350, -0.0224, -0.0248],
              [ 0.0159, -0.0219, -0.0127]],
    
             ...,
    
             [[ 0.0058,  0.0068, -0.0087],
              [-0.0037,  0.0230,  0.0233],
              [-0.0366, -0.0026, -0.0189]],
    
             [[-0.0141,  0.0222, -0.0415],
              [-0.0459,  0.0135,  0.0050],
              [ 0.0079,  0.0372, -0.0156]],
    
             [[-0.0328, -0.0193, -0.0409],
              [ 0.0119,  0.0201,  0.0032],
              [-0.0222, -0.0082,  0.0064]]],


​    
            [[[ 0.0242, -0.0264,  0.0295],
              [ 0.0264,  0.0230,  0.0105],
              [-0.0207, -0.0118, -0.0314]],
    
             [[-0.0118,  0.0110, -0.0088],
              [ 0.0255,  0.0467, -0.0211],
              [-0.0334,  0.0163, -0.0048]],
    
             [[ 0.0081,  0.0102, -0.0042],
              [-0.0248,  0.0082, -0.0155],
              [-0.0041,  0.0004, -0.0073]],
    
             ...,
    
             [[-0.0181, -0.0056,  0.0169],
              [ 0.0221,  0.0088,  0.0321],
              [ 0.0145, -0.0208,  0.0133]],
    
             [[-0.0042, -0.0308, -0.0267],
              [-0.0023, -0.0079,  0.0153],
              [-0.0004, -0.0119, -0.0183]],
    
             [[-0.0288, -0.0120,  0.0191],
              [-0.0142,  0.0351,  0.0207],
              [ 0.0019, -0.0065, -0.0266]]],


​    
            ...,


​    
            [[[ 0.0199,  0.0022,  0.0199],
              [-0.0171, -0.0496,  0.0146],
              [ 0.0214, -0.0213,  0.0159]],
    
             [[-0.0022,  0.0159,  0.0062],
              [-0.0030, -0.0032, -0.0108],
              [-0.0481, -0.0035, -0.0002]],
    
             [[-0.0147,  0.0247, -0.0213],
              [ 0.0371,  0.0003, -0.0025],
              [-0.0174, -0.0376, -0.0072]],
    
             ...,
    
             [[ 0.0167, -0.0006,  0.0055],
              [ 0.0026, -0.0105,  0.0006],
              [-0.0327, -0.0348,  0.0079]],
    
             [[-0.0455,  0.0031,  0.0042],
              [ 0.0278,  0.0231,  0.0283],
              [ 0.0002,  0.0230, -0.0246]],
    
             [[ 0.0163, -0.0081, -0.0103],
              [ 0.0099,  0.0322,  0.0340],
              [-0.0082, -0.0066, -0.0001]]],


​    
            [[[-0.0228,  0.0106,  0.0129],
              [-0.0187, -0.0123, -0.0146],
              [ 0.0074, -0.0158,  0.0317]],
    
             [[ 0.0152,  0.0024, -0.0054],
              [-0.0139,  0.0057, -0.0325],
              [ 0.0182, -0.0334, -0.0137]],
    
             [[-0.0183, -0.0541,  0.0039],
              [-0.0131, -0.0093,  0.0177],
              [ 0.0458,  0.0213,  0.0033]],
    
             ...,
    
             [[ 0.0030, -0.0052, -0.0268],
              [-0.0207, -0.0085, -0.0107],
              [ 0.0027,  0.0044,  0.0266]],
    
             [[-0.0208,  0.0075, -0.0117],
              [-0.0437, -0.0346, -0.0385],
              [ 0.0183, -0.0017,  0.0119]],
    
             [[-0.0279, -0.0053, -0.0061],
              [-0.0021,  0.0037,  0.0264],
              [-0.0009,  0.0113,  0.0025]]],


​    
            [[[-0.0166,  0.0277, -0.0019],
              [-0.0078,  0.0233,  0.0373],
              [ 0.0110, -0.0101,  0.0001]],
    
             [[ 0.0176,  0.0080,  0.0250],
              [-0.0071,  0.0051,  0.0149],
              [ 0.0005,  0.0050,  0.0129]],
    
             [[-0.0034, -0.0264, -0.0136],
              [-0.0303, -0.0102, -0.0298],
              [-0.0090,  0.0044, -0.0005]],
    
             ...,
    
             [[ 0.0032, -0.0098, -0.0129],
              [ 0.0101,  0.0107, -0.0183],
              [ 0.0147,  0.0256,  0.0420]],
    
             [[-0.0143,  0.0458, -0.0042],
              [ 0.0407, -0.0196, -0.0276],
              [-0.0008,  0.0726, -0.0162]],
    
             [[-0.0229, -0.0024,  0.0218],
              [ 0.0104,  0.0120, -0.0031],
              [-0.0102, -0.0265, -0.0226]]]])), ('layer4.1.bn1.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1.])), ('layer4.1.bn1.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer4.1.bn1.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer4.1.bn1.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1.])), ('layer4.1.bn1.num_batches_tracked', tensor(0)), ('layer4.1.conv2.weight', tensor([[[[-2.3662e-02, -2.3513e-02, -1.5573e-02],
              [-4.7517e-02,  5.9073e-03, -1.3614e-03],
              [ 6.0780e-03, -3.8839e-02, -1.7578e-02]],
    
             [[-2.1779e-02,  4.0308e-02,  6.1145e-03],
              [-5.9949e-03, -5.6621e-03,  1.4169e-02],
              [ 6.7464e-03,  1.3252e-02, -3.0208e-03]],
    
             [[ 4.5111e-02, -2.0537e-02,  1.5476e-02],
              [-5.1928e-03, -2.4852e-03, -2.5094e-02],
              [ 1.7014e-02,  3.2177e-02,  1.0081e-02]],
    
             ...,
    
             [[ 1.6567e-02,  2.2133e-05, -1.3088e-02],
              [-1.8307e-02, -1.1853e-02, -2.9698e-03],
              [-2.6929e-02,  2.2143e-02,  6.0593e-03]],
    
             [[ 3.9457e-02,  3.0668e-02, -1.2497e-02],
              [ 5.0543e-03, -1.1222e-02,  1.6106e-02],
              [ 1.5617e-02,  9.0761e-04, -1.2669e-02]],
    
             [[-9.1282e-03,  3.6710e-02,  4.3944e-03],
              [-8.8752e-03,  1.4686e-02,  1.0446e-02],
              [-1.8295e-02, -2.2993e-02, -2.3620e-02]]],


​    
            [[[ 1.6531e-03,  3.4567e-03, -1.9429e-02],
              [-4.6900e-03,  1.0452e-02, -2.5580e-02],
              [-1.4039e-03,  7.5230e-03, -3.8997e-03]],
    
             [[-1.0146e-02, -1.2023e-03, -6.3622e-03],
              [-5.0349e-03,  1.8959e-02, -2.4189e-02],
              [-1.9850e-02,  1.4023e-02, -1.9445e-02]],
    
             [[ 9.0402e-03, -3.0418e-02,  8.1724e-03],
              [ 3.7860e-02, -1.2944e-02,  9.7460e-03],
              [-1.5397e-02, -2.0535e-02,  2.0626e-02]],
    
             ...,
    
             [[-3.3120e-03, -1.3496e-02,  1.1123e-02],
              [-3.2833e-02, -4.7743e-04, -3.2559e-02],
              [ 4.8121e-03, -1.0991e-02,  1.1493e-04]],
    
             [[-2.0775e-02,  3.9096e-03, -3.9679e-03],
              [ 2.9163e-02,  1.8294e-02, -3.9337e-03],
              [-5.7133e-04,  2.3218e-03, -8.7641e-03]],
    
             [[ 3.8756e-03, -3.3386e-02,  4.3730e-02],
              [-3.0961e-02,  1.6407e-02,  1.0110e-02],
              [-1.6114e-02,  3.1494e-02,  5.9744e-03]]],


​    
            [[[-1.1329e-02,  3.3126e-03, -4.6241e-03],
              [-2.7044e-02,  6.5909e-03,  7.0866e-03],
              [ 5.2363e-02,  1.5017e-02, -1.7317e-02]],
    
             [[-1.3637e-02,  4.2175e-03,  9.4972e-03],
              [ 6.6826e-03, -3.8088e-03,  7.7308e-03],
              [-2.2740e-02, -1.5828e-02, -2.2909e-02]],
    
             [[-4.5026e-03,  4.6851e-03,  4.6103e-02],
              [ 1.2477e-02,  3.2316e-02,  1.1858e-02],
              [-1.6457e-02,  1.1231e-04,  2.8827e-02]],
    
             ...,
    
             [[ 3.1597e-02,  2.6750e-02, -9.7675e-03],
              [ 2.5121e-02,  1.2314e-02, -1.3105e-02],
              [ 1.2193e-02, -5.0444e-02, -3.3051e-02]],
    
             [[-2.4225e-02, -2.7506e-02, -8.8601e-03],
              [ 2.6399e-02,  2.7519e-02, -4.3771e-02],
              [-1.1470e-02, -4.0470e-03,  2.0637e-02]],
    
             [[-6.1738e-03, -3.3873e-02, -1.0252e-02],
              [ 2.0712e-02, -1.3556e-02,  1.4516e-02],
              [ 7.6622e-05,  1.0441e-02, -2.4969e-02]]],


​    
            ...,


​    
            [[[-1.2288e-03, -1.9742e-03, -1.8614e-02],
              [-6.3986e-03, -4.2116e-03,  2.3176e-02],
              [ 1.0561e-02, -1.6420e-02,  3.4507e-02]],
    
             [[-1.9764e-02,  1.0785e-02,  1.6745e-02],
              [-6.5887e-03,  1.5259e-03, -2.6146e-02],
              [ 2.1095e-02, -3.6375e-02, -2.7022e-03]],
    
             [[ 2.9852e-02, -2.4478e-02,  4.6899e-03],
              [ 1.6306e-02, -1.7728e-04, -3.1754e-04],
              [ 5.4857e-03,  3.3921e-02,  2.7303e-03]],
    
             ...,
    
             [[-2.8838e-02, -2.3566e-02, -1.5317e-02],
              [-1.8346e-02, -5.1785e-03,  2.4209e-02],
              [-2.9087e-02, -2.0016e-02,  1.8509e-02]],
    
             [[ 8.3490e-03,  1.7051e-02, -2.7071e-03],
              [-7.8314e-03,  6.2532e-03,  1.2643e-02],
              [-9.9559e-03, -4.6574e-02, -3.2087e-02]],
    
             [[-1.1705e-02,  1.8888e-02,  3.4632e-02],
              [ 5.1837e-02, -3.0410e-02, -1.1340e-02],
              [ 1.4330e-02, -1.1469e-02, -8.6637e-03]]],


​    
            [[[ 3.0953e-03, -2.9400e-02, -5.0638e-03],
              [-1.7371e-02, -5.2132e-03, -1.8046e-02],
              [ 6.5511e-03,  3.0614e-02, -3.1634e-03]],
    
             [[ 3.3873e-02, -1.3166e-02,  1.1041e-02],
              [-3.8080e-02,  2.4966e-02,  4.4413e-02],
              [-7.7000e-03,  2.9861e-02, -2.8791e-02]],
    
             [[ 6.4641e-03,  1.5620e-02,  1.2418e-02],
              [ 1.8234e-02,  9.1384e-03,  1.9138e-02],
              [-6.1269e-02,  6.9481e-03, -1.1410e-02]],
    
             ...,
    
             [[ 1.1664e-02,  4.8882e-03, -1.9306e-03],
              [ 2.6281e-02,  4.4745e-03,  2.5958e-03],
              [-3.7051e-02, -2.4621e-02,  1.3470e-02]],
    
             [[ 1.1898e-02,  3.2344e-02,  2.6440e-02],
              [-1.1959e-02, -4.2995e-02,  2.4245e-02],
              [-1.5575e-02, -1.0455e-02,  1.6821e-02]],
    
             [[ 7.6697e-03,  5.1326e-02,  1.4794e-02],
              [-1.4745e-02, -3.4990e-03, -4.9270e-02],
              [ 1.6760e-02,  3.9897e-03,  1.0378e-02]]],


​    
            [[[ 3.4175e-02,  2.4683e-04,  1.1436e-02],
              [ 1.7215e-02,  1.4469e-02, -1.3782e-02],
              [ 2.6343e-02, -1.7095e-03,  1.0328e-02]],
    
             [[ 1.5810e-02, -6.1180e-03,  2.4916e-03],
              [ 1.6777e-02, -1.3712e-02,  7.2718e-03],
              [ 1.4679e-02, -4.3712e-03,  2.2391e-02]],
    
             [[ 2.1680e-02, -7.1656e-03,  9.3273e-03],
              [ 3.4611e-03, -1.0776e-03, -7.4966e-03],
              [-1.9199e-02, -3.3206e-02,  7.8546e-03]],
    
             ...,
    
             [[-2.0247e-02, -9.2025e-03,  6.8522e-03],
              [ 7.8949e-03,  1.2932e-02, -8.9096e-03],
              [-2.7274e-02,  5.1701e-02,  4.2695e-03]],
    
             [[ 1.3120e-02, -1.8770e-02,  2.2096e-02],
              [ 4.8489e-02, -3.2794e-02, -1.3497e-02],
              [-1.7710e-02,  1.3417e-02,  7.2844e-03]],
    
             [[ 4.3833e-03, -4.3744e-04,  9.6720e-03],
              [ 1.1254e-02,  3.1494e-02,  7.3637e-03],
              [-2.7838e-02, -9.8798e-03, -4.8723e-04]]]])), ('layer4.1.bn2.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1.])), ('layer4.1.bn2.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer4.1.bn2.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer4.1.bn2.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1.])), ('layer4.1.bn2.num_batches_tracked', tensor(0)), ('fc.weight', tensor([[ 0.0363, -0.0408, -0.0379,  ...,  0.0182,  0.0142, -0.0282],
            [-0.0035,  0.0050, -0.0371,  ..., -0.0427,  0.0100, -0.0073],
            [-0.0090, -0.0298, -0.0032,  ..., -0.0061,  0.0058,  0.0414],
            ...,
            [ 0.0085, -0.0284,  0.0028,  ...,  0.0014, -0.0304, -0.0351],
            [ 0.0405, -0.0179, -0.0233,  ...,  0.0163,  0.0255, -0.0224],
            [ 0.0163, -0.0030, -0.0436,  ..., -0.0361, -0.0100,  0.0359]])), ('fc.bias', tensor([-3.3117e-02,  2.1381e-02,  4.5191e-04,  1.9420e-02,  3.1861e-02,
             3.8301e-02,  1.6528e-02, -1.1133e-03,  1.2325e-02,  1.7361e-02,
             2.5049e-02,  1.4298e-02,  2.9451e-02, -1.8848e-02, -1.4263e-02,
             2.3015e-02,  2.6016e-02, -1.1961e-02, -3.4514e-02, -1.1216e-02,
             1.1031e-02,  2.1724e-02,  9.8035e-05, -2.2028e-02, -1.6308e-02,
             3.0718e-02,  8.8301e-03, -3.4585e-02, -7.8474e-03,  1.5382e-02,
             2.6976e-02,  4.3321e-02, -4.2952e-02, -3.4341e-02,  2.0501e-02,
            -2.9527e-02,  1.3484e-02,  1.6248e-02, -2.2084e-02,  8.7246e-04,
            -2.9450e-02,  3.9827e-02,  1.1267e-02,  1.9803e-02, -2.3606e-02,
            -4.0999e-02, -2.5357e-02,  5.5554e-03, -3.4448e-02, -1.9038e-02,
             3.7538e-02,  7.3885e-04,  1.3478e-02,  1.5418e-03,  1.4757e-02,
            -3.7121e-02, -3.5226e-02,  2.0180e-02, -1.8588e-02,  4.7064e-03,
             2.2375e-02,  1.1967e-02,  1.2655e-02, -3.7507e-02,  1.7396e-02,
             2.9527e-02, -8.1724e-03, -3.5185e-02, -4.3861e-03, -3.8405e-03,
            -3.4795e-02,  7.9800e-03, -3.9885e-02, -3.4188e-02,  3.4676e-02,
            -3.0962e-02, -1.2788e-02, -2.7568e-02,  3.2951e-02, -1.6121e-02,
             2.9050e-02,  6.3742e-03, -3.2456e-02, -2.8273e-02, -7.6407e-03,
             2.2433e-02,  8.2769e-03, -2.9409e-02, -2.5392e-02,  3.3147e-04,
            -4.4186e-05, -4.3394e-02, -2.1682e-02, -5.7963e-03, -2.8437e-02,
            -1.5697e-02,  3.9307e-02, -4.0329e-02, -2.3179e-02,  2.4079e-02,
             2.4714e-02,  2.8919e-02, -4.1711e-02,  2.4463e-02,  3.3694e-02,
            -3.9821e-02,  2.2279e-02,  1.5718e-02, -3.6740e-02,  3.2902e-02,
            -3.3871e-02, -3.2231e-02,  8.6428e-03,  1.6780e-02,  4.1209e-02,
             2.6154e-02, -3.0900e-02, -1.9897e-02, -1.1001e-02, -3.8112e-02,
            -6.0327e-03, -2.9308e-02, -3.2270e-02, -2.9932e-02,  4.0400e-02,
             1.4672e-02, -3.4729e-02, -1.7627e-02,  1.4553e-02,  4.6136e-03,
            -1.4569e-02, -3.8026e-02, -2.1089e-02, -2.5645e-02, -3.5489e-02,
             4.4113e-02,  3.4505e-02, -9.0649e-03, -3.8817e-02, -2.2010e-02,
            -3.4405e-02, -1.2182e-03,  8.8558e-03,  2.9489e-02, -4.2498e-02,
             1.5451e-02,  1.1058e-02,  1.7396e-02, -1.2795e-02, -4.2771e-02,
            -3.1412e-02,  1.6232e-02, -1.6277e-02,  2.2794e-02, -3.7098e-02,
            -5.2723e-03, -2.1384e-02, -2.0074e-02,  9.3457e-03,  1.8284e-02,
             2.6006e-02, -2.6507e-04,  1.1843e-02, -2.1611e-02,  1.1691e-02,
             2.9002e-02,  3.2130e-02, -3.7340e-02,  2.5418e-03,  2.9189e-02,
            -1.6373e-02, -2.9543e-02, -3.2752e-02, -2.8742e-02, -2.8770e-02,
             3.9741e-02, -3.8369e-02, -5.1904e-03,  2.9692e-02,  5.7396e-03,
            -3.7989e-02, -3.0075e-02, -1.8731e-03, -3.8163e-02, -4.0778e-02,
            -7.3722e-03,  4.0741e-02, -2.1000e-03,  2.0497e-03, -2.4638e-02,
            -3.0484e-02, -4.1319e-02,  2.9949e-02, -4.0966e-02, -3.1474e-02,
             3.4290e-02,  1.1897e-02, -2.1155e-02, -2.9783e-02,  3.4406e-02,
             2.8104e-02,  3.4051e-02,  4.0026e-02, -3.0636e-02,  1.5356e-02,
             2.4776e-02, -3.3609e-02, -4.1504e-02, -2.6813e-03, -2.4690e-02,
             1.3166e-02, -3.7950e-02,  3.1911e-02, -1.6918e-02,  3.1129e-02,
            -3.4166e-02,  4.7001e-03, -4.4326e-03,  2.1315e-02, -3.9182e-02,
             3.3979e-03, -3.6935e-02,  9.7632e-03,  2.5121e-02, -2.7804e-02,
             9.4276e-03,  4.9146e-03,  3.8149e-02,  3.9888e-02,  6.3460e-03,
            -6.0112e-03,  2.4483e-02, -2.9934e-02,  2.3124e-02, -2.8490e-02,
            -8.6811e-03,  2.2993e-02,  4.3186e-03, -2.5025e-02, -4.2579e-02,
             2.7283e-02,  8.4686e-03,  1.7222e-02, -5.4135e-03,  1.3769e-02,
            -3.9609e-02,  2.0245e-02, -2.4285e-02, -5.5607e-05,  3.8660e-02,
             2.5464e-02,  2.2624e-02,  2.4475e-02, -3.2772e-02,  8.8715e-03,
            -1.9322e-03, -3.1793e-02, -3.9052e-02, -3.0444e-02, -3.7943e-02,
            -2.2011e-02, -3.9018e-03,  3.3250e-03,  2.7783e-03, -2.0854e-02,
            -2.6978e-02, -3.4884e-02,  4.1169e-02,  3.9958e-02, -3.9000e-02,
             2.5006e-02, -2.1077e-02, -1.3210e-02, -1.6881e-02,  2.3695e-02,
             1.7328e-03, -2.9827e-02, -6.1723e-03,  3.7020e-02,  2.1721e-02,
             2.8889e-03,  1.1058e-02, -8.7944e-03, -3.4843e-02,  2.9914e-02,
             8.9737e-03,  4.0784e-02, -3.7105e-02,  3.4704e-02, -3.5057e-02,
            -2.9317e-02, -1.7463e-02, -1.6207e-02, -4.1927e-03,  8.3389e-04,
             3.6975e-02, -3.5106e-02, -1.1557e-02,  1.0856e-02,  1.3658e-02,
             5.7960e-03,  2.5348e-02, -1.9471e-02, -4.0130e-02, -1.7840e-02,
             3.6776e-02,  1.0897e-02, -1.1202e-02, -2.6724e-02, -1.2862e-02,
             1.9320e-02, -2.8532e-02,  3.2379e-02, -2.2447e-02, -2.2893e-02,
            -5.2274e-04, -2.6122e-02,  1.4663e-02,  1.6062e-02, -2.1751e-02,
            -2.9038e-02, -2.8538e-02, -3.3143e-02, -4.1589e-02, -2.2380e-02,
             2.4770e-02,  2.9548e-02, -3.1688e-02, -1.4831e-02, -2.9779e-02,
             1.4684e-02, -6.2382e-03,  2.7696e-02, -3.3836e-02, -4.1328e-04,
            -2.1557e-02,  1.1007e-02, -4.3869e-03, -2.4330e-02, -2.5874e-02,
            -1.5663e-03,  3.3370e-02,  1.7552e-02,  1.0547e-02,  1.4226e-02,
            -9.3844e-05, -2.2725e-02, -3.8917e-02, -4.1002e-02, -4.2495e-02,
             3.0813e-02, -1.4693e-02,  1.4772e-02, -3.6432e-02,  1.1980e-02,
             3.2839e-02,  3.5414e-02,  3.3558e-03,  2.3640e-02,  3.8349e-02,
             4.3575e-02, -6.2014e-03,  5.3419e-03, -3.1280e-02,  3.6914e-02,
            -1.5123e-02,  3.3062e-02,  3.0890e-02, -3.9429e-03, -4.1753e-02,
             2.2143e-02, -1.9111e-02,  1.0017e-02, -2.5360e-02,  4.0554e-02,
             1.7179e-02, -2.9932e-02, -1.3201e-02, -1.4634e-02,  2.4067e-02,
            -1.6131e-02, -2.9521e-02,  3.8612e-02,  7.2837e-03,  4.3081e-04,
            -3.7409e-02, -1.1084e-02, -3.2180e-02,  4.7240e-04,  3.7482e-02,
             3.9070e-02, -5.7358e-04, -1.6119e-02,  1.5712e-02, -3.2270e-02,
             8.3880e-03, -1.9446e-02,  1.3668e-02,  1.1749e-02, -1.0522e-02,
             4.3861e-02,  3.0141e-02, -2.9129e-02, -2.4849e-02, -1.2251e-02,
             2.8610e-02, -2.3811e-02, -1.5605e-02,  3.3063e-02,  1.0504e-02,
            -7.7839e-03,  1.6718e-02,  1.2006e-02, -3.0025e-03, -3.3952e-02,
             1.5182e-02,  1.7610e-02, -1.2439e-02,  3.7906e-02,  4.3673e-02,
             1.6637e-02,  1.2565e-02,  4.1261e-02,  1.8202e-02, -3.0006e-02,
             3.7283e-03,  3.4605e-02,  2.7511e-03, -4.4229e-03,  2.7645e-02,
            -1.5111e-02,  2.8954e-02, -4.0425e-02, -1.0690e-02,  3.5759e-02,
             4.2533e-02,  1.3286e-02,  2.1400e-02,  3.2778e-02,  3.3202e-02,
            -1.9615e-02,  3.7081e-02,  4.0888e-02,  3.6719e-02, -2.3201e-02,
            -3.5237e-02,  3.9244e-02, -1.2778e-02,  3.8095e-02, -1.9599e-02,
            -3.6543e-02, -8.3564e-03,  7.5366e-03, -4.3592e-02,  8.1332e-03,
            -3.8283e-02, -1.7549e-02,  4.3591e-02,  1.6252e-02,  3.3563e-02,
            -2.4057e-02,  3.4805e-02,  2.7358e-02,  3.1244e-02, -6.4368e-03,
            -1.5385e-02,  2.9766e-03, -4.0208e-02, -2.9103e-02, -1.9667e-02,
             3.2045e-03, -3.5145e-02, -2.6877e-02,  2.4310e-02,  1.5016e-02,
            -3.7517e-02,  2.7698e-02, -3.5734e-04,  2.9763e-02,  7.0784e-03,
            -2.9078e-02, -2.3531e-02, -3.9588e-02,  1.7974e-02, -4.0546e-02,
             3.5108e-02,  1.3049e-03,  4.0017e-02,  2.1876e-02, -1.4840e-02,
             2.0077e-03,  1.2187e-03, -1.0499e-02, -1.9889e-02,  1.1983e-02,
            -3.3845e-02, -3.1640e-02, -3.9850e-02, -3.6663e-02,  1.2043e-03,
             2.4583e-02, -3.0066e-02, -3.8949e-02, -4.6746e-03, -9.6048e-03,
            -2.0147e-02, -1.3608e-02,  1.0943e-02, -2.6662e-04,  4.1414e-02,
             4.1538e-02,  4.0596e-03, -4.2921e-02, -3.8691e-02,  1.3438e-02,
             2.0696e-02,  3.8188e-03, -3.8685e-02,  3.6010e-04,  2.3068e-03,
             1.8161e-02,  4.0909e-02,  2.5874e-02, -3.9502e-02,  8.9886e-03,
            -1.7827e-02,  2.7839e-02,  1.3329e-02,  4.1513e-02, -4.2627e-02,
            -3.7625e-02, -3.0720e-02, -3.4754e-02,  1.4254e-02, -2.3538e-02,
             3.9863e-03, -2.3081e-02, -3.1748e-02,  1.2053e-02,  6.2845e-03,
            -4.3717e-02,  3.8105e-02, -5.7647e-03, -3.0263e-02,  4.1685e-02,
            -1.1686e-02,  1.6059e-02,  1.9182e-02,  4.0167e-03,  4.0292e-03,
            -5.0742e-03,  1.4587e-02,  1.1656e-02, -2.6953e-02,  1.8458e-02,
            -1.4548e-02,  1.8667e-02,  4.3167e-03, -1.8348e-02,  2.7135e-02,
             1.2274e-03, -4.2907e-02, -2.1928e-03, -3.2981e-03, -1.0747e-02,
            -2.2316e-02,  2.9696e-02,  2.1058e-02,  1.8506e-02, -5.3214e-03,
            -7.3174e-03, -3.6476e-02, -3.5608e-02,  2.4610e-02, -1.6470e-02,
            -1.1781e-02,  1.2980e-02, -2.4338e-02,  3.6172e-02,  3.9334e-02,
            -1.2570e-02, -4.3619e-02,  4.2439e-02,  1.3377e-02, -4.3007e-02,
             2.5111e-02, -1.0785e-02, -2.0383e-02,  1.0891e-02,  7.6394e-03,
             3.2379e-02,  1.0264e-02, -3.7485e-02,  4.2377e-02,  4.4162e-02,
            -2.0059e-02, -1.6613e-02, -2.0599e-02,  1.3724e-02,  1.5848e-02,
            -6.4984e-03, -2.8599e-02, -3.9005e-02, -2.5267e-02, -3.7383e-02,
            -3.7925e-02, -3.0174e-02,  2.5201e-02,  2.7934e-03, -2.1073e-02,
            -1.2213e-02,  3.9711e-02,  2.8561e-02, -2.6600e-02,  3.6301e-03,
            -1.0460e-02, -2.0751e-02, -2.4667e-02,  1.0336e-02, -3.8812e-02,
            -1.6428e-02,  2.7372e-02, -3.0376e-03, -1.0425e-02, -3.6220e-02,
             1.9789e-02, -1.9636e-03,  1.1677e-02, -6.8549e-03, -2.9949e-02,
            -2.1845e-02,  3.3612e-02,  1.4938e-02,  3.7081e-02, -1.9472e-02,
             2.8941e-02,  7.4000e-03, -1.3529e-02, -3.1744e-02, -3.4879e-02,
            -3.7831e-02, -2.3466e-02,  2.0166e-03, -2.0493e-02,  1.9948e-02,
            -3.3814e-02,  2.6432e-02,  3.5718e-02,  3.7295e-02, -1.1016e-02,
             3.0097e-02, -2.7377e-02,  2.8207e-02,  2.2716e-03,  4.2445e-02,
             2.4413e-04, -3.8713e-02,  2.4490e-02, -4.3551e-02,  4.1845e-02,
            -3.4120e-02,  3.1620e-02,  1.8219e-02,  8.5968e-03, -3.6503e-02,
             3.1027e-02, -1.5036e-02, -3.2103e-02,  2.6419e-02,  2.5318e-02,
            -3.0718e-02, -3.0912e-02,  3.1243e-02,  4.1520e-02, -1.0320e-02,
             3.3256e-02, -2.8246e-02, -1.0273e-02, -2.2718e-02,  3.7776e-02,
            -5.7787e-03,  3.4068e-02, -3.2193e-03,  3.2025e-03,  1.0997e-02,
             3.0308e-02,  2.6697e-02,  2.3445e-02,  1.3668e-02,  5.8259e-03,
             1.4710e-02, -1.4076e-02, -9.1691e-03, -2.1146e-02, -1.4024e-02,
             3.6937e-02,  3.3068e-02, -1.0702e-03, -3.4627e-02,  1.3308e-02,
            -9.1929e-03, -1.2564e-02,  5.6581e-03, -3.6414e-02,  3.8871e-02,
             1.1841e-02,  4.2039e-02,  4.3033e-02, -4.2008e-03, -3.0057e-02,
            -4.0631e-02, -1.6801e-02,  4.4721e-04, -1.9978e-02, -2.9911e-02,
            -3.7106e-02,  2.5949e-02,  6.6971e-03,  2.8034e-02, -9.3827e-03,
             3.8626e-02, -3.9483e-02,  3.4315e-02, -1.2284e-02,  4.4942e-03,
            -2.8916e-02, -3.7983e-02, -1.4357e-02, -3.6765e-02, -4.3759e-02,
            -7.6536e-03,  1.7625e-02,  3.2218e-02,  2.9999e-02,  4.3099e-02,
            -1.0266e-03, -3.3792e-02,  6.3643e-03,  3.9727e-02,  2.2211e-02,
            -1.3953e-02, -9.1497e-03, -2.3724e-02,  2.7131e-02, -1.6413e-03,
            -4.1132e-02,  3.4204e-02,  2.1569e-02,  5.0671e-03,  3.7351e-02,
             4.3558e-02,  2.0884e-02,  2.9375e-02,  4.1709e-02,  2.8462e-02,
            -8.8910e-04,  7.5950e-03, -2.8751e-02,  2.7188e-02, -1.1303e-02,
            -8.4523e-03,  1.5901e-02,  5.8650e-03, -3.0121e-02,  1.5643e-02,
            -4.3617e-02,  3.8544e-02, -1.3150e-02, -2.6992e-02, -3.9682e-03,
             3.0088e-02,  3.4921e-02,  3.4194e-02, -9.4354e-03,  3.8235e-02,
            -2.7295e-02,  8.2966e-03,  3.0110e-02,  1.6849e-03,  1.5671e-02,
            -2.0562e-02, -2.0207e-02, -3.5759e-02,  3.4332e-03,  1.2668e-02,
             4.0891e-02, -1.1468e-02, -1.2866e-02, -3.1233e-02, -4.3509e-02,
             2.5003e-02,  5.0937e-03, -1.6270e-02, -7.7638e-03, -2.7409e-02,
             3.3274e-02,  4.1334e-02,  4.3927e-02, -3.7319e-02,  2.0777e-02,
             1.4294e-02, -4.2854e-03,  1.4250e-02, -4.0259e-02,  3.2913e-02,
             2.3235e-02, -4.0119e-02,  3.8079e-02, -2.0954e-02,  9.4767e-03,
            -2.7012e-02, -6.8499e-03, -5.9213e-03,  2.9294e-02,  8.0973e-03,
            -3.5484e-02, -3.8084e-03, -6.2772e-03,  1.9013e-02, -3.4680e-03,
            -3.3491e-02,  2.4230e-02, -3.8064e-02,  3.1807e-02,  3.2885e-02,
            -1.4008e-02, -2.2544e-02, -1.3461e-02, -5.2985e-04,  3.4565e-02,
             2.2960e-02, -1.5113e-02,  1.2364e-02,  2.9868e-02, -8.8774e-03,
            -2.9295e-02,  2.8589e-02, -1.1447e-02, -4.4494e-03,  3.6675e-02,
            -4.0005e-02,  4.0165e-02, -1.8027e-02, -2.6828e-02,  3.4661e-02,
            -4.0563e-02,  1.6194e-02,  4.0141e-02,  3.8339e-02, -3.7378e-02,
            -3.5936e-02, -1.2882e-02,  2.0961e-02, -4.1095e-02, -1.4003e-02,
             1.5604e-03,  2.6667e-03,  2.2384e-03, -1.5536e-02, -6.7732e-04,
             3.6904e-02, -3.4456e-02, -4.0145e-02,  3.2038e-02, -3.1337e-02,
            -2.7019e-03, -2.1952e-02, -3.2226e-02, -3.0426e-02, -2.2712e-02,
             2.1189e-02,  3.0328e-02, -3.2485e-02,  3.7748e-02,  1.1909e-03,
             2.8606e-02, -1.8905e-02, -2.3313e-02,  8.7787e-03, -5.1719e-04,
             2.7339e-03,  2.9143e-02, -3.1501e-02, -8.9803e-03, -4.0308e-02,
             1.6124e-02, -4.1005e-03,  2.4144e-02,  1.1426e-02,  3.6999e-02,
             1.7658e-02, -1.8783e-03,  9.3975e-04, -3.6401e-02, -2.6279e-02,
             2.0788e-02,  2.5431e-03, -2.3389e-02,  2.8518e-02,  2.2169e-02,
             2.7188e-03,  2.5363e-02,  1.5379e-02, -1.8507e-04, -1.0163e-02,
            -5.8487e-03, -2.7887e-02, -2.1268e-02, -4.3728e-02, -1.8425e-02,
             3.6177e-02, -2.5185e-02, -2.7398e-02, -4.0522e-02, -3.2207e-02,
             3.8161e-02, -4.1884e-02,  1.4452e-02, -2.9603e-02,  1.8466e-02,
            -8.1972e-03, -3.5820e-02,  1.9415e-02,  1.5580e-02, -2.1561e-02,
             4.2275e-02, -2.5455e-02, -1.8684e-02,  1.9697e-02,  1.0054e-02,
             1.7712e-02, -3.9250e-02, -2.6507e-04, -2.2316e-02, -6.9716e-03,
             6.9062e-03,  3.9247e-02,  4.6194e-04, -1.9644e-02, -8.0433e-03,
             4.0123e-02,  4.0256e-02,  3.1510e-02, -2.2922e-02, -2.1645e-02,
            -1.2244e-02,  2.9483e-02,  1.2072e-03,  3.2402e-02,  4.0311e-02,
            -3.2406e-03,  4.1794e-02, -4.3893e-02,  1.2676e-02, -1.0208e-02,
             2.8095e-02, -4.2569e-02, -7.5497e-03,  2.3483e-02, -2.5475e-02,
             2.5934e-02, -1.0603e-02,  4.0301e-02,  3.3295e-02, -5.6134e-03,
            -2.9092e-02, -2.0971e-02, -3.8250e-02, -1.3003e-02,  2.5778e-02,
             1.7962e-02, -4.2304e-03,  4.4162e-02,  2.9964e-02,  3.2598e-03,
            -2.2691e-02, -1.2638e-02,  2.3293e-02,  1.7401e-02, -3.6645e-02,
            -1.2597e-02, -3.5031e-02,  4.3186e-02,  6.0022e-03, -2.4539e-02,
             2.1690e-02,  1.4755e-02,  1.6306e-02,  3.9170e-02,  7.2215e-03,
             3.9204e-02,  2.5716e-02,  3.5606e-02,  2.0549e-02, -1.3898e-02,
            -1.3565e-02,  1.7732e-02, -4.3100e-02, -2.8189e-03,  2.8215e-02]))])



```python
model2 = torchvision.models.resnet18()
model2.fc = nn.Linear(512,10)
```

Now, we want to load model1's parameters to model2, however, they have different structure, we can't simply use **model2.load()** or **model2.load_state_dict()**. we should load part of the parameters of the model1 to model2. For examlpe, if we want to load the convolution layers parameters, we can do like below.


```python
'''
    parameters are saved as dict in .pt file, so, we should get its keys first.
'''
print(model2.state_dict().keys())
```

    odict_keys(['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var', 'bn1.num_batches_tracked', 'layer1.0.conv1.weight', 'layer1.0.bn1.weight', 'layer1.0.bn1.bias', 'layer1.0.bn1.running_mean', 'layer1.0.bn1.running_var', 'layer1.0.bn1.num_batches_tracked', 'layer1.0.conv2.weight', 'layer1.0.bn2.weight', 'layer1.0.bn2.bias', 'layer1.0.bn2.running_mean', 'layer1.0.bn2.running_var', 'layer1.0.bn2.num_batches_tracked', 'layer1.1.conv1.weight', 'layer1.1.bn1.weight', 'layer1.1.bn1.bias', 'layer1.1.bn1.running_mean', 'layer1.1.bn1.running_var', 'layer1.1.bn1.num_batches_tracked', 'layer1.1.conv2.weight', 'layer1.1.bn2.weight', 'layer1.1.bn2.bias', 'layer1.1.bn2.running_mean', 'layer1.1.bn2.running_var', 'layer1.1.bn2.num_batches_tracked', 'layer2.0.conv1.weight', 'layer2.0.bn1.weight', 'layer2.0.bn1.bias', 'layer2.0.bn1.running_mean', 'layer2.0.bn1.running_var', 'layer2.0.bn1.num_batches_tracked', 'layer2.0.conv2.weight', 'layer2.0.bn2.weight', 'layer2.0.bn2.bias', 'layer2.0.bn2.running_mean', 'layer2.0.bn2.running_var', 'layer2.0.bn2.num_batches_tracked', 'layer2.0.downsample.0.weight', 'layer2.0.downsample.1.weight', 'layer2.0.downsample.1.bias', 'layer2.0.downsample.1.running_mean', 'layer2.0.downsample.1.running_var', 'layer2.0.downsample.1.num_batches_tracked', 'layer2.1.conv1.weight', 'layer2.1.bn1.weight', 'layer2.1.bn1.bias', 'layer2.1.bn1.running_mean', 'layer2.1.bn1.running_var', 'layer2.1.bn1.num_batches_tracked', 'layer2.1.conv2.weight', 'layer2.1.bn2.weight', 'layer2.1.bn2.bias', 'layer2.1.bn2.running_mean', 'layer2.1.bn2.running_var', 'layer2.1.bn2.num_batches_tracked', 'layer3.0.conv1.weight', 'layer3.0.bn1.weight', 'layer3.0.bn1.bias', 'layer3.0.bn1.running_mean', 'layer3.0.bn1.running_var', 'layer3.0.bn1.num_batches_tracked', 'layer3.0.conv2.weight', 'layer3.0.bn2.weight', 'layer3.0.bn2.bias', 'layer3.0.bn2.running_mean', 'layer3.0.bn2.running_var', 'layer3.0.bn2.num_batches_tracked', 'layer3.0.downsample.0.weight', 'layer3.0.downsample.1.weight', 'layer3.0.downsample.1.bias', 'layer3.0.downsample.1.running_mean', 'layer3.0.downsample.1.running_var', 'layer3.0.downsample.1.num_batches_tracked', 'layer3.1.conv1.weight', 'layer3.1.bn1.weight', 'layer3.1.bn1.bias', 'layer3.1.bn1.running_mean', 'layer3.1.bn1.running_var', 'layer3.1.bn1.num_batches_tracked', 'layer3.1.conv2.weight', 'layer3.1.bn2.weight', 'layer3.1.bn2.bias', 'layer3.1.bn2.running_mean', 'layer3.1.bn2.running_var', 'layer3.1.bn2.num_batches_tracked', 'layer4.0.conv1.weight', 'layer4.0.bn1.weight', 'layer4.0.bn1.bias', 'layer4.0.bn1.running_mean', 'layer4.0.bn1.running_var', 'layer4.0.bn1.num_batches_tracked', 'layer4.0.conv2.weight', 'layer4.0.bn2.weight', 'layer4.0.bn2.bias', 'layer4.0.bn2.running_mean', 'layer4.0.bn2.running_var', 'layer4.0.bn2.num_batches_tracked', 'layer4.0.downsample.0.weight', 'layer4.0.downsample.1.weight', 'layer4.0.downsample.1.bias', 'layer4.0.downsample.1.running_mean', 'layer4.0.downsample.1.running_var', 'layer4.0.downsample.1.num_batches_tracked', 'layer4.1.conv1.weight', 'layer4.1.bn1.weight', 'layer4.1.bn1.bias', 'layer4.1.bn1.running_mean', 'layer4.1.bn1.running_var', 'layer4.1.bn1.num_batches_tracked', 'layer4.1.conv2.weight', 'layer4.1.bn2.weight', 'layer4.1.bn2.bias', 'layer4.1.bn2.running_mean', 'layer4.1.bn2.running_var', 'layer4.1.bn2.num_batches_tracked', 'fc.weight', 'fc.bias'])


We can see the last layer of convolution is **'layer4.1.bn2.num_batches_tracked'**. Now, let's load it to our new model.


```python
pretrained_params = torch.load('./params/resnet18_params.pt')
model2_params = model2.state_dict()
for (pretrained_key,pretrained_val), (model2_key,model2_val) in zip(list(pretrained_params.items()),list(model2_params.items())):
    model2_params[model2_key] = pretrained_val
    if model2_key == 'layer4.1.bn2.num_batches_tracked':
        break

# don't forget to load parametes dict to your model!!!
model2.load_state_dict(model2_params)

print(model1.state_dict())
```

    OrderedDict([('conv1.weight', tensor([[[[ 1.2353e-02, -3.5713e-02, -4.6419e-02,  ..., -5.4487e-04,
               -2.5580e-02,  9.5039e-03],
              [-1.4113e-02, -2.1687e-02, -7.3655e-02,  ..., -2.4545e-02,
                3.7708e-02,  2.4177e-02],
              [-2.6923e-02, -2.2991e-03,  6.8104e-03,  ...,  3.0040e-02,
                2.5846e-02,  1.0737e-03],
              ...,
              [ 1.7352e-02, -2.8815e-02, -3.1425e-02,  ..., -1.4524e-02,
               -1.5380e-02,  1.5295e-02],
              [ 4.0697e-02, -6.9606e-03,  3.0921e-02,  ...,  4.7491e-03,
                2.9805e-03, -2.2234e-02],
              [-3.6224e-02, -1.5144e-02,  1.1833e-02,  ...,  5.6690e-03,
               -6.7935e-03,  7.5249e-03]],
    
             [[ 1.0802e-02, -1.3089e-02, -1.0400e-02,  ...,  5.3511e-03,
                1.5235e-02, -2.3676e-02],
              [ 5.4103e-03,  1.6151e-02,  4.3026e-02,  ...,  3.7497e-02,
                4.3525e-02, -2.6875e-02],
              [ 3.1831e-02,  5.4097e-02, -1.4268e-02,  ..., -3.4951e-02,
                3.2749e-02, -1.7871e-02],
              ...,
              [-5.7626e-02, -1.8986e-02,  6.0225e-03,  ...,  2.7881e-02,
               -7.7839e-03, -6.0318e-03],
              [ 1.9256e-02,  3.0078e-02,  3.5863e-02,  ...,  3.4886e-02,
                2.4019e-02,  8.8592e-03],
              [ 3.7452e-02,  5.7697e-03,  1.0191e-03,  ..., -2.0031e-02,
               -4.0363e-03, -1.6401e-02]],
    
             [[ 2.5891e-02,  1.7239e-02,  3.0213e-02,  ...,  1.0534e-02,
                5.1747e-02, -7.5586e-03],
              [-1.0796e-03,  1.6125e-02, -1.9219e-02,  ..., -2.1511e-02,
                3.4048e-02, -3.2933e-02],
              [ 7.1547e-03,  4.2137e-02, -9.0451e-03,  ...,  6.6442e-03,
                2.5878e-02,  2.4297e-02],
              ...,
              [-2.2563e-03,  7.9977e-04,  2.3270e-02,  ..., -3.2405e-02,
                6.0747e-04, -1.4864e-02],
              [-1.2239e-02, -1.9019e-02,  2.0364e-02,  ..., -4.4587e-02,
               -1.4965e-03, -7.7182e-03],
              [-4.3396e-02,  1.7862e-02, -1.6113e-03,  ...,  2.7045e-02,
                7.1188e-03,  1.3907e-02]]],


​    
            [[[ 5.5103e-03,  1.3440e-02, -5.4135e-02,  ...,  1.5246e-02,
                7.0073e-03, -6.8188e-03],
              [ 2.5464e-02,  1.9965e-03,  1.1401e-02,  ...,  2.2967e-02,
               -2.3158e-02, -4.8028e-02],
              [-3.0861e-02,  4.3261e-02,  2.2991e-04,  ...,  7.2141e-03,
               -2.5534e-03, -1.6390e-02],
              ...,
              [-2.2476e-03,  1.5484e-02, -1.8076e-02,  ...,  4.2129e-02,
               -2.6390e-02,  1.1068e-02],
              [-1.4368e-02, -2.3337e-02, -3.3074e-02,  ..., -4.1228e-02,
               -2.1975e-02,  1.5222e-02],
              [ 1.0343e-03,  1.2913e-02,  1.4365e-02,  ..., -1.7734e-02,
               -2.8202e-02,  1.5239e-02]],
    
             [[-1.9221e-02, -4.1880e-02, -3.4249e-02,  ..., -2.2087e-02,
               -9.4479e-03, -4.2507e-02],
              [ 1.5276e-02, -1.7290e-02, -5.0062e-03,  ..., -3.3725e-02,
                8.4538e-02,  7.4596e-02],
              [ 1.5014e-02,  1.3399e-02, -1.5247e-02,  ...,  6.1760e-03,
                1.0639e-03, -3.7298e-03],
              ...,
              [-1.7920e-02,  4.6874e-03, -1.3490e-02,  ...,  8.3387e-03,
                1.7101e-02, -2.1144e-02],
              [ 1.6083e-02, -2.8346e-02, -1.0181e-02,  ...,  8.7736e-03,
                1.7608e-02,  1.3936e-03],
              [-1.3562e-02, -1.6548e-02, -1.0019e-02,  ..., -3.1214e-03,
               -4.5485e-02,  1.8684e-02]],
    
             [[ 1.6536e-03, -6.2918e-02,  3.2885e-02,  ..., -1.2632e-02,
               -1.9542e-02, -4.5302e-02],
              [-2.3802e-02, -3.7273e-03,  8.8764e-03,  ...,  1.1835e-03,
                3.5745e-02,  8.3474e-03],
              [-1.8022e-02,  1.2632e-02,  4.6297e-02,  ..., -1.1841e-02,
                1.9780e-02, -9.1604e-03],
              ...,
              [ 1.4568e-02, -6.0503e-02,  3.2855e-02,  ..., -1.6141e-02,
               -3.4471e-03,  3.1018e-02],
              [-1.8240e-02, -2.3850e-02, -1.0398e-02,  ..., -3.8220e-02,
               -1.7804e-02, -2.1852e-03],
              [ 1.8869e-03,  3.8349e-02, -6.0193e-03,  ..., -3.2453e-02,
                3.5725e-02,  3.7736e-02]]],


​    
            [[[-3.7067e-02,  2.5998e-02, -1.5807e-03,  ..., -4.3575e-03,
                8.9018e-03, -2.6813e-02],
              [-6.3004e-02,  2.5725e-02,  2.2938e-02,  ..., -6.4212e-03,
               -4.9227e-02,  8.6667e-03],
              [-1.0451e-02,  7.7711e-03,  1.2346e-02,  ...,  2.1020e-03,
                3.6730e-02,  8.9196e-03],
              ...,
              [-1.3248e-02, -1.3267e-02, -1.6068e-02,  ...,  1.0950e-02,
               -5.2188e-03, -4.5454e-04],
              [-1.7372e-02, -1.7221e-02, -3.5878e-02,  ..., -1.0645e-02,
               -5.3494e-03, -2.7517e-02],
              [ 3.0780e-02, -2.9385e-02, -1.5382e-03,  ..., -2.7368e-02,
               -5.5117e-02,  9.2951e-04]],
    
             [[-2.9642e-02,  3.0652e-02, -2.0592e-02,  ...,  3.3554e-03,
               -5.0659e-03,  2.5336e-02],
              [-1.7673e-02,  6.2349e-03,  1.8002e-03,  ...,  4.5638e-02,
                1.4157e-02,  6.8525e-03],
              [ 2.9224e-02, -1.0199e-02,  5.6825e-03,  ..., -4.1673e-03,
                1.3161e-02,  6.3385e-02],
              ...,
              [ 1.3108e-02, -1.1734e-02, -7.6878e-03,  ...,  2.2743e-02,
               -3.8229e-02,  1.6747e-02],
              [ 2.5327e-03,  2.6047e-03,  5.3567e-02,  ..., -2.6054e-03,
               -1.4169e-02,  9.2331e-04],
              [ 1.1231e-03,  4.1346e-02,  7.1592e-03,  ..., -1.8857e-02,
               -2.1284e-02,  3.1787e-03]],
    
             [[-2.2749e-02,  2.8438e-02,  1.7338e-02,  ..., -1.4904e-02,
                8.4677e-03,  2.4564e-02],
              [ 1.9473e-02, -2.8390e-02,  1.3051e-02,  ...,  1.4458e-03,
                6.6112e-03,  2.8940e-02],
              [ 1.2877e-02, -6.3633e-03, -1.5618e-02,  ...,  7.4578e-03,
               -1.8088e-02,  3.8192e-02],
              ...,
              [-6.9419e-03, -2.7709e-02,  4.2880e-02,  ..., -8.6494e-03,
                7.7893e-03,  1.4865e-02],
              [ 3.1384e-03,  3.7284e-03, -1.2248e-02,  ...,  4.7131e-02,
                9.8473e-03,  2.4719e-02],
              [ 9.0937e-03, -9.5125e-03,  3.4045e-02,  ...,  3.5318e-02,
                8.1867e-03, -1.7129e-02]]],


​    
            ...,


​    
            [[[ 1.0484e-04,  4.1647e-02, -4.4440e-02,  ..., -2.5187e-02,
               -2.6889e-02, -2.6403e-03],
              [-1.4718e-02,  3.9153e-02,  2.1888e-02,  ...,  4.7834e-02,
               -4.0136e-02,  1.2690e-02],
              [-2.0801e-02, -4.4061e-03,  9.3861e-03,  ...,  4.2037e-02,
                9.2128e-03, -1.1401e-02],
              ...,
              [ 1.7732e-02,  2.5918e-02,  7.7917e-03,  ..., -5.1466e-02,
               -4.4485e-02,  8.7730e-03],
              [-4.3991e-03,  1.5231e-02,  6.1977e-03,  ..., -2.1098e-02,
               -7.3863e-03,  7.3081e-02],
              [ 2.5606e-02,  2.2408e-02,  1.3843e-02,  ..., -2.2067e-02,
                1.4076e-03,  3.6694e-03]],
    
             [[-4.1789e-02, -1.2482e-02, -6.4251e-03,  ..., -4.3878e-03,
               -7.4479e-03, -1.3693e-02],
              [ 1.2541e-02,  1.0336e-02, -2.9981e-02,  ...,  1.6250e-02,
                1.5679e-02, -2.3492e-03],
              [-1.8501e-02,  3.0069e-03, -2.4815e-02,  ...,  4.0587e-02,
               -4.5073e-02,  1.6156e-02],
              ...,
              [ 1.5642e-02, -5.8200e-04,  5.1297e-03,  ...,  2.7395e-02,
                6.1172e-03,  2.5805e-02],
              [-2.5227e-02,  1.1054e-02, -4.7216e-02,  ..., -2.8864e-02,
                3.1125e-02,  7.3301e-03],
              [ 1.4667e-02, -3.2257e-02,  2.9765e-02,  ...,  1.9084e-02,
               -6.4017e-03,  2.5315e-02]],
    
             [[ 1.7699e-02, -9.3588e-03, -6.6574e-03,  ..., -3.5212e-02,
               -3.1355e-02, -3.9393e-02],
              [ 1.0191e-02,  3.0520e-02, -6.5790e-03,  ...,  1.8725e-02,
                3.6009e-03, -1.1696e-02],
              [-3.4537e-03,  1.8168e-02, -4.9313e-03,  ...,  2.8893e-02,
                1.4810e-02,  1.2410e-02],
              ...,
              [-3.0489e-02,  2.8490e-02, -2.0475e-02,  ...,  2.5340e-02,
               -1.2772e-02,  4.1910e-03],
              [ 1.2047e-02, -1.3051e-02, -1.0325e-02,  ..., -7.2207e-03,
                1.3946e-02, -3.5824e-02],
              [-3.5725e-02,  5.6465e-03, -4.8942e-02,  ...,  2.7280e-02,
               -2.2560e-02,  2.2423e-02]]],


​    
            [[[ 2.5215e-02, -2.6037e-02, -7.1189e-03,  ...,  1.2421e-02,
               -2.3827e-02,  1.2885e-02],
              [ 2.8025e-02,  2.2198e-02, -1.9905e-02,  ..., -2.6249e-02,
                3.4476e-02,  1.1111e-03],
              [ 2.0601e-02,  1.3386e-02,  9.3798e-03,  ...,  1.8897e-02,
                4.5602e-02,  5.2975e-03],
              ...,
              [ 1.6043e-02, -2.6220e-02,  2.8817e-02,  ...,  3.2787e-02,
               -2.0426e-02, -4.5844e-03],
              [ 2.8145e-03,  9.9836e-03,  1.7517e-02,  ...,  9.4083e-03,
               -3.1759e-02,  6.0568e-02],
              [-9.7894e-04,  3.2081e-02,  1.0885e-02,  ..., -2.7470e-02,
                7.8087e-03, -8.2926e-03]],
    
             [[ 1.2586e-02,  1.4858e-02,  2.2889e-02,  ..., -2.5693e-02,
                6.6764e-04, -2.6042e-02],
              [-1.7399e-02, -4.7715e-03,  7.1666e-03,  ...,  4.3989e-02,
               -4.7763e-02, -1.5933e-03],
              [-3.5319e-03,  1.7150e-02, -3.7306e-02,  ...,  2.9311e-02,
                2.1062e-02,  1.7357e-02],
              ...,
              [-1.5538e-02,  3.4753e-03,  1.9574e-02,  ...,  3.0604e-02,
                9.5223e-03,  4.1064e-02],
              [ 1.7160e-02, -1.9599e-02,  3.3547e-04,  ...,  5.6494e-02,
               -3.5979e-02,  8.9146e-03],
              [-1.0057e-03,  7.0559e-03,  5.7961e-03,  ...,  1.1160e-02,
               -2.6502e-02,  8.7206e-03]],
    
             [[-1.1577e-03, -2.7506e-04, -2.8923e-03,  ..., -1.4175e-03,
               -2.3723e-02, -2.3104e-02],
              [ 8.9705e-03,  2.4740e-02, -2.5172e-02,  ..., -8.8374e-03,
               -4.6386e-02, -2.0492e-02],
              [-1.4191e-02, -1.8026e-02, -3.3038e-02,  ..., -2.7241e-02,
                1.5081e-02,  1.5494e-02],
              ...,
              [ 8.4246e-03,  3.8451e-03, -1.7167e-02,  ...,  6.2993e-02,
               -3.7641e-04,  1.7011e-02],
              [-4.9899e-02,  3.7582e-02,  1.6764e-02,  ...,  8.1563e-03,
                5.6094e-03, -2.6289e-03],
              [-1.7065e-02, -8.2094e-03,  2.0225e-02,  ...,  1.5519e-02,
                3.7813e-02,  3.7859e-02]]],


​    
            [[[ 1.1671e-02,  8.9922e-05,  1.3341e-02,  ...,  3.7564e-02,
               -1.7323e-02,  1.5196e-02],
              [-2.3953e-02,  2.9051e-03, -5.9366e-02,  ..., -1.5524e-02,
               -1.0648e-02, -8.8292e-03],
              [ 4.2525e-03, -7.0732e-03, -5.1372e-04,  ..., -1.2803e-02,
                1.0091e-03, -3.1203e-02],
              ...,
              [ 8.3300e-03, -1.3929e-02,  4.4173e-02,  ..., -4.3772e-03,
                4.5299e-02,  6.0889e-03],
              [-2.7856e-02, -1.4683e-02, -1.5261e-02,  ..., -2.5578e-02,
               -1.2986e-02,  8.5385e-03],
              [ 1.8987e-02,  1.7251e-02,  2.2474e-03,  ...,  5.2491e-03,
                2.7538e-02,  1.9099e-02]],
    
             [[-2.1429e-02, -2.1452e-02, -1.8188e-03,  ..., -1.9078e-02,
               -2.6875e-02,  1.6612e-02],
              [-1.4193e-02,  2.2775e-02, -7.5320e-03,  ...,  4.0329e-02,
                2.3638e-03,  1.8719e-02],
              [ 1.2235e-02,  7.1510e-03, -2.9710e-02,  ..., -3.6395e-02,
               -1.5017e-02, -3.0130e-02],
              ...,
              [-4.3853e-03,  1.2023e-02,  1.9581e-02,  ...,  1.8423e-02,
               -1.5625e-02, -3.9582e-02],
              [ 2.1389e-02, -1.0032e-02,  3.3228e-02,  ...,  5.2692e-02,
                1.8387e-02,  1.1849e-02],
              [-5.2513e-02,  3.9669e-02, -3.7646e-02,  ..., -3.4407e-02,
               -3.2430e-02, -5.8756e-03]],
    
             [[-1.5025e-02,  5.4491e-03,  3.3839e-03,  ..., -4.3292e-02,
               -4.1655e-02, -6.3504e-02],
              [-2.1221e-02, -1.5059e-03,  6.1776e-04,  ...,  4.8302e-03,
                2.3602e-02,  2.6395e-02],
              [-2.4673e-02, -7.1872e-03, -1.2602e-02,  ...,  3.5135e-02,
               -4.5383e-03,  2.2941e-03],
              ...,
              [-2.0872e-02,  1.5452e-02, -3.1187e-02,  ...,  4.5221e-03,
               -2.5530e-02, -1.4707e-02],
              [ 4.1107e-02, -3.1206e-02,  2.8682e-02,  ..., -7.0993e-03,
                3.0871e-02, -1.2732e-02],
              [-7.9013e-03,  2.0054e-02, -6.2115e-03,  ...,  3.4874e-02,
                2.2789e-03,  1.9972e-02]]]])), ('bn1.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])), ('bn1.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('bn1.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('bn1.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])), ('bn1.num_batches_tracked', tensor(0)), ('layer1.0.conv1.weight', tensor([[[[ 3.0539e-02,  1.9263e-02, -9.5153e-03],
              [-1.6868e-02,  8.1200e-02, -5.0737e-02],
              [-1.4941e-02,  9.8156e-02,  2.4128e-02]],
    
             [[-3.5668e-03, -1.7678e-03,  1.0533e-02],
              [-1.2527e-01, -4.3889e-02, -1.9012e-02],
              [-1.1522e-02, -5.6890e-02,  3.5555e-02]],
    
             [[-1.3885e-02, -4.6435e-02,  8.2544e-02],
              [ 4.4329e-02,  9.3898e-02,  2.1418e-02],
              [-2.4573e-02, -2.0268e-03,  4.4004e-02]],
    
             ...,
    
             [[-1.6858e-02,  1.8185e-02,  9.6267e-02],
              [ 2.6345e-02, -5.6629e-02,  5.5235e-03],
              [-3.3296e-02,  9.4378e-02, -6.3945e-02]],
    
             [[ 2.9833e-02,  8.4714e-02,  3.4327e-02],
              [ 8.7805e-02,  1.6598e-03,  1.1578e-01],
              [-1.2608e-02, -7.5151e-03,  5.4985e-02]],
    
             [[-1.2946e-01, -3.8235e-02,  2.0858e-02],
              [ 3.6629e-02, -8.0922e-03,  4.9266e-02],
              [ 5.9833e-02, -1.4829e-01, -7.2516e-03]]],


​    
            [[[ 3.1191e-02, -4.3731e-02, -2.6382e-02],
              [ 1.4370e-01, -1.5643e-02,  3.0076e-02],
              [-3.9352e-02,  3.6075e-02,  1.3779e-03]],
    
             [[-9.7976e-02, -4.2582e-02,  4.7527e-02],
              [-6.0998e-02,  1.4464e-02, -4.8669e-02],
              [-5.6894e-02,  8.1457e-02, -2.4365e-02]],
    
             [[ 1.6716e-02,  1.4763e-04, -6.4492e-02],
              [-4.4191e-02,  1.8873e-01,  8.8353e-02],
              [ 2.9974e-02,  1.6756e-02,  2.0861e-02]],
    
             ...,
    
             [[ 1.2659e-01,  5.4396e-02, -3.1809e-02],
              [-2.4254e-02, -2.8308e-02, -5.3568e-02],
              [ 7.3916e-02,  1.2409e-01,  1.6906e-02]],
    
             [[ 8.7342e-02,  4.5410e-02, -1.6811e-02],
              [-1.1112e-02, -3.1272e-02,  5.4239e-02],
              [-3.2455e-02,  4.1273e-02,  5.1288e-03]],
    
             [[-5.1213e-02,  6.2314e-02, -1.4645e-02],
              [ 4.9925e-02, -3.3413e-02, -1.7102e-02],
              [-3.2561e-02,  2.9704e-02,  1.0166e-01]]],


​    
            [[[ 1.7685e-02, -4.6276e-02, -6.1931e-02],
              [-1.6014e-02, -1.3426e-01,  1.3155e-01],
              [-1.0861e-02,  1.0231e-01,  5.3349e-02]],
    
             [[ 4.5723e-02,  4.2911e-02, -8.4300e-03],
              [ 1.5883e-02,  9.9476e-02, -2.7056e-02],
              [ 1.6891e-02, -1.5752e-04, -9.7016e-02]],
    
             [[-1.2807e-01, -5.4600e-02, -2.2462e-02],
              [-6.2221e-03, -1.8678e-02, -9.5671e-02],
              [-7.4823e-02,  2.1588e-02, -4.0575e-02]],
    
             ...,
    
             [[ 5.5803e-03, -1.7282e-02,  3.0226e-02],
              [ 5.1910e-02,  6.6598e-02,  1.0962e-01],
              [-8.4714e-02,  7.3650e-02, -2.5640e-02]],
    
             [[ 8.5122e-02,  4.1985e-02, -1.2679e-03],
              [-1.4886e-01,  4.8878e-02, -2.1148e-02],
              [-8.1039e-02, -7.3524e-02, -3.2161e-02]],
    
             [[-8.3176e-02, -5.2273e-02,  7.0170e-02],
              [-9.2101e-02, -1.6115e-03, -8.0622e-02],
              [ 6.2005e-02,  1.0547e-01, -5.1940e-02]]],


​    
            ...,


​    
            [[[ 4.6587e-02,  3.1479e-02, -3.1340e-02],
              [ 6.5531e-02, -5.0680e-02,  5.0297e-02],
              [-3.1221e-02,  4.4492e-02, -4.1228e-02]],
    
             [[-3.2066e-02, -3.0490e-02,  6.1165e-02],
              [-7.8898e-03, -8.2159e-02, -2.2750e-02],
              [ 3.4592e-02,  1.3200e-01,  8.1046e-02]],
    
             [[-5.5585e-02, -5.0348e-03, -1.4225e-02],
              [ 2.0807e-02,  2.7146e-02, -9.3803e-02],
              [ 4.9110e-02, -1.8202e-02, -6.5716e-02]],
    
             ...,
    
             [[ 6.8095e-02, -8.5798e-02,  5.9580e-02],
              [-2.2132e-03, -6.3908e-02, -3.7729e-02],
              [-3.3713e-02,  4.3529e-02,  2.3565e-02]],
    
             [[-1.2734e-02,  4.0865e-02, -2.0637e-02],
              [ 2.6333e-02,  3.1064e-02, -1.1274e-01],
              [ 4.4891e-02, -5.4836e-02, -1.0355e-01]],
    
             [[ 1.7384e-02,  8.2354e-02,  3.6510e-02],
              [-7.5621e-03,  5.1190e-02, -1.0679e-01],
              [-2.5223e-02,  1.5098e-02, -8.4929e-02]]],


​    
            [[[-2.3767e-02, -1.9151e-02, -1.0226e-01],
              [ 8.5812e-02, -6.7412e-02, -6.5305e-02],
              [-3.5956e-02,  2.2744e-02, -1.7631e-02]],
    
             [[-9.5714e-02, -7.8447e-02, -3.2280e-02],
              [ 7.4174e-02, -4.2331e-02, -9.6931e-02],
              [-1.7788e-02, -1.7947e-02, -2.3771e-02]],
    
             [[-1.7588e-02, -1.0841e-01, -1.7130e-02],
              [-1.1424e-01, -2.8594e-02,  2.8611e-02],
              [ 3.2761e-02, -6.3870e-02,  5.6570e-02]],
    
             ...,
    
             [[-5.4793e-02, -8.1073e-02, -3.8775e-02],
              [ 5.4089e-02,  8.1337e-02, -1.2124e-01],
              [ 9.2851e-02,  1.0174e-02,  1.1949e-03]],
    
             [[ 1.2050e-02,  8.1603e-02, -1.9838e-02],
              [-2.0533e-02,  4.8026e-02,  1.2331e-02],
              [-3.0673e-03,  1.5969e-02, -7.5717e-02]],
    
             [[ 2.3808e-03, -4.0620e-02,  5.3948e-02],
              [-1.8374e-02,  7.5871e-03,  5.1573e-02],
              [-4.2799e-02, -2.3825e-02,  9.0677e-03]]],


​    
            [[[-1.4914e-01,  6.0750e-02, -1.5681e-02],
              [ 8.3451e-02, -1.2266e-02,  1.3369e-02],
              [-1.2846e-01, -3.8551e-02,  9.5243e-03]],
    
             [[ 6.4532e-02, -9.8366e-03,  5.3057e-02],
              [ 8.9660e-02,  1.6758e-03,  3.0947e-02],
              [ 1.0565e-01, -1.0911e-01, -3.9327e-02]],
    
             [[-3.4817e-02,  1.1678e-01, -1.0543e-02],
              [ 4.7430e-02,  2.3337e-02,  6.9885e-02],
              [ 2.7513e-02, -8.1519e-03, -7.1742e-02]],
    
             ...,
    
             [[ 3.2491e-02, -7.0260e-02, -3.3710e-02],
              [ 1.0237e-02,  9.9978e-02,  2.9672e-02],
              [-9.3573e-03,  1.0182e-01,  1.6146e-02]],
    
             [[ 1.6679e-01,  2.2240e-02, -8.0446e-02],
              [-7.6930e-02, -4.4717e-02,  3.0263e-02],
              [ 6.5176e-02,  1.7422e-03, -1.6081e-02]],
    
             [[ 5.1144e-02, -6.7699e-02, -7.4803e-02],
              [-6.8858e-03,  8.3474e-02, -1.4611e-02],
              [ 8.9938e-03, -5.9545e-02,  6.8011e-02]]]])), ('layer1.0.bn1.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])), ('layer1.0.bn1.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer1.0.bn1.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer1.0.bn1.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])), ('layer1.0.bn1.num_batches_tracked', tensor(0)), ('layer1.0.conv2.weight', tensor([[[[ 0.0083, -0.0330,  0.0239],
              [ 0.1444,  0.0189,  0.0167],
              [-0.0436,  0.0294,  0.0107]],
    
             [[-0.1209, -0.0254, -0.0941],
              [-0.0487, -0.0630, -0.0906],
              [ 0.0006,  0.0342,  0.0982]],
    
             [[-0.0041,  0.0920, -0.0070],
              [-0.0273,  0.0331, -0.0145],
              [-0.1147, -0.0142,  0.0629]],
    
             ...,
    
             [[-0.0096, -0.0182, -0.0159],
              [ 0.1239, -0.0338,  0.0627],
              [-0.0305, -0.0864,  0.0739]],
    
             [[-0.0162, -0.0785, -0.1316],
              [ 0.0478,  0.0196,  0.0629],
              [ 0.0621, -0.0325,  0.0875]],
    
             [[ 0.0043, -0.1119,  0.0232],
              [ 0.0625,  0.1038, -0.0756],
              [ 0.0425, -0.0416,  0.0315]]],


​    
            [[[-0.0910, -0.0469,  0.0288],
              [ 0.0337, -0.0080, -0.0733],
              [ 0.0270, -0.0115, -0.0522]],
    
             [[-0.0318,  0.0210,  0.0072],
              [ 0.0030,  0.0356,  0.0582],
              [-0.0695,  0.0288,  0.0182]],
    
             [[-0.0327,  0.0090, -0.0537],
              [-0.0192,  0.0546, -0.0595],
              [-0.0541,  0.0084,  0.0343]],
    
             ...,
    
             [[ 0.0542,  0.0615, -0.0681],
              [-0.0219,  0.1173, -0.0402],
              [ 0.0097, -0.0537,  0.0559]],
    
             [[-0.0379,  0.0716, -0.0035],
              [ 0.0098,  0.0187,  0.0190],
              [-0.0257, -0.0989, -0.0778]],
    
             [[ 0.0420,  0.0235,  0.0480],
              [-0.0723, -0.0717, -0.0880],
              [ 0.0378, -0.0219,  0.0515]]],


​    
            [[[-0.0492,  0.0226,  0.0223],
              [-0.0254, -0.0417,  0.0176],
              [-0.0005, -0.0251, -0.0185]],
    
             [[-0.0328, -0.0242, -0.0487],
              [-0.0183,  0.0093, -0.0453],
              [ 0.0281, -0.0482,  0.0656]],
    
             [[-0.0157, -0.0337,  0.1204],
              [ 0.0717,  0.0375,  0.0046],
              [ 0.1011, -0.0440,  0.0141]],
    
             ...,
    
             [[-0.0673, -0.0259, -0.0246],
              [-0.0129,  0.0625, -0.0236],
              [ 0.0643,  0.0036, -0.0228]],
    
             [[-0.0136,  0.0921, -0.0378],
              [-0.0193, -0.0240, -0.0241],
              [-0.0187, -0.0062, -0.0296]],
    
             [[ 0.1154, -0.0782,  0.0653],
              [ 0.0156, -0.0046,  0.0296],
              [ 0.1194,  0.0948, -0.0432]]],


​    
            ...,


​    
            [[[ 0.0365,  0.0108,  0.0052],
              [-0.0331, -0.0711,  0.0663],
              [ 0.0771,  0.0402,  0.0853]],
    
             [[-0.0210,  0.0714,  0.0206],
              [ 0.0606, -0.0247, -0.0115],
              [-0.0478, -0.0496,  0.1077]],
    
             [[-0.0271,  0.0521,  0.0589],
              [-0.0165,  0.0817, -0.0201],
              [-0.0580, -0.0378, -0.0737]],
    
             ...,
    
             [[ 0.0976,  0.0284,  0.0424],
              [-0.0407, -0.0614, -0.0880],
              [-0.0502, -0.0829,  0.0614]],
    
             [[-0.0403,  0.0362, -0.0232],
              [-0.0203, -0.0789,  0.0276],
              [-0.0240,  0.0607,  0.0470]],
    
             [[-0.0507, -0.0425, -0.0552],
              [ 0.0509,  0.0233, -0.0280],
              [-0.1407, -0.0146, -0.0659]]],


​    
            [[[-0.0501,  0.0019,  0.0723],
              [-0.0173,  0.0494, -0.0947],
              [ 0.0874, -0.0464, -0.0561]],
    
             [[ 0.1013,  0.0606, -0.0128],
              [ 0.0755, -0.0198,  0.0805],
              [-0.0607,  0.0667,  0.0699]],
    
             [[-0.0165,  0.0337, -0.0096],
              [-0.0110, -0.0783,  0.0059],
              [-0.0598, -0.0082, -0.0286]],
    
             ...,
    
             [[ 0.0175,  0.0038,  0.0052],
              [-0.0550, -0.0232, -0.0211],
              [ 0.0114,  0.0307,  0.0737]],
    
             [[-0.0087, -0.0385, -0.0230],
              [-0.0099,  0.0482,  0.0191],
              [-0.0498,  0.0061,  0.0107]],
    
             [[-0.0619, -0.0375, -0.0460],
              [-0.0748,  0.0587,  0.0155],
              [-0.0382, -0.1296,  0.0401]]],


​    
            [[[ 0.1032,  0.0737, -0.0121],
              [-0.0740,  0.0474,  0.1038],
              [ 0.0264,  0.0181, -0.0661]],
    
             [[ 0.0893, -0.1562,  0.0106],
              [-0.0010, -0.0407,  0.0568],
              [-0.0355, -0.0063, -0.0501]],
    
             [[ 0.0838, -0.0432, -0.1053],
              [-0.0758, -0.0197,  0.0002],
              [-0.0392,  0.0217, -0.0483]],
    
             ...,
    
             [[ 0.0440,  0.1094,  0.0016],
              [-0.0285,  0.0497,  0.0423],
              [-0.0162, -0.0153, -0.0864]],
    
             [[ 0.0398,  0.0847,  0.0057],
              [-0.0832, -0.0192,  0.0490],
              [ 0.0811,  0.0741, -0.0071]],
    
             [[-0.0179, -0.1659, -0.0107],
              [ 0.0372,  0.0517,  0.0248],
              [-0.0744,  0.0986, -0.0061]]]])), ('layer1.0.bn2.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])), ('layer1.0.bn2.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer1.0.bn2.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer1.0.bn2.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])), ('layer1.0.bn2.num_batches_tracked', tensor(0)), ('layer1.1.conv1.weight', tensor([[[[-0.1661, -0.0325,  0.0481],
              [ 0.0180,  0.0693,  0.0220],
              [-0.0003, -0.0421,  0.1044]],
    
             [[-0.0187,  0.0235, -0.0151],
              [-0.0813,  0.0226, -0.0949],
              [ 0.0097,  0.0214,  0.0410]],
    
             [[-0.0135,  0.0218,  0.0028],
              [ 0.0019, -0.0467, -0.0480],
              [ 0.0054, -0.0323, -0.0525]],
    
             ...,
    
             [[-0.1421,  0.0737,  0.0375],
              [ 0.0137,  0.0635, -0.0040],
              [ 0.0347, -0.0009, -0.0568]],
    
             [[ 0.0090,  0.0043,  0.0387],
              [-0.0073, -0.0276,  0.0165],
              [ 0.0033, -0.1153, -0.0409]],
    
             [[-0.0409, -0.0183,  0.0113],
              [-0.0867,  0.0243, -0.0858],
              [-0.0934,  0.1038, -0.1234]]],


​    
            [[[-0.0077,  0.0054, -0.0260],
              [ 0.0113,  0.0551,  0.0002],
              [-0.0482, -0.0336,  0.0117]],
    
             [[ 0.0295, -0.0811,  0.0316],
              [-0.0098, -0.0331, -0.0203],
              [-0.0209,  0.0733, -0.0355]],
    
             [[-0.0141,  0.0304,  0.0505],
              [-0.0175,  0.0566, -0.0006],
              [-0.0098,  0.0682,  0.0371]],
    
             ...,
    
             [[ 0.0124, -0.0007, -0.0031],
              [-0.0143,  0.1253, -0.0289],
              [-0.0202, -0.0201, -0.0772]],
    
             [[-0.0043,  0.0408, -0.0973],
              [-0.0419, -0.0009,  0.0605],
              [ 0.0404, -0.0502,  0.0407]],
    
             [[ 0.0026,  0.0231, -0.0167],
              [ 0.0552, -0.0043,  0.0487],
              [ 0.0675,  0.0704, -0.0067]]],


​    
            [[[-0.0854,  0.1115,  0.0739],
              [ 0.0499, -0.1231,  0.0368],
              [-0.0748,  0.0232, -0.0357]],
    
             [[ 0.0052,  0.0502, -0.0465],
              [ 0.0301,  0.0489,  0.0320],
              [ 0.0301, -0.0610,  0.0776]],
    
             [[ 0.0165,  0.0352,  0.0138],
              [ 0.0055,  0.0054, -0.0178],
              [ 0.0606,  0.0947,  0.0279]],
    
             ...,
    
             [[-0.0012, -0.0921,  0.0333],
              [-0.0367, -0.1523, -0.0472],
              [ 0.0398,  0.0314, -0.1023]],
    
             [[ 0.1261, -0.0244, -0.0582],
              [-0.1445, -0.0890, -0.1329],
              [-0.0689,  0.1158,  0.0307]],
    
             [[-0.0027, -0.0194,  0.0237],
              [ 0.0144, -0.0117,  0.0261],
              [-0.0502, -0.0103, -0.0098]]],


​    
            ...,


​    
            [[[ 0.1260,  0.0500, -0.0215],
              [-0.0723, -0.1556, -0.0259],
              [-0.0719, -0.0888, -0.0642]],
    
             [[ 0.0804,  0.0390, -0.0229],
              [-0.0479, -0.0119,  0.0435],
              [ 0.1310,  0.0729, -0.0791]],
    
             [[ 0.0423, -0.1180, -0.0675],
              [-0.0425, -0.1203,  0.0044],
              [ 0.0638,  0.0859,  0.0578]],
    
             ...,
    
             [[ 0.0593,  0.0736, -0.0529],
              [-0.0108, -0.0475, -0.0877],
              [-0.0467,  0.0087,  0.0352]],
    
             [[ 0.2153, -0.1379,  0.0200],
              [-0.0499,  0.0454, -0.1016],
              [ 0.0157, -0.0685,  0.1457]],
    
             [[-0.0687,  0.0630, -0.0864],
              [-0.1005,  0.0715,  0.0174],
              [-0.0171,  0.0036, -0.0891]]],


​    
            [[[-0.0446, -0.0626, -0.0922],
              [-0.0193,  0.0444,  0.0487],
              [ 0.0412,  0.0356, -0.0672]],
    
             [[-0.0293,  0.0417, -0.0478],
              [-0.0681, -0.0323, -0.0284],
              [-0.0575, -0.0768,  0.0582]],
    
             [[-0.0893, -0.0532, -0.0345],
              [-0.0412, -0.0468,  0.0752],
              [-0.0912,  0.0663,  0.0715]],
    
             ...,
    
             [[ 0.1167,  0.0335,  0.0335],
              [-0.0106, -0.1084, -0.0503],
              [ 0.0413,  0.0730, -0.0194]],
    
             [[ 0.1289,  0.0234, -0.0755],
              [ 0.0372,  0.0043,  0.0784],
              [-0.0068,  0.0358,  0.0871]],
    
             [[ 0.0393,  0.0068,  0.0128],
              [ 0.0424, -0.0368, -0.0324],
              [ 0.0648,  0.0259,  0.0306]]],


​    
            [[[ 0.0691,  0.0583,  0.0003],
              [-0.1522, -0.0230, -0.0126],
              [-0.0031,  0.0124,  0.1019]],
    
             [[ 0.0132,  0.0181, -0.0091],
              [ 0.0380, -0.1025, -0.0396],
              [ 0.1248, -0.0299,  0.0515]],
    
             [[-0.0309, -0.0312, -0.0271],
              [ 0.0667,  0.0953, -0.0649],
              [ 0.0105, -0.0209, -0.0163]],
    
             ...,
    
             [[-0.1490,  0.0701, -0.0396],
              [-0.0523,  0.0123, -0.0107],
              [ 0.1744,  0.0265,  0.0721]],
    
             [[ 0.0931,  0.0099,  0.0383],
              [ 0.0188, -0.0077, -0.0936],
              [-0.0108,  0.0174,  0.1103]],
    
             [[-0.0816, -0.0310, -0.0625],
              [ 0.0325, -0.0471,  0.0362],
              [-0.0239, -0.0185,  0.0184]]]])), ('layer1.1.bn1.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])), ('layer1.1.bn1.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer1.1.bn1.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer1.1.bn1.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])), ('layer1.1.bn1.num_batches_tracked', tensor(0)), ('layer1.1.conv2.weight', tensor([[[[-0.0418,  0.0517, -0.0367],
              [-0.1072,  0.0787, -0.0783],
              [ 0.0037, -0.0077,  0.0740]],
    
             [[-0.0156, -0.0413, -0.0214],
              [ 0.0883, -0.1097,  0.0673],
              [ 0.0476, -0.1062,  0.0265]],
    
             [[-0.0783, -0.0212, -0.0339],
              [-0.0383,  0.0644,  0.0198],
              [ 0.0287,  0.0573, -0.0919]],
    
             ...,
    
             [[ 0.0576,  0.0018, -0.0310],
              [-0.0161, -0.1002,  0.0616],
              [-0.0237,  0.0053,  0.0048]],
    
             [[-0.0185, -0.0040,  0.0640],
              [-0.0283,  0.0555,  0.0292],
              [ 0.0827, -0.0251,  0.0036]],
    
             [[ 0.0546,  0.1813, -0.0761],
              [ 0.0566, -0.0218, -0.1569],
              [ 0.1152, -0.0317,  0.0405]]],


​    
            [[[-0.1422,  0.0019, -0.1374],
              [ 0.1368, -0.0819, -0.0020],
              [-0.0233,  0.0156,  0.0488]],
    
             [[-0.0368,  0.0828, -0.0558],
              [ 0.0523, -0.0483, -0.0482],
              [ 0.1053,  0.1068,  0.0187]],
    
             [[ 0.0198, -0.0718, -0.1419],
              [-0.0948,  0.0147,  0.0190],
              [ 0.0108, -0.0117, -0.0271]],
    
             ...,
    
             [[ 0.0678,  0.0224,  0.0125],
              [ 0.0632, -0.0364,  0.0726],
              [ 0.0148, -0.1674, -0.1287]],
    
             [[ 0.0382,  0.0220, -0.0394],
              [-0.0342, -0.0371,  0.0544],
              [-0.0029, -0.0019, -0.0645]],
    
             [[-0.0156, -0.0399, -0.1027],
              [ 0.1213,  0.0268, -0.0537],
              [ 0.0584, -0.0146,  0.0157]]],


​    
            [[[ 0.0946, -0.0692,  0.1115],
              [ 0.0107,  0.0624, -0.0535],
              [-0.0596,  0.0018,  0.0396]],
    
             [[-0.0548,  0.1033,  0.0296],
              [ 0.0219, -0.0318, -0.0630],
              [-0.0180, -0.0005,  0.0548]],
    
             [[ 0.0556,  0.1429,  0.0296],
              [ 0.0171, -0.0699, -0.0222],
              [-0.1353, -0.0404,  0.0345]],
    
             ...,
    
             [[-0.0066, -0.0689, -0.1081],
              [ 0.0955,  0.0188,  0.0044],
              [-0.0013, -0.0772,  0.0168]],
    
             [[-0.0169,  0.0162, -0.0034],
              [-0.0236,  0.0275,  0.0925],
              [-0.0112,  0.0091, -0.0394]],
    
             [[ 0.0500,  0.0612, -0.0636],
              [-0.0369,  0.1176, -0.0574],
              [-0.0291, -0.0182,  0.0071]]],


​    
            ...,


​    
            [[[ 0.1107, -0.0451, -0.0485],
              [ 0.0133, -0.0131,  0.0128],
              [ 0.0743,  0.0387, -0.0319]],
    
             [[-0.0395, -0.0511,  0.0265],
              [ 0.0023,  0.0313,  0.0538],
              [ 0.0274, -0.0821,  0.0272]],
    
             [[ 0.0004,  0.0754, -0.0057],
              [ 0.0763,  0.0108, -0.0086],
              [-0.0390,  0.0788, -0.0507]],
    
             ...,
    
             [[ 0.0911,  0.0784,  0.0418],
              [ 0.0081,  0.0178, -0.0586],
              [ 0.0143,  0.0875, -0.0307]],
    
             [[ 0.1231,  0.0539,  0.0040],
              [ 0.0395, -0.0399, -0.1014],
              [ 0.0648, -0.0134,  0.0969]],
    
             [[-0.0551, -0.0911,  0.0094],
              [-0.0094, -0.1176,  0.0225],
              [ 0.0309, -0.0439, -0.0350]]],


​    
            [[[-0.0802, -0.0111, -0.0389],
              [-0.0039, -0.0396, -0.0477],
              [ 0.0213, -0.0263,  0.0047]],
    
             [[-0.0593, -0.0311, -0.0076],
              [ 0.1850,  0.0092, -0.0523],
              [-0.0179,  0.1118, -0.0099]],
    
             [[-0.0127,  0.0157,  0.0159],
              [ 0.0758, -0.0141, -0.0721],
              [ 0.0239,  0.1099, -0.0094]],
    
             ...,
    
             [[-0.0427,  0.0406,  0.0056],
              [-0.0218, -0.0121, -0.0541],
              [ 0.0533, -0.1114, -0.0181]],
    
             [[-0.0203, -0.0509, -0.0655],
              [ 0.0229,  0.0841,  0.0253],
              [ 0.0395, -0.0941, -0.0103]],
    
             [[-0.0830,  0.0291, -0.0449],
              [-0.0625,  0.0190,  0.0918],
              [-0.0615,  0.0039,  0.0896]]],


​    
            [[[-0.0533,  0.0376, -0.0035],
              [ 0.0514,  0.0254, -0.1093],
              [-0.0729, -0.0984,  0.1304]],
    
             [[-0.0579,  0.0398, -0.0262],
              [-0.0217,  0.0503, -0.0140],
              [-0.0552, -0.0712, -0.0095]],
    
             [[ 0.0142, -0.0578,  0.0958],
              [-0.0318,  0.0626,  0.0492],
              [ 0.0109, -0.0047,  0.0003]],
    
             ...,
    
             [[-0.0039,  0.0532,  0.0530],
              [ 0.0090,  0.0223,  0.0167],
              [-0.0387, -0.0130,  0.0584]],
    
             [[ 0.0535, -0.1143,  0.0704],
              [ 0.0114, -0.0757, -0.0231],
              [ 0.1362, -0.0145, -0.0142]],
    
             [[ 0.0470, -0.0066,  0.0616],
              [ 0.0179,  0.0076,  0.0384],
              [-0.0093, -0.0557, -0.0846]]]])), ('layer1.1.bn2.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])), ('layer1.1.bn2.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer1.1.bn2.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer1.1.bn2.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])), ('layer1.1.bn2.num_batches_tracked', tensor(0)), ('layer2.0.conv1.weight', tensor([[[[ 0.0060, -0.0458,  0.0395],
              [-0.0618, -0.0014, -0.0316],
              [ 0.0437,  0.0058,  0.0027]],
    
             [[-0.0855, -0.0436, -0.0019],
              [-0.0467,  0.0367, -0.0278],
              [-0.0004,  0.0849,  0.0615]],
    
             [[-0.0099,  0.0283,  0.0683],
              [ 0.0167,  0.0170,  0.0051],
              [-0.0412, -0.0289, -0.0280]],
    
             ...,
    
             [[ 0.0478, -0.0383,  0.0187],
              [ 0.0094,  0.0047,  0.0491],
              [ 0.0179,  0.0175, -0.0291]],
    
             [[-0.0653, -0.0411, -0.0138],
              [ 0.1275,  0.0323,  0.0157],
              [-0.0130,  0.0325,  0.0376]],
    
             [[-0.0172, -0.0395,  0.0027],
              [ 0.0210,  0.0518,  0.0195],
              [-0.0436,  0.0678,  0.0457]]],


​    
            [[[-0.0013, -0.0328, -0.0262],
              [-0.0115,  0.0324, -0.0278],
              [-0.0248, -0.0294, -0.0380]],
    
             [[ 0.0403, -0.0017,  0.0553],
              [ 0.0593, -0.0345, -0.0149],
              [ 0.0094,  0.0113,  0.0617]],
    
             [[ 0.0438,  0.0013,  0.0569],
              [ 0.0134,  0.0698,  0.0032],
              [-0.0487,  0.0060, -0.0422]],
    
             ...,
    
             [[-0.0056,  0.0620, -0.0209],
              [-0.0107,  0.0245,  0.0321],
              [-0.0604,  0.0308, -0.0498]],
    
             [[-0.0384,  0.0313,  0.0267],
              [-0.0731,  0.0370,  0.0448],
              [ 0.0489,  0.0586, -0.0123]],
    
             [[-0.0310,  0.0247,  0.0184],
              [ 0.0207, -0.0285, -0.0191],
              [ 0.0201, -0.0094, -0.0130]]],


​    
            [[[-0.0183, -0.0379, -0.0875],
              [-0.0086, -0.0389, -0.0356],
              [ 0.0400, -0.0403,  0.1065]],
    
             [[-0.0492,  0.0258,  0.0319],
              [ 0.0183,  0.0280, -0.0278],
              [-0.0338, -0.1121, -0.0628]],
    
             [[-0.0242, -0.0331, -0.0384],
              [-0.0234, -0.0100, -0.0630],
              [ 0.0317,  0.0313, -0.0515]],
    
             ...,
    
             [[-0.0236, -0.0411,  0.0166],
              [ 0.0699,  0.0918,  0.0101],
              [-0.0005, -0.0006, -0.0425]],
    
             [[-0.0410,  0.0628, -0.0840],
              [ 0.0098,  0.0228, -0.0583],
              [-0.0094,  0.0215, -0.0637]],
    
             [[ 0.0215,  0.0117, -0.0682],
              [-0.0111,  0.0199,  0.0780],
              [ 0.0050,  0.0571,  0.0253]]],


​    
            ...,


​    
            [[[-0.0746, -0.0486, -0.0010],
              [ 0.0341,  0.0851, -0.0946],
              [ 0.0124,  0.0472, -0.0573]],
    
             [[-0.0189,  0.0290, -0.0303],
              [-0.0232, -0.0205, -0.0168],
              [-0.0034,  0.0630,  0.0066]],
    
             [[-0.0389, -0.0413, -0.0489],
              [-0.0304, -0.0109, -0.0292],
              [ 0.0476,  0.0005,  0.0348]],
    
             ...,
    
             [[ 0.0478,  0.0152,  0.0667],
              [ 0.0524, -0.0323,  0.0056],
              [-0.0133, -0.0292,  0.0614]],
    
             [[ 0.0556, -0.0114,  0.0356],
              [-0.0693,  0.0634, -0.0174],
              [ 0.0692,  0.0518, -0.0460]],
    
             [[-0.0132,  0.0179, -0.0121],
              [-0.0056,  0.0573, -0.0743],
              [-0.0128, -0.0058, -0.0049]]],


​    
            [[[ 0.0172,  0.0307,  0.0437],
              [-0.0358, -0.0098,  0.0533],
              [-0.0702, -0.0728,  0.0780]],
    
             [[ 0.0749, -0.0362, -0.0053],
              [ 0.0096, -0.0204, -0.0239],
              [-0.0154, -0.0101, -0.0086]],
    
             [[ 0.0047,  0.0374, -0.0289],
              [-0.0600,  0.0487, -0.0130],
              [-0.0032, -0.0242,  0.0271]],
    
             ...,
    
             [[-0.0029,  0.0010, -0.0515],
              [ 0.0176, -0.0491, -0.0399],
              [-0.0052,  0.0752, -0.0279]],
    
             [[ 0.0449,  0.0155, -0.0454],
              [ 0.0128,  0.0712,  0.0472],
              [-0.0417,  0.0190,  0.0454]],
    
             [[-0.0674,  0.0464,  0.0473],
              [ 0.0133, -0.0986, -0.0194],
              [ 0.0300,  0.0219, -0.0223]]],


​    
            [[[ 0.0609, -0.0621,  0.0276],
              [ 0.0091, -0.0020, -0.0011],
              [ 0.0309, -0.0084, -0.0435]],
    
             [[ 0.0111,  0.0236,  0.0367],
              [ 0.0792,  0.0743, -0.0432],
              [-0.0540,  0.0395,  0.0420]],
    
             [[-0.0225, -0.0245, -0.0029],
              [-0.0392,  0.0383,  0.0899],
              [-0.0118,  0.0049, -0.0263]],
    
             ...,
    
             [[ 0.1031,  0.0167, -0.0020],
              [-0.0125, -0.0907,  0.0373],
              [ 0.0090, -0.0008,  0.0524]],
    
             [[ 0.0812,  0.0085, -0.0226],
              [ 0.0177, -0.0148, -0.0286],
              [-0.0171, -0.0206,  0.0571]],
    
             [[-0.0742,  0.0241,  0.0427],
              [-0.0483, -0.0376, -0.0237],
              [-0.0554, -0.0395, -0.0414]]]])), ('layer2.0.bn1.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1.])), ('layer2.0.bn1.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer2.0.bn1.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer2.0.bn1.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1.])), ('layer2.0.bn1.num_batches_tracked', tensor(0)), ('layer2.0.conv2.weight', tensor([[[[ 0.0540,  0.0808, -0.0557],
              [-0.0042, -0.0145,  0.0696],
              [-0.0208,  0.0225, -0.0438]],
    
             [[ 0.0146,  0.0077, -0.0104],
              [ 0.0063,  0.0570, -0.0525],
              [-0.0059,  0.0452, -0.0325]],
    
             [[ 0.0307,  0.0341, -0.0237],
              [ 0.0053, -0.0322,  0.0116],
              [ 0.0380, -0.0227,  0.0056]],
    
             ...,
    
             [[ 0.0728, -0.0403,  0.0429],
              [ 0.0005,  0.0043,  0.0282],
              [-0.0084, -0.0714, -0.0208]],
    
             [[-0.0540,  0.0761,  0.0295],
              [-0.0189,  0.0028, -0.0063],
              [-0.0500,  0.0112,  0.0140]],
    
             [[ 0.0347,  0.0827,  0.0492],
              [-0.0118, -0.0020,  0.0466],
              [ 0.0434,  0.0436, -0.0186]]],


​    
            [[[ 0.0460,  0.0043,  0.0196],
              [-0.0271, -0.0468,  0.0041],
              [ 0.0331, -0.0697,  0.0376]],
    
             [[ 0.0281, -0.0401,  0.0246],
              [-0.0353, -0.0218,  0.0143],
              [ 0.0669,  0.0624,  0.0319]],
    
             [[-0.0116,  0.0075, -0.0165],
              [ 0.0110, -0.0511,  0.0491],
              [ 0.0134,  0.0530,  0.0903]],
    
             ...,
    
             [[-0.0339,  0.0166,  0.0286],
              [ 0.0027,  0.0117,  0.0407],
              [-0.0431, -0.0342,  0.0097]],
    
             [[-0.0032,  0.0125, -0.0275],
              [-0.0431,  0.0234, -0.0412],
              [ 0.0423,  0.0734, -0.0414]],
    
             [[-0.0598,  0.0072,  0.0379],
              [ 0.0426, -0.0440, -0.0191],
              [-0.0481,  0.0893,  0.0237]]],


​    
            [[[-0.0825,  0.0553,  0.0074],
              [-0.0255, -0.0539,  0.0232],
              [ 0.0644, -0.0174, -0.0372]],
    
             [[ 0.0341, -0.0136,  0.0040],
              [ 0.0033, -0.0074,  0.0289],
              [ 0.0321,  0.0334,  0.0246]],
    
             [[ 0.0643,  0.0417,  0.0225],
              [ 0.0257, -0.0056,  0.0148],
              [ 0.0348,  0.0281, -0.0416]],
    
             ...,
    
             [[ 0.0449,  0.0257, -0.0047],
              [-0.0270,  0.0014, -0.0060],
              [ 0.0515, -0.0391, -0.0946]],
    
             [[ 0.0207,  0.0787,  0.0350],
              [-0.0195,  0.0555,  0.0372],
              [ 0.0180,  0.0108, -0.0047]],
    
             [[-0.0596, -0.0661, -0.0033],
              [ 0.0371,  0.0503, -0.0218],
              [-0.0576, -0.0514,  0.0902]]],


​    
            ...,


​    
            [[[ 0.0294,  0.0230, -0.0115],
              [-0.0338, -0.0647, -0.0426],
              [-0.0279, -0.0551,  0.0729]],
    
             [[ 0.0125,  0.0363,  0.0218],
              [ 0.0022, -0.0080, -0.0459],
              [-0.0155, -0.0217, -0.0062]],
    
             [[ 0.0237, -0.0554,  0.0558],
              [-0.0203,  0.0602, -0.0062],
              [ 0.0857,  0.0023,  0.0523]],
    
             ...,
    
             [[ 0.0596, -0.0441,  0.0076],
              [-0.0520, -0.0061,  0.0128],
              [ 0.0390,  0.0791,  0.0416]],
    
             [[-0.0093, -0.0717, -0.0024],
              [-0.0657, -0.0172, -0.0540],
              [ 0.0390,  0.0569, -0.0246]],
    
             [[-0.0669, -0.0047, -0.0136],
              [-0.0264,  0.0379,  0.0256],
              [ 0.0443, -0.0414,  0.0119]]],


​    
            [[[-0.0158,  0.0465, -0.0227],
              [-0.0108, -0.0593,  0.0290],
              [-0.0309,  0.0075, -0.0199]],
    
             [[-0.0493, -0.0702, -0.0206],
              [-0.0124,  0.0799,  0.0100],
              [-0.0214,  0.0253, -0.0078]],
    
             [[-0.0163,  0.0854,  0.0402],
              [ 0.0191, -0.0416,  0.0141],
              [ 0.0074, -0.0067,  0.0804]],
    
             ...,
    
             [[ 0.0352,  0.0655, -0.0062],
              [ 0.0447,  0.0479, -0.0708],
              [-0.0972, -0.0279,  0.0688]],
    
             [[ 0.0050, -0.0125,  0.0006],
              [-0.0513,  0.0188,  0.0887],
              [-0.0286, -0.0418, -0.0104]],
    
             [[-0.0491,  0.1084,  0.0515],
              [-0.0180, -0.0015,  0.0720],
              [ 0.0138, -0.0039, -0.0229]]],


​    
            [[[-0.0086, -0.0610,  0.0271],
              [ 0.0088,  0.0534, -0.0652],
              [ 0.0101, -0.0364,  0.0920]],
    
             [[ 0.0252, -0.0443, -0.0188],
              [ 0.0025, -0.0267, -0.0080],
              [-0.0067, -0.0207, -0.0606]],
    
             [[-0.0613,  0.0134,  0.0378],
              [ 0.0246,  0.0262, -0.0212],
              [ 0.0537,  0.0398, -0.0308]],
    
             ...,
    
             [[-0.0633, -0.0193, -0.0111],
              [-0.0126,  0.0047,  0.0053],
              [-0.0018,  0.0107, -0.0034]],
    
             [[ 0.0355, -0.0341, -0.0109],
              [-0.0062,  0.0130,  0.0540],
              [-0.0594, -0.0286, -0.0381]],
    
             [[ 0.0183,  0.0292, -0.0305],
              [-0.0375,  0.0597,  0.0681],
              [-0.0246, -0.0031,  0.0534]]]])), ('layer2.0.bn2.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1.])), ('layer2.0.bn2.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer2.0.bn2.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer2.0.bn2.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1.])), ('layer2.0.bn2.num_batches_tracked', tensor(0)), ('layer2.0.downsample.0.weight', tensor([[[[ 0.1134]],
    
             [[-0.1649]],
    
             [[-0.2037]],
    
             ...,
    
             [[ 0.1360]],
    
             [[-0.0981]],
    
             [[ 0.0617]]],


​    
            [[[ 0.0900]],
    
             [[-0.1207]],
    
             [[-0.2714]],
    
             ...,
    
             [[-0.1491]],
    
             [[ 0.1718]],
    
             [[ 0.0035]]],


​    
            [[[-0.1024]],
    
             [[-0.0853]],
    
             [[ 0.1771]],
    
             ...,
    
             [[-0.0016]],
    
             [[-0.1849]],
    
             [[ 0.0911]]],


​    
            ...,


​    
            [[[-0.1319]],
    
             [[ 0.0694]],
    
             [[-0.1359]],
    
             ...,
    
             [[ 0.0161]],
    
             [[ 0.1369]],
    
             [[ 0.1154]]],


​    
            [[[-0.1115]],
    
             [[ 0.1137]],
    
             [[-0.2520]],
    
             ...,
    
             [[ 0.0064]],
    
             [[ 0.0804]],
    
             [[-0.1589]]],


​    
            [[[ 0.0434]],
    
             [[ 0.1527]],
    
             [[-0.1698]],
    
             ...,
    
             [[ 0.0994]],
    
             [[ 0.0780]],
    
             [[ 0.0740]]]])), ('layer2.0.downsample.1.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1.])), ('layer2.0.downsample.1.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer2.0.downsample.1.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer2.0.downsample.1.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1.])), ('layer2.0.downsample.1.num_batches_tracked', tensor(0)), ('layer2.1.conv1.weight', tensor([[[[ 9.4917e-03, -4.3838e-02, -1.4113e-02],
              [-1.9540e-02, -1.5401e-02,  1.4366e-02],
              [-7.2431e-02,  1.1573e-02, -1.3649e-02]],
    
             [[ 3.1076e-02,  1.4257e-02,  3.6470e-03],
              [-4.7784e-02, -5.1621e-02,  3.6865e-02],
              [-3.6935e-02, -2.3126e-02,  2.8439e-02]],
    
             [[ 1.4482e-02,  1.5604e-02, -5.7814e-03],
              [ 2.0195e-02,  3.4757e-03, -8.1251e-02],
              [ 2.8865e-03, -4.2297e-02,  6.1348e-02]],
    
             ...,
    
             [[ 6.1181e-02,  3.1304e-02,  1.2904e-02],
              [-2.4378e-02, -4.7457e-02,  2.4194e-02],
              [ 2.4862e-02, -5.0659e-02, -9.5623e-02]],
    
             [[-3.2832e-02, -4.2360e-02,  2.1370e-02],
              [-1.7944e-03, -1.2385e-01, -4.8749e-02],
              [-3.2802e-02,  8.7864e-02, -2.3640e-02]],
    
             [[ 6.9454e-02, -2.9245e-02, -4.7851e-02],
              [ 3.1639e-02,  1.2180e-02, -5.6808e-02],
              [ 1.1535e-02, -4.1574e-02, -1.1260e-02]]],


​    
            [[[-5.7193e-02,  7.4393e-03,  2.2646e-02],
              [-1.6073e-02, -6.0812e-02,  3.1450e-02],
              [ 1.1325e-02,  7.1660e-03,  1.9514e-02]],
    
             [[-3.6434e-03,  5.9549e-02, -1.9878e-02],
              [ 4.5325e-02,  1.5327e-02,  3.3561e-02],
              [-3.9024e-02,  6.6292e-02, -3.1064e-03]],
    
             [[-5.6671e-03, -1.0653e-02,  1.0467e-01],
              [ 4.3120e-02, -2.2607e-02, -7.7391e-02],
              [ 6.2994e-03, -1.5461e-02, -3.6156e-02]],
    
             ...,
    
             [[ 3.7762e-02,  2.3886e-03, -7.0734e-02],
              [-4.2752e-02,  4.1623e-02,  1.5848e-02],
              [ 1.6811e-02, -8.4648e-02, -8.8035e-03]],
    
             [[ 3.5259e-02,  5.1821e-02, -7.0861e-02],
              [-2.0294e-02,  1.6550e-02,  2.0257e-03],
              [ 6.0949e-02,  1.2421e-02,  7.3805e-02]],
    
             [[ 4.3864e-02, -2.3545e-02,  2.6641e-02],
              [-1.3562e-02, -2.0005e-02, -2.2738e-02],
              [-6.9720e-03,  4.0579e-02,  6.4031e-02]]],


​    
            [[[ 5.4271e-02, -1.2097e-02,  9.9753e-02],
              [ 7.4491e-02,  5.3236e-02,  1.0788e-02],
              [ 4.6727e-03, -1.3132e-02, -5.1397e-03]],
    
             [[ 6.5068e-02, -9.5091e-03, -4.7880e-02],
              [-1.8116e-02,  5.0310e-02, -4.3630e-03],
              [ 3.4612e-03, -4.3647e-02,  1.3044e-02]],
    
             [[ 2.4180e-03,  2.5471e-02,  3.7343e-02],
              [-1.7611e-02, -5.6464e-02, -3.4999e-02],
              [-2.7549e-02, -5.7016e-03, -4.2026e-02]],
    
             ...,
    
             [[-7.9049e-03, -3.4917e-02, -5.0150e-04],
              [-6.1644e-02,  2.9234e-02,  2.4467e-02],
              [ 6.4167e-03,  2.9870e-02,  7.5125e-02]],
    
             [[ 7.6612e-02,  1.1932e-02, -1.4564e-02],
              [-4.4840e-02,  8.0319e-03,  4.2495e-02],
              [-4.8409e-02,  4.5992e-02,  2.3031e-02]],
    
             [[ 3.2587e-02, -5.6621e-02,  6.2170e-02],
              [-3.2940e-02, -1.6148e-02, -7.8749e-03],
              [ 1.5296e-02,  6.6066e-03,  2.1501e-02]]],


​    
            ...,


​    
            [[[ 4.3392e-02,  3.1892e-02, -6.0912e-02],
              [ 3.2236e-02, -6.1438e-02, -4.4012e-02],
              [-3.4353e-02,  6.7961e-02, -5.4611e-02]],
    
             [[ 1.8713e-02, -9.7891e-02, -5.6852e-02],
              [ 2.9484e-02, -4.0038e-02,  5.6397e-02],
              [ 2.2133e-02, -3.3515e-02,  3.2406e-02]],
    
             [[-2.7721e-02,  2.2127e-02,  2.9530e-02],
              [-2.6102e-02, -3.8631e-02,  6.8731e-02],
              [ 1.9735e-02,  2.3008e-02, -2.3933e-02]],
    
             ...,
    
             [[ 4.1398e-02,  2.2786e-02,  2.7265e-03],
              [ 1.0733e-02,  3.9280e-02, -2.9558e-03],
              [-5.1938e-02, -1.9259e-02,  4.2349e-02]],
    
             [[ 7.5985e-03, -9.4925e-02,  2.1317e-02],
              [-1.9697e-02,  3.9288e-02,  1.6268e-02],
              [-8.2106e-02, -5.6089e-03,  9.8829e-02]],
    
             [[ 2.0950e-03, -2.4346e-02,  3.8180e-02],
              [-4.8120e-03,  3.7703e-03,  3.2822e-02],
              [-2.1882e-02, -8.5669e-02, -5.5339e-02]]],


​    
            [[[-3.9782e-02, -2.8178e-02,  2.1350e-02],
              [-1.5101e-02, -6.2741e-02, -4.7504e-02],
              [ 1.9134e-02, -3.2309e-02,  3.7014e-02]],
    
             [[-4.6494e-02,  5.6103e-02,  1.2124e-03],
              [ 1.2678e-02, -2.2464e-02,  3.6343e-02],
              [ 1.7750e-02,  5.7882e-02, -3.4187e-02]],
    
             [[-4.0532e-02, -4.7067e-02, -2.5017e-02],
              [ 3.1092e-02, -2.5320e-02, -4.8343e-02],
              [-7.0592e-03,  6.9279e-02,  4.1107e-03]],
    
             ...,
    
             [[-5.4115e-03, -4.6132e-02, -4.2962e-02],
              [ 2.1316e-02, -2.9461e-02,  8.0669e-02],
              [ 7.3475e-03, -6.2416e-02,  5.8797e-02]],
    
             [[ 2.8009e-02,  9.4438e-02,  2.4128e-02],
              [-5.1240e-03,  3.9849e-02, -1.9139e-05],
              [ 9.9925e-03,  2.3025e-02, -3.0954e-02]],
    
             [[ 1.6193e-02, -5.7257e-02,  4.7540e-03],
              [-5.2892e-02, -2.7952e-02, -2.2088e-02],
              [-2.2044e-02, -5.4004e-02,  5.4337e-02]]],


​    
            [[[-2.6053e-02,  9.5196e-03, -2.1971e-02],
              [ 7.5675e-02, -5.6186e-02, -7.1327e-02],
              [-7.3842e-04, -2.4744e-02,  3.8442e-02]],
    
             [[ 2.0697e-03,  4.5354e-02,  6.5955e-02],
              [ 7.3361e-03,  1.9311e-02,  2.2453e-03],
              [ 1.1895e-02,  1.2448e-02, -1.5129e-02]],
    
             [[-1.0624e-02,  4.9166e-02,  3.1875e-02],
              [ 4.2217e-02,  1.3336e-02, -2.4965e-02],
              [-1.5078e-02, -4.1329e-02,  1.7680e-03]],
    
             ...,
    
             [[ 2.1686e-02, -8.3606e-03,  3.4883e-02],
              [-2.4252e-02, -8.9345e-03,  6.1014e-02],
              [-1.0333e-02, -2.7579e-02,  3.4201e-02]],
    
             [[ 1.1051e-01,  3.1364e-02, -4.1041e-02],
              [-1.1251e-02, -5.9290e-02,  3.4159e-02],
              [-7.5320e-03,  4.0232e-02, -4.2174e-02]],
    
             [[-3.2418e-03, -8.3922e-03,  8.1281e-02],
              [-6.7691e-02,  5.3527e-02, -1.5334e-02],
              [-2.7017e-02,  1.2073e-02,  3.9451e-02]]]])), ('layer2.1.bn1.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1.])), ('layer2.1.bn1.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer2.1.bn1.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer2.1.bn1.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1.])), ('layer2.1.bn1.num_batches_tracked', tensor(0)), ('layer2.1.conv2.weight', tensor([[[[ 0.0438,  0.0212,  0.0536],
              [-0.0553, -0.0061,  0.0488],
              [ 0.0429,  0.0411,  0.0124]],
    
             [[ 0.0867,  0.0072,  0.0142],
              [ 0.0055,  0.0552, -0.0237],
              [ 0.0047,  0.0041,  0.0014]],
    
             [[-0.0873, -0.1168,  0.0350],
              [ 0.0639, -0.0410, -0.0236],
              [ 0.0454,  0.0339,  0.0153]],
    
             ...,
    
             [[ 0.0595, -0.0314,  0.0183],
              [ 0.0088,  0.0639, -0.0579],
              [ 0.0012, -0.0317,  0.0295]],
    
             [[ 0.0010, -0.0198,  0.0331],
              [-0.1408,  0.0007,  0.0637],
              [-0.0242,  0.0030,  0.0096]],
    
             [[ 0.0049, -0.0033,  0.0685],
              [ 0.0282, -0.0911,  0.0314],
              [-0.0009, -0.0623,  0.0361]]],


​    
            [[[-0.0130,  0.0253, -0.0279],
              [ 0.0479, -0.0155,  0.0235],
              [ 0.0929, -0.0080,  0.0621]],
    
             [[ 0.0703,  0.0640, -0.0015],
              [ 0.0293, -0.0201,  0.0015],
              [-0.0222, -0.0073,  0.0475]],
    
             [[ 0.0537, -0.0159,  0.0414],
              [-0.0113, -0.0737,  0.0194],
              [-0.0251, -0.0452,  0.0056]],
    
             ...,
    
             [[ 0.0374,  0.0207, -0.0172],
              [-0.0302, -0.0282, -0.0555],
              [-0.0704,  0.0335,  0.0391]],
    
             [[-0.0483,  0.0278, -0.0649],
              [-0.0218,  0.0291,  0.0120],
              [-0.0715, -0.0882, -0.0135]],
    
             [[-0.0408,  0.0279, -0.0953],
              [-0.0277, -0.0323, -0.0265],
              [-0.0082,  0.0475,  0.0367]]],


​    
            [[[ 0.0643,  0.0171, -0.0050],
              [ 0.0072,  0.0043,  0.0748],
              [-0.0254, -0.1025, -0.0675]],
    
             [[ 0.0136, -0.0239,  0.0070],
              [-0.0154, -0.0906, -0.0549],
              [ 0.0133, -0.0315, -0.0086]],
    
             [[-0.0007,  0.0256,  0.0499],
              [ 0.0102, -0.0533,  0.0108],
              [ 0.0190, -0.0124, -0.0424]],
    
             ...,
    
             [[ 0.0334,  0.0582,  0.0360],
              [ 0.0600, -0.0246,  0.0014],
              [-0.0664, -0.0340, -0.0272]],
    
             [[ 0.0595,  0.0349, -0.0132],
              [ 0.0824, -0.0058,  0.0064],
              [-0.0066,  0.0201, -0.0285]],
    
             [[ 0.0537,  0.0192,  0.0188],
              [ 0.0184,  0.0452,  0.0640],
              [-0.0817,  0.0401, -0.0109]]],


​    
            ...,


​    
            [[[-0.0428, -0.0149, -0.0246],
              [ 0.0046,  0.0200, -0.0761],
              [-0.0081,  0.0070,  0.0307]],
    
             [[-0.0494,  0.0473,  0.0065],
              [-0.0317, -0.0046,  0.0469],
              [ 0.0110, -0.0626, -0.0298]],
    
             [[ 0.0476, -0.0788, -0.0107],
              [-0.0166,  0.0018, -0.0068],
              [ 0.0084,  0.0426,  0.0553]],
    
             ...,
    
             [[ 0.0197,  0.0296, -0.0125],
              [ 0.0059, -0.0097, -0.0440],
              [-0.0721,  0.0200,  0.1105]],
    
             [[ 0.0202,  0.0191,  0.0226],
              [-0.0082, -0.0265,  0.0410],
              [-0.0283,  0.0376, -0.0068]],
    
             [[ 0.0086,  0.0258, -0.0505],
              [ 0.0324, -0.0182, -0.0452],
              [ 0.0141, -0.0192, -0.0145]]],


​    
            [[[ 0.0459, -0.0163,  0.0096],
              [ 0.0127,  0.0464,  0.0216],
              [ 0.0046,  0.0333, -0.0478]],
    
             [[ 0.0362,  0.0332,  0.0251],
              [ 0.0559,  0.0016, -0.0122],
              [-0.0081,  0.0381,  0.0250]],
    
             [[ 0.0286, -0.0459,  0.0419],
              [ 0.0129, -0.0341, -0.0141],
              [ 0.0174, -0.0138, -0.0706]],
    
             ...,
    
             [[ 0.0469, -0.0872, -0.0281],
              [-0.0472, -0.0288, -0.0116],
              [-0.0058,  0.0350, -0.0293]],
    
             [[-0.0359,  0.0015, -0.0225],
              [ 0.0304,  0.0128,  0.0108],
              [ 0.0566, -0.0777,  0.0529]],
    
             [[ 0.0267,  0.0671, -0.0195],
              [-0.0036, -0.0272,  0.0465],
              [ 0.0214, -0.0128,  0.0035]]],


​    
            [[[-0.0081,  0.0398, -0.0227],
              [-0.0413,  0.0186, -0.0158],
              [ 0.0049, -0.0042, -0.0161]],
    
             [[-0.0488, -0.0421, -0.0690],
              [ 0.0391,  0.0251,  0.0164],
              [ 0.0043,  0.0059,  0.0018]],
    
             [[ 0.0076,  0.1163,  0.0076],
              [ 0.0276, -0.0786, -0.0247],
              [ 0.0491, -0.0236,  0.0099]],
    
             ...,
    
             [[-0.0124,  0.0348, -0.0156],
              [-0.0292, -0.0776, -0.0081],
              [ 0.0320, -0.0436,  0.0371]],
    
             [[ 0.0447,  0.0725,  0.0158],
              [-0.1048, -0.0343,  0.0236],
              [-0.0035, -0.0635,  0.0495]],
    
             [[-0.0558, -0.0184,  0.0068],
              [ 0.0774,  0.0130, -0.0256],
              [ 0.0515, -0.0177, -0.0094]]]])), ('layer2.1.bn2.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1.])), ('layer2.1.bn2.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer2.1.bn2.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer2.1.bn2.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1.])), ('layer2.1.bn2.num_batches_tracked', tensor(0)), ('layer3.0.conv1.weight', tensor([[[[ 0.0349,  0.0086,  0.0096],
              [ 0.0413, -0.0102,  0.0055],
              [-0.0163, -0.0677,  0.0016]],
    
             [[ 0.0014, -0.0110,  0.0038],
              [ 0.0337,  0.0020, -0.0030],
              [-0.0236,  0.0057, -0.0509]],
    
             [[ 0.0155,  0.0119,  0.0056],
              [-0.0105, -0.0323,  0.0536],
              [-0.0747, -0.0145, -0.0404]],
    
             ...,
    
             [[ 0.0295, -0.0132,  0.0087],
              [ 0.0296,  0.0195,  0.0187],
              [-0.0496,  0.0062, -0.0463]],
    
             [[-0.0356,  0.0047, -0.0013],
              [ 0.0156, -0.0075, -0.0235],
              [-0.0105, -0.0305,  0.0782]],
    
             [[-0.0231,  0.0091,  0.0305],
              [ 0.0142,  0.0132,  0.0348],
              [ 0.0044,  0.0219,  0.0029]]],


​    
            [[[-0.0212,  0.0364, -0.0290],
              [ 0.0127,  0.0041, -0.0074],
              [-0.0006,  0.0569, -0.0181]],
    
             [[-0.0113,  0.0053, -0.0675],
              [-0.0503, -0.0165, -0.0439],
              [-0.0322, -0.0382, -0.0123]],
    
             [[ 0.0327,  0.0066, -0.0186],
              [-0.0042, -0.0269, -0.0184],
              [-0.0141,  0.0079,  0.0137]],
    
             ...,
    
             [[-0.0125, -0.0250, -0.0081],
              [-0.0542,  0.0288,  0.0271],
              [-0.0183,  0.0235,  0.0012]],
    
             [[ 0.0596, -0.0349,  0.0526],
              [ 0.0047,  0.0208, -0.0436],
              [ 0.0365,  0.0079, -0.0054]],
    
             [[ 0.0479,  0.0087, -0.0030],
              [-0.0075,  0.0429, -0.0259],
              [-0.0032, -0.0156, -0.0009]]],


​    
            [[[-0.0249,  0.0367,  0.0297],
              [ 0.0061, -0.0402, -0.0070],
              [-0.0449, -0.0183, -0.0054]],
    
             [[ 0.0308,  0.0283, -0.0199],
              [ 0.0424, -0.0101,  0.0193],
              [ 0.0449,  0.0070,  0.0582]],
    
             [[-0.0426, -0.0077, -0.0369],
              [ 0.0001, -0.0265, -0.0589],
              [-0.0601, -0.0479, -0.0013]],
    
             ...,
    
             [[-0.0179, -0.0244, -0.0579],
              [-0.0459, -0.0029, -0.0151],
              [ 0.0263, -0.0004, -0.0187]],
    
             [[ 0.0074, -0.0004,  0.0086],
              [ 0.0284,  0.0654, -0.0165],
              [ 0.0116, -0.0059,  0.0304]],
    
             [[ 0.0535, -0.0324, -0.0140],
              [-0.0323,  0.0213,  0.0131],
              [-0.0326, -0.0430,  0.0530]]],


​    
            ...,


​    
            [[[ 0.0449, -0.0052, -0.0313],
              [-0.0396,  0.0049, -0.0056],
              [-0.0410, -0.0122, -0.0070]],
    
             [[ 0.0247,  0.0044, -0.0206],
              [ 0.0302, -0.0333,  0.0366],
              [ 0.0454,  0.0860, -0.0144]],
    
             [[-0.0065, -0.0059, -0.0134],
              [-0.0098,  0.0045, -0.0063],
              [ 0.0162,  0.0272,  0.0029]],
    
             ...,
    
             [[ 0.0081, -0.0118, -0.0031],
              [ 0.0490, -0.0305,  0.0092],
              [-0.0716, -0.0051,  0.0091]],
    
             [[-0.0138,  0.0322,  0.0029],
              [-0.0223,  0.0339,  0.0149],
              [ 0.0173, -0.0205, -0.0313]],
    
             [[-0.0080, -0.0018, -0.0041],
              [ 0.0237,  0.0120, -0.0249],
              [-0.0533, -0.0087,  0.0407]]],


​    
            [[[-0.0691, -0.0210,  0.0125],
              [ 0.0003,  0.0235, -0.0084],
              [ 0.0596, -0.0081, -0.0231]],
    
             [[ 0.0002,  0.0441,  0.0161],
              [ 0.0233, -0.0274,  0.0003],
              [-0.0047,  0.0399,  0.0414]],
    
             [[ 0.0064, -0.0306,  0.0459],
              [-0.0374,  0.0177, -0.0209],
              [-0.0426, -0.0197, -0.0247]],
    
             ...,
    
             [[-0.0193,  0.0245,  0.0153],
              [-0.0099, -0.0507, -0.0386],
              [ 0.0577, -0.0096,  0.0134]],
    
             [[-0.0476,  0.0122, -0.0419],
              [ 0.0218,  0.0256,  0.0191],
              [-0.0145, -0.0224, -0.0050]],
    
             [[ 0.0110,  0.0047, -0.0037],
              [ 0.0061,  0.0668,  0.0475],
              [ 0.0066, -0.0046, -0.0025]]],


​    
            [[[ 0.0083, -0.0538, -0.0203],
              [-0.0113, -0.0147, -0.0328],
              [ 0.0206, -0.0062, -0.0044]],
    
             [[ 0.0410,  0.0431,  0.0183],
              [-0.0123,  0.0068,  0.0033],
              [-0.0031,  0.0313, -0.0670]],
    
             [[ 0.0212, -0.0293, -0.0204],
              [ 0.0063,  0.0166, -0.0234],
              [-0.0406, -0.0198, -0.0353]],
    
             ...,
    
             [[-0.0170, -0.0230, -0.0056],
              [ 0.0055,  0.0298,  0.0040],
              [-0.0243, -0.0092,  0.0778]],
    
             [[-0.0132,  0.0287,  0.0016],
              [ 0.0520,  0.0449,  0.0065],
              [ 0.0165, -0.0415,  0.0275]],
    
             [[-0.0423, -0.0387, -0.0262],
              [ 0.0171, -0.0358, -0.0033],
              [ 0.0362,  0.0118,  0.0392]]]])), ('layer3.0.bn1.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1.])), ('layer3.0.bn1.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer3.0.bn1.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer3.0.bn1.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1.])), ('layer3.0.bn1.num_batches_tracked', tensor(0)), ('layer3.0.conv2.weight', tensor([[[[-0.0204,  0.0307, -0.0113],
              [ 0.0715, -0.0123,  0.0073],
              [ 0.0461, -0.0622, -0.0495]],
    
             [[-0.0041, -0.0623, -0.0655],
              [-0.0062, -0.0174,  0.0399],
              [-0.0096,  0.0002, -0.0127]],
    
             [[ 0.0214, -0.0274,  0.0407],
              [ 0.0415, -0.0221,  0.0415],
              [ 0.0362,  0.0206,  0.0486]],
    
             ...,
    
             [[-0.0109,  0.0002,  0.0343],
              [ 0.0070,  0.0327,  0.0177],
              [ 0.0101, -0.0220,  0.0310]],
    
             [[ 0.0088,  0.0096, -0.0066],
              [-0.0532, -0.0274, -0.0142],
              [ 0.0370, -0.0454, -0.0089]],
    
             [[-0.0248, -0.0109, -0.0664],
              [ 0.0129, -0.0072,  0.0324],
              [-0.0111,  0.0036,  0.0151]]],


​    
            [[[-0.0192, -0.0985, -0.0125],
              [-0.0136, -0.0548, -0.0440],
              [-0.0898, -0.0056, -0.0413]],
    
             [[ 0.0045,  0.0264,  0.0087],
              [ 0.0075, -0.0535, -0.0234],
              [ 0.0048,  0.0244, -0.0081]],
    
             [[ 0.0031,  0.0129, -0.0103],
              [ 0.0397,  0.0222,  0.0207],
              [-0.0562, -0.1118, -0.0240]],
    
             ...,
    
             [[ 0.0356, -0.0236,  0.0706],
              [ 0.0396,  0.0216, -0.0232],
              [-0.0299, -0.0489,  0.0286]],
    
             [[-0.0415, -0.0207, -0.0064],
              [-0.0407,  0.0791,  0.0062],
              [ 0.0288,  0.0222,  0.0014]],
    
             [[ 0.0111,  0.0380, -0.0231],
              [ 0.0161,  0.0108, -0.0158],
              [-0.0293,  0.0718, -0.0129]]],


​    
            [[[ 0.0088, -0.0482, -0.0320],
              [-0.0327,  0.0047, -0.0238],
              [ 0.0105,  0.0399,  0.0064]],
    
             [[ 0.0056, -0.0405, -0.0146],
              [ 0.0072, -0.0119,  0.0366],
              [ 0.0215,  0.0121, -0.0282]],
    
             [[-0.0020, -0.0566, -0.0365],
              [ 0.0665, -0.0455,  0.0041],
              [-0.0060, -0.0327,  0.0613]],
    
             ...,
    
             [[ 0.0061, -0.0231,  0.0126],
              [-0.0126,  0.0249, -0.0173],
              [ 0.0305, -0.0202, -0.0125]],
    
             [[ 0.0108,  0.0124, -0.0241],
              [-0.0519, -0.0344,  0.0101],
              [ 0.0030,  0.0403, -0.0448]],
    
             [[-0.0054, -0.0195, -0.0558],
              [-0.0163,  0.0378,  0.0286],
              [ 0.0061,  0.0207,  0.0359]]],


​    
            ...,


​    
            [[[ 0.0380, -0.0335, -0.0105],
              [ 0.0251,  0.0047,  0.0110],
              [ 0.0437,  0.0054,  0.0125]],
    
             [[ 0.0091,  0.0064, -0.0246],
              [-0.0438,  0.0140,  0.0633],
              [ 0.0193,  0.0032, -0.0254]],
    
             [[-0.0193,  0.0379,  0.0345],
              [ 0.0015,  0.0637,  0.0273],
              [ 0.0088,  0.0133,  0.0551]],
    
             ...,
    
             [[-0.0201,  0.0015, -0.0151],
              [ 0.0344, -0.0493, -0.0246],
              [-0.0080,  0.0391, -0.0078]],
    
             [[ 0.0100, -0.0149,  0.0163],
              [-0.0002,  0.0105,  0.0341],
              [-0.0005, -0.0172,  0.0095]],
    
             [[-0.0250, -0.0026,  0.0116],
              [ 0.0039, -0.0077, -0.0106],
              [-0.0030,  0.0147,  0.0239]]],


​    
            [[[-0.0267, -0.0428,  0.0060],
              [-0.0337,  0.0093,  0.0431],
              [-0.0431, -0.0147, -0.0194]],
    
             [[-0.0112, -0.0124, -0.0457],
              [ 0.0364,  0.0053, -0.0210],
              [ 0.0062, -0.0032, -0.0576]],
    
             [[ 0.0411, -0.0081,  0.0161],
              [ 0.0104, -0.0017,  0.0217],
              [ 0.0425, -0.0259, -0.0102]],
    
             ...,
    
             [[-0.0566,  0.0281,  0.0561],
              [ 0.0386, -0.0370,  0.0405],
              [ 0.0224,  0.0461,  0.0256]],
    
             [[ 0.0308,  0.0206, -0.0410],
              [-0.0365, -0.0139, -0.0191],
              [-0.0479,  0.0091,  0.0462]],
    
             [[ 0.0405,  0.0053, -0.0278],
              [ 0.0221, -0.0220, -0.0342],
              [-0.0027,  0.0194, -0.0425]]],


​    
            [[[ 0.0046,  0.0232, -0.0319],
              [-0.0342,  0.0621, -0.0501],
              [-0.0247, -0.0112,  0.0576]],
    
             [[-0.0053,  0.0199, -0.0020],
              [-0.0033,  0.0155, -0.0357],
              [ 0.0627,  0.0041,  0.0158]],
    
             [[-0.0559, -0.0015, -0.0111],
              [-0.0504, -0.0118,  0.0309],
              [-0.0191,  0.0127,  0.0020]],
    
             ...,
    
             [[-0.0005, -0.0383, -0.0425],
              [ 0.0177,  0.0012,  0.0175],
              [ 0.0022, -0.0020, -0.0347]],
    
             [[ 0.0148,  0.0054,  0.0406],
              [-0.0098,  0.0169, -0.0666],
              [-0.0345,  0.0198, -0.0046]],
    
             [[-0.0136, -0.0206,  0.0022],
              [-0.0020,  0.0172, -0.0251],
              [-0.0197,  0.0181, -0.0603]]]])), ('layer3.0.bn2.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1.])), ('layer3.0.bn2.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer3.0.bn2.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer3.0.bn2.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1.])), ('layer3.0.bn2.num_batches_tracked', tensor(0)), ('layer3.0.downsample.0.weight', tensor([[[[ 5.1302e-02]],
    
             [[-1.2627e-01]],
    
             [[-1.3083e-01]],
    
             ...,
    
             [[-1.6266e-02]],
    
             [[ 1.1629e-01]],
    
             [[ 1.3444e-01]]],


​    
            [[[-6.1614e-02]],
    
             [[-6.8547e-02]],
    
             [[ 5.8207e-02]],
    
             ...,
    
             [[ 1.1938e-02]],
    
             [[ 2.0041e-02]],
    
             [[ 2.1884e-04]]],


​    
            [[[ 8.9777e-03]],
    
             [[-4.8123e-02]],
    
             [[ 3.0489e-02]],
    
             ...,
    
             [[-4.8910e-02]],
    
             [[ 8.1343e-02]],
    
             [[-8.4297e-03]]],


​    
            ...,


​    
            [[[ 7.5705e-02]],
    
             [[ 1.9363e-01]],
    
             [[ 8.0216e-02]],
    
             ...,
    
             [[ 9.8609e-03]],
    
             [[-2.6596e-01]],
    
             [[-5.2704e-03]]],


​    
            [[[-1.1560e-01]],
    
             [[-1.1692e-01]],
    
             [[ 4.2977e-02]],
    
             ...,
    
             [[ 4.9820e-02]],
    
             [[-1.2323e-01]],
    
             [[ 1.6390e-01]]],


​    
            [[[ 4.4047e-02]],
    
             [[ 7.3217e-02]],
    
             [[ 2.5563e-01]],
    
             ...,
    
             [[-1.6249e-03]],
    
             [[-1.0374e-02]],
    
             [[-7.1804e-03]]]])), ('layer3.0.downsample.1.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1.])), ('layer3.0.downsample.1.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer3.0.downsample.1.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer3.0.downsample.1.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1.])), ('layer3.0.downsample.1.num_batches_tracked', tensor(0)), ('layer3.1.conv1.weight', tensor([[[[-3.4275e-03, -3.2765e-02, -4.8360e-02],
              [ 2.6047e-02,  2.4244e-02,  8.3237e-03],
              [ 8.2318e-03,  2.0020e-02, -2.9594e-04]],
    
             [[-1.2054e-03,  3.2403e-02,  2.0327e-02],
              [ 3.0869e-02,  2.3991e-02, -4.9800e-03],
              [-2.4696e-02, -2.1574e-02,  3.2120e-02]],
    
             [[ 2.4407e-03,  4.5500e-02,  1.7414e-02],
              [-1.5649e-02,  1.2575e-02,  2.1246e-02],
              [-2.9785e-02,  2.3647e-03, -4.6126e-03]],
    
             ...,
    
             [[-8.2805e-05, -3.3302e-02,  1.2111e-02],
              [-4.4140e-02,  2.9691e-02,  2.3848e-02],
              [ 1.4394e-02, -2.0125e-02, -2.3710e-02]],
    
             [[ 1.1411e-02, -2.1530e-02,  4.1833e-02],
              [-3.2720e-02, -2.6466e-03,  5.9094e-02],
              [ 1.2959e-02, -4.3469e-03, -1.8603e-02]],
    
             [[-4.6260e-03,  2.6415e-02,  4.3674e-02],
              [-5.0332e-02,  1.0870e-02,  2.7126e-02],
              [ 2.2240e-02,  5.7367e-02,  1.8207e-02]]],


​    
            [[[-2.3301e-02, -4.3126e-02, -2.3901e-02],
              [-1.3154e-02, -1.4424e-02, -1.7940e-02],
              [-3.6215e-02,  4.5187e-02, -5.8700e-03]],
    
             [[ 4.2955e-03,  4.3265e-02, -2.2313e-03],
              [ 7.0994e-02, -8.8121e-03,  1.3649e-02],
              [-1.0897e-02, -2.3306e-03,  5.2530e-02]],
    
             [[-1.2553e-02,  9.9124e-04, -7.9737e-02],
              [ 4.8922e-03,  3.4057e-02,  1.0713e-02],
              [ 7.8766e-02,  2.7916e-02, -3.0844e-02]],
    
             ...,
    
             [[ 1.7683e-02,  1.9431e-03,  1.0674e-02],
              [-2.0575e-02,  9.2218e-03, -2.6168e-03],
              [ 7.8172e-03, -4.8062e-02, -2.3314e-02]],
    
             [[ 3.6147e-02,  2.6981e-02, -2.2957e-04],
              [-1.5095e-02,  3.7287e-03,  1.0717e-03],
              [-1.3981e-02, -2.9080e-02, -9.6914e-03]],
    
             [[ 1.4152e-02,  4.3945e-02,  1.7273e-02],
              [-4.5323e-02,  4.5672e-03,  6.1765e-02],
              [-3.4651e-02,  1.6901e-02,  3.5999e-03]]],


​    
            [[[ 3.2745e-02, -2.1093e-02,  2.6111e-02],
              [ 3.9376e-02, -4.1046e-03, -6.1104e-03],
              [ 1.3050e-02,  6.6005e-02, -1.5202e-02]],
    
             [[-3.6875e-02,  3.7787e-02,  1.5600e-02],
              [-1.3169e-02, -3.4448e-03,  5.3856e-03],
              [-2.4479e-03,  1.6841e-02,  1.8229e-02]],
    
             [[-1.3233e-02, -3.7768e-02, -1.8962e-02],
              [-2.0258e-02, -9.4328e-03,  2.3798e-02],
              [-4.5409e-02,  8.9546e-03,  9.7676e-03]],
    
             ...,
    
             [[-3.8426e-02,  3.6415e-02, -2.1356e-02],
              [-6.9219e-02,  5.6381e-03, -1.0655e-02],
              [-5.3993e-02, -1.0081e-02,  1.2257e-02]],
    
             [[-6.5070e-02, -3.2924e-03,  5.2459e-02],
              [-2.5407e-02, -3.0754e-02,  5.7905e-03],
              [-3.2969e-02, -4.4555e-02, -1.8686e-02]],
    
             [[-1.5556e-02,  1.0232e-02, -1.9892e-02],
              [ 3.1916e-02, -5.5386e-02, -4.2912e-02],
              [ 3.9086e-02, -1.7610e-02,  3.8135e-02]]],


​    
            ...,


​    
            [[[ 8.9523e-03,  9.0386e-03,  3.7514e-04],
              [-2.3940e-03, -1.2129e-02,  1.1425e-02],
              [ 5.1709e-02,  1.4023e-02, -1.3509e-02]],
    
             [[-2.8031e-02,  2.5869e-02,  4.1954e-03],
              [-2.0250e-03,  8.6634e-03,  3.2324e-03],
              [ 6.4992e-02, -8.3147e-03, -4.1640e-03]],
    
             [[-1.9022e-02, -5.3721e-03, -4.4217e-02],
              [-2.2197e-02, -2.5634e-02, -8.3819e-03],
              [ 3.4498e-02, -3.6383e-02,  4.7910e-03]],
    
             ...,
    
             [[-1.2622e-02, -5.1117e-02, -6.9676e-03],
              [-3.2503e-02, -7.8702e-03, -2.7234e-02],
              [-1.7722e-02, -1.9462e-03, -3.1503e-02]],
    
             [[-1.8082e-02,  1.1581e-02, -2.7600e-03],
              [-4.6376e-02, -4.0566e-03,  5.9485e-02],
              [-4.0068e-02, -1.3325e-03,  3.5468e-02]],
    
             [[-2.7588e-02, -3.6860e-03,  2.1761e-02],
              [-1.0829e-02,  1.4175e-02,  9.4780e-03],
              [ 2.0903e-02,  1.3979e-02, -4.8911e-02]]],


​    
            [[[-4.0552e-02,  2.3846e-02,  4.9954e-02],
              [-1.7661e-02,  1.4004e-02, -2.9632e-02],
              [-3.1077e-02,  6.6514e-03,  2.4366e-02]],
    
             [[ 5.1884e-02,  3.1370e-02, -7.4215e-03],
              [-1.8851e-02,  2.6021e-03, -7.0751e-03],
              [-5.2249e-02, -1.9212e-02,  1.6598e-02]],
    
             [[ 2.6262e-02, -5.4647e-04,  2.4515e-02],
              [ 3.4015e-02,  1.3750e-02, -2.9688e-02],
              [ 2.9974e-02,  3.1654e-02,  2.4101e-02]],
    
             ...,
    
             [[ 8.5331e-03, -2.7333e-02,  2.1504e-02],
              [-2.8443e-02,  2.2886e-02,  5.2746e-02],
              [-3.3169e-02,  6.6165e-02,  1.9914e-02]],
    
             [[ 1.1873e-03, -2.0247e-03,  1.8708e-02],
              [ 1.6251e-02, -1.1317e-02, -2.6039e-02],
              [ 9.7906e-03,  2.3926e-02, -5.7490e-02]],
    
             [[ 6.5744e-02, -1.4836e-02, -3.7426e-02],
              [-8.7107e-03, -2.1662e-02,  6.9513e-03],
              [ 7.7147e-04,  2.2458e-02,  4.4468e-02]]],


​    
            [[[-3.1892e-02, -2.4033e-02,  2.3010e-03],
              [ 3.7565e-02,  1.0014e-02, -2.2165e-02],
              [-2.6159e-03,  2.6453e-02, -4.9073e-02]],
    
             [[ 2.9138e-02, -2.6143e-02, -1.7554e-02],
              [-2.6148e-03,  2.1903e-02, -5.5489e-03],
              [ 5.0457e-02,  2.1847e-02, -4.6261e-02]],
    
             [[-2.3481e-02, -1.0568e-03, -3.2478e-02],
              [-1.7053e-03,  2.4769e-02,  2.6193e-02],
              [ 1.6752e-02, -1.1499e-02,  4.0311e-02]],
    
             ...,
    
             [[-1.3489e-02,  5.3589e-02, -2.1646e-04],
              [ 2.0271e-02, -1.2201e-02,  1.7955e-02],
              [ 4.1370e-02,  2.8870e-02, -3.5185e-02]],
    
             [[-1.3736e-02,  8.2726e-03, -5.6179e-02],
              [ 2.3764e-02, -3.3681e-02,  2.2471e-02],
              [ 1.2665e-02,  3.0401e-02, -2.2962e-02]],
    
             [[-1.3585e-02, -5.9026e-03, -2.0017e-02],
              [ 1.4092e-02,  3.8301e-02, -2.9398e-02],
              [-5.5344e-03,  4.3024e-02,  1.2914e-02]]]])), ('layer3.1.bn1.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1.])), ('layer3.1.bn1.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer3.1.bn1.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer3.1.bn1.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1.])), ('layer3.1.bn1.num_batches_tracked', tensor(0)), ('layer3.1.conv2.weight', tensor([[[[ 3.0450e-02, -1.4764e-02,  1.2484e-02],
              [-3.1183e-02,  3.3724e-02,  1.9898e-02],
              [ 5.9339e-02, -1.7990e-03, -1.9772e-02]],
    
             [[-1.5687e-03, -4.2487e-02, -3.7112e-02],
              [-1.4221e-02, -2.5835e-02,  3.9170e-03],
              [ 6.1574e-03, -1.5033e-02, -1.6639e-02]],
    
             [[ 1.4094e-02,  1.0711e-02,  1.0824e-02],
              [ 2.0790e-02, -2.7543e-02,  1.6675e-02],
              [-1.1795e-02,  1.4660e-02, -1.7106e-02]],
    
             ...,
    
             [[-1.4501e-02,  2.8201e-02, -7.0925e-02],
              [ 3.4783e-02,  1.3036e-02,  1.5069e-02],
              [-4.9781e-02, -1.2876e-02, -5.8367e-02]],
    
             [[ 1.3604e-02,  5.0310e-03,  1.9656e-02],
              [ 1.2169e-02, -1.0567e-02, -3.1374e-02],
              [-3.7885e-02, -3.9021e-02, -4.5678e-04]],
    
             [[-7.7496e-02, -3.9379e-02,  3.9398e-02],
              [ 1.6972e-02,  5.6611e-02, -1.7317e-03],
              [ 3.0593e-02,  6.3763e-02, -2.7644e-03]]],


​    
            [[[-1.3291e-02, -1.5206e-02,  5.2298e-03],
              [ 1.1955e-02, -8.3960e-03, -2.1701e-02],
              [ 2.8219e-03,  3.8049e-02,  6.9573e-03]],
    
             [[ 1.7531e-02, -7.3248e-03,  2.9376e-02],
              [ 2.9584e-02,  5.8657e-02, -2.7732e-02],
              [ 3.5657e-02, -5.7662e-02,  3.0640e-02]],
    
             [[-2.1569e-02,  5.9803e-02,  3.7876e-02],
              [ 3.3871e-02,  4.0264e-02,  1.2637e-02],
              [ 5.3023e-02, -1.1335e-02,  1.7939e-02]],
    
             ...,
    
             [[-1.2867e-02, -6.5142e-03, -2.3125e-02],
              [-3.9135e-02,  1.6539e-02, -3.0539e-02],
              [ 2.1629e-02, -3.8552e-02,  1.1575e-02]],
    
             [[ 3.0371e-02,  5.6315e-03,  1.2514e-04],
              [-8.9490e-03, -5.3495e-02,  1.2492e-02],
              [-3.3766e-02,  6.2749e-02, -3.1363e-03]],
    
             [[ 6.9556e-03,  4.1174e-02,  1.4969e-02],
              [-1.3804e-02,  3.0142e-02,  7.5959e-03],
              [-6.8422e-03,  3.4523e-02, -3.5308e-02]]],


​    
            [[[-1.3718e-02,  2.6032e-02, -3.5351e-02],
              [ 1.2929e-02,  1.9278e-02,  2.6253e-02],
              [-4.4458e-03, -3.0676e-02,  6.2885e-03]],
    
             [[-2.3253e-02,  5.8394e-02, -2.7177e-04],
              [ 9.8116e-05, -3.4065e-02,  8.9029e-03],
              [ 9.4137e-03, -3.1040e-02,  5.1619e-04]],
    
             [[-4.7903e-02,  1.4733e-02,  3.7089e-02],
              [-5.0217e-03,  4.9756e-02, -1.6572e-02],
              [ 3.3901e-03, -6.9980e-03,  1.0569e-02]],
    
             ...,
    
             [[ 1.0835e-02,  1.4543e-02, -2.7965e-02],
              [-2.9713e-03, -5.1880e-02, -3.5625e-03],
              [ 3.2518e-03,  1.9563e-02, -6.8342e-03]],
    
             [[-1.7425e-02,  4.1145e-02, -1.6075e-02],
              [-1.2845e-03, -4.9576e-03,  6.3727e-03],
              [ 2.9496e-04,  1.0430e-02,  9.8068e-03]],
    
             [[ 3.5511e-02,  2.3129e-02,  2.8021e-02],
              [ 1.4639e-02,  4.2938e-03,  1.4175e-02],
              [-1.7044e-03, -3.6358e-02,  4.8874e-02]]],


​    
            ...,


​    
            [[[ 8.3247e-03,  5.5425e-02,  8.3526e-03],
              [-2.4693e-02,  9.9390e-04,  3.3968e-02],
              [-4.3386e-03,  8.9296e-04, -1.1349e-02]],
    
             [[ 5.2219e-03, -3.1748e-02,  7.4649e-04],
              [-7.1650e-03,  6.8017e-03,  7.7711e-02],
              [-2.1689e-02, -2.5007e-02,  5.9812e-02]],
    
             [[-2.8304e-02,  2.6397e-02,  2.8205e-02],
              [ 8.4211e-02,  1.1275e-02,  4.8635e-03],
              [ 1.1111e-02,  2.4489e-02, -2.2332e-03]],
    
             ...,
    
             [[-2.5757e-02,  5.5498e-03, -2.1972e-02],
              [-1.3406e-02, -2.0665e-02, -2.7517e-03],
              [-2.4359e-02,  2.7043e-03,  2.5349e-02]],
    
             [[ 5.1658e-03, -2.9786e-02,  1.2704e-02],
              [-1.8020e-02,  8.5598e-02,  6.6740e-04],
              [-3.1628e-03,  2.3645e-02, -6.4903e-02]],
    
             [[ 3.9627e-03, -5.2094e-03,  1.3886e-02],
              [-3.7860e-02,  1.8379e-02,  6.1846e-02],
              [ 8.3205e-03,  2.6255e-02,  3.3783e-02]]],


​    
            [[[-2.0964e-02,  2.5370e-02, -1.7020e-02],
              [ 1.0891e-02,  8.4425e-03,  1.8108e-02],
              [-2.9098e-02,  8.4492e-03, -1.9419e-02]],
    
             [[ 1.1922e-03, -5.1480e-02,  3.8803e-03],
              [-3.6808e-02, -1.7441e-02, -3.5299e-02],
              [-1.3415e-02,  1.5315e-02,  2.6672e-02]],
    
             [[ 9.0886e-03, -2.7243e-03,  3.2336e-02],
              [-5.1367e-03,  2.2698e-02, -2.7158e-02],
              [ 1.0612e-02,  9.1343e-04,  1.0016e-02]],
    
             ...,
    
             [[ 8.2980e-03,  2.2242e-02,  1.6844e-02],
              [-4.2482e-02, -2.4660e-02,  9.3187e-03],
              [-2.8374e-02,  1.1788e-02, -1.8709e-02]],
    
             [[-5.1907e-02,  4.9372e-02, -4.9451e-02],
              [ 1.4267e-02,  3.3285e-02, -2.5228e-02],
              [-6.8552e-03, -1.6252e-03,  8.3553e-03]],
    
             [[-1.4484e-03,  2.2049e-02,  2.0003e-02],
              [-4.5934e-02, -7.7408e-03,  4.2970e-02],
              [ 2.1615e-02,  2.7941e-02, -1.1171e-02]]],


​    
            [[[ 5.4081e-02,  3.3310e-03, -1.5288e-02],
              [ 2.5752e-02,  5.8497e-03, -1.9194e-02],
              [-7.3654e-03,  3.2929e-02,  3.5923e-02]],
    
             [[-4.3991e-02,  2.6179e-02,  3.3552e-02],
              [-3.3413e-02,  2.3890e-02, -2.4189e-02],
              [ 1.3159e-02, -2.3851e-02,  1.2658e-02]],
    
             [[ 3.1942e-02,  1.4981e-03,  2.0295e-03],
              [ 4.3999e-02,  4.4064e-02,  1.1983e-03],
              [-2.9893e-02,  3.2847e-02,  3.0525e-03]],
    
             ...,
    
             [[ 3.1411e-02,  3.0203e-02, -5.8562e-02],
              [ 3.5739e-02,  2.6084e-02, -6.9919e-02],
              [ 1.5704e-02, -2.5062e-02,  4.1598e-02]],
    
             [[ 2.2958e-03,  3.9184e-02, -1.4012e-02],
              [-2.3085e-02,  5.7279e-02,  2.8511e-02],
              [ 4.4926e-02, -2.0987e-02, -4.3386e-03]],
    
             [[ 5.2276e-02, -1.1884e-03,  1.3984e-02],
              [ 2.3814e-02, -6.6492e-02, -3.9726e-02],
              [-2.3462e-03,  6.1467e-02,  5.5971e-02]]]])), ('layer3.1.bn2.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1.])), ('layer3.1.bn2.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer3.1.bn2.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer3.1.bn2.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1.])), ('layer3.1.bn2.num_batches_tracked', tensor(0)), ('layer4.0.conv1.weight', tensor([[[[-3.4426e-02, -2.0650e-02, -7.9379e-03],
              [ 1.4907e-02,  9.9145e-03,  1.6605e-02],
              [ 4.1901e-02, -3.8256e-03,  4.6165e-02]],
    
             [[-1.5916e-02, -1.4827e-02, -1.6131e-02],
              [-2.8016e-02,  2.7889e-03, -7.1107e-03],
              [-1.0538e-02, -5.5127e-02,  2.8052e-02]],
    
             [[-1.2081e-02, -2.3949e-02, -2.0703e-02],
              [ 2.6517e-04, -1.4399e-02,  2.0012e-02],
              [-1.8030e-02, -2.1231e-03,  6.6343e-03]],
    
             ...,
    
             [[-1.7246e-02,  2.4135e-02,  4.2051e-03],
              [ 2.6897e-02, -1.6369e-04, -1.1309e-02],
              [-1.5673e-02, -8.3443e-03,  4.1823e-03]],
    
             [[ 4.3984e-02, -1.7815e-02, -1.4942e-02],
              [-5.1513e-02, -8.1108e-04, -1.5165e-02],
              [-3.6811e-02,  5.4820e-03, -2.3470e-02]],
    
             [[ 1.8343e-02,  1.3291e-02,  3.3124e-03],
              [ 1.5544e-02,  8.9084e-03, -2.0378e-04],
              [-2.5889e-02, -1.6304e-03,  1.8099e-02]]],


​    
            [[[-3.4747e-02, -9.6332e-04,  3.2263e-02],
              [-1.2929e-02,  3.1741e-03,  8.1934e-04],
              [ 1.9111e-02, -2.2590e-02, -3.7585e-02]],
    
             [[-1.3702e-02, -4.8879e-02,  9.8150e-03],
              [ 2.5271e-02,  1.1344e-02,  2.6328e-02],
              [-2.1827e-02,  2.6530e-02, -3.6134e-02]],
    
             [[ 6.9475e-03,  8.4351e-03,  1.2942e-02],
              [ 1.0882e-02, -1.3784e-02,  9.6637e-04],
              [ 3.6921e-02, -3.2078e-02,  2.7090e-03]],
    
             ...,
    
             [[-5.9892e-03, -1.2388e-02,  9.6632e-04],
              [ 1.9213e-04, -7.2601e-03, -7.1639e-03],
              [-9.3838e-03, -4.8896e-03, -3.3557e-02]],
    
             [[-4.0732e-02, -1.1723e-02, -2.5068e-02],
              [ 3.5916e-02, -2.2435e-02, -3.1604e-02],
              [ 3.6318e-02,  5.6179e-03, -1.8664e-02]],
    
             [[-2.7697e-02,  2.9838e-02,  7.5057e-03],
              [ 5.4007e-03,  1.8735e-02, -2.6802e-02],
              [-2.9978e-02,  3.4745e-02,  4.1988e-02]]],


​    
            [[[-1.0869e-03,  5.6779e-03,  2.2907e-02],
              [ 1.4753e-02,  2.7519e-02,  4.3633e-03],
              [-8.5497e-04, -2.6687e-02,  1.4616e-04]],
    
             [[-4.4301e-02, -1.5542e-02, -1.8199e-02],
              [-1.7307e-02, -2.4363e-02,  1.7238e-02],
              [ 1.8318e-02, -2.4690e-03, -3.2064e-02]],
    
             [[ 2.7114e-02, -3.1437e-03, -2.7026e-02],
              [ 5.1856e-02, -2.4643e-02,  3.7309e-02],
              [ 2.6793e-02, -1.8849e-02, -1.9014e-02]],
    
             ...,
    
             [[ 6.0458e-03,  2.8492e-02, -1.6163e-03],
              [-1.1290e-02,  1.5984e-02,  2.1983e-02],
              [ 1.5031e-02,  3.2189e-02, -2.7406e-02]],
    
             [[-5.8427e-03, -4.1422e-02, -1.9051e-02],
              [-6.2273e-03,  1.3993e-02, -9.1453e-03],
              [-1.4653e-02,  2.3670e-02,  2.3815e-02]],
    
             [[-2.4436e-03,  1.2416e-02, -2.6396e-02],
              [-2.7468e-02,  9.6938e-03,  8.4615e-04],
              [ 4.1347e-04,  7.8903e-03, -1.8545e-02]]],


​    
            ...,


​    
            [[[-2.0768e-02,  1.2019e-02, -2.8248e-02],
              [ 1.9572e-02,  2.1437e-02,  1.7574e-02],
              [ 1.0294e-02,  2.8026e-02,  2.3316e-02]],
    
             [[-3.8005e-02, -2.4392e-02, -4.8353e-04],
              [-2.5244e-02,  1.8302e-02,  2.2747e-02],
              [-8.1275e-03, -2.3235e-02,  2.9404e-03]],
    
             [[-6.3621e-03,  2.6156e-02,  3.4662e-03],
              [-1.3349e-02,  1.1119e-02,  8.4367e-03],
              [-1.6682e-02, -7.1614e-03,  2.2505e-02]],
    
             ...,
    
             [[ 1.1390e-02,  2.0807e-02, -9.0026e-03],
              [-7.5694e-03, -1.2274e-03,  6.5852e-03],
              [-9.6348e-03,  7.7879e-03, -3.0931e-02]],
    
             [[-1.5925e-02,  5.9851e-04,  2.2808e-03],
              [ 3.4779e-02, -3.0730e-03, -1.7820e-02],
              [ 4.7367e-03,  3.5228e-03,  1.0464e-02]],
    
             [[-1.5196e-02, -1.9263e-02,  3.0209e-02],
              [ 4.8933e-03, -1.1437e-02,  2.8073e-02],
              [-3.4256e-03,  8.6841e-06,  1.0678e-03]]],


​    
            [[[-3.0823e-03,  1.4280e-03, -6.2772e-03],
              [ 9.2778e-04,  5.1217e-03,  9.3867e-03],
              [-1.0111e-02, -2.2206e-02, -2.5241e-03]],
    
             [[-1.1326e-02,  5.2809e-03,  2.0117e-02],
              [-3.3499e-02,  3.8196e-02,  1.9622e-02],
              [ 4.1473e-03, -2.8321e-02, -7.3684e-03]],
    
             [[-3.1827e-03, -2.1147e-02, -7.2015e-03],
              [-1.2156e-02,  3.5554e-03, -1.0348e-02],
              [-1.7716e-02, -3.9093e-02,  1.4604e-02]],
    
             ...,
    
             [[-2.9968e-02,  1.0996e-02,  1.4550e-02],
              [ 3.8767e-02,  2.4672e-02,  1.1089e-03],
              [ 7.6358e-03, -4.3723e-03,  9.2216e-03]],
    
             [[ 3.4968e-02,  2.2976e-03,  7.0608e-03],
              [-1.1412e-02,  4.5701e-03,  1.3914e-03],
              [-3.1444e-02,  4.0768e-02,  6.1653e-03]],
    
             [[ 1.7533e-02, -3.0494e-02,  1.0099e-02],
              [ 3.1681e-02, -1.6135e-02, -2.6885e-02],
              [ 1.1824e-02,  2.4516e-02,  1.0581e-02]]],


​    
            [[[-3.2708e-04,  5.9305e-03, -1.2218e-02],
              [ 1.7925e-02,  3.7259e-02,  1.1810e-02],
              [ 3.8261e-03,  2.1109e-02, -1.2404e-02]],
    
             [[-2.2040e-02, -7.9393e-03,  3.8319e-02],
              [-2.2752e-02, -7.0236e-03, -2.0490e-02],
              [-1.4765e-02,  2.1094e-04, -5.3436e-03]],
    
             [[ 1.1482e-02, -1.5475e-02,  8.2683e-03],
              [-3.4944e-03,  5.6585e-04, -9.3152e-03],
              [ 1.1439e-02,  1.9590e-02,  6.4645e-03]],
    
             ...,
    
             [[ 3.8313e-02,  2.4285e-02, -1.1131e-02],
              [-1.0242e-04,  4.0969e-02, -2.9994e-02],
              [ 7.1082e-03,  2.2107e-02,  6.5453e-03]],
    
             [[-9.8455e-03,  7.9057e-04,  4.8128e-02],
              [ 2.9453e-02, -2.8000e-02, -8.5380e-03],
              [-2.4060e-02, -2.8138e-02, -4.4874e-02]],
    
             [[-3.5345e-02, -1.9844e-02, -9.3805e-04],
              [-1.2879e-02, -1.6884e-02, -1.4061e-02],
              [ 2.4718e-02,  1.6698e-02,  5.1346e-03]]]])), ('layer4.0.bn1.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1.])), ('layer4.0.bn1.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer4.0.bn1.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer4.0.bn1.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1.])), ('layer4.0.bn1.num_batches_tracked', tensor(0)), ('layer4.0.conv2.weight', tensor([[[[ 3.9987e-02,  1.6377e-02,  1.1537e-02],
              [ 1.5429e-04,  3.3687e-03,  2.6769e-02],
              [ 4.5581e-02, -2.2978e-02,  2.2034e-02]],
    
             [[-2.7881e-02,  6.1255e-03, -1.2566e-02],
              [-1.0602e-02,  2.0413e-02,  3.6084e-02],
              [ 3.5311e-02, -1.3656e-02, -1.1612e-02]],
    
             [[ 2.2937e-02,  1.9162e-02, -5.4952e-03],
              [ 1.1285e-02,  1.8485e-02,  1.6047e-02],
              [ 2.0785e-02, -2.7041e-02, -1.5259e-02]],
    
             ...,
    
             [[ 4.8956e-03,  2.0282e-02, -5.3931e-03],
              [ 1.0222e-02,  1.8082e-02, -1.3942e-02],
              [ 2.6756e-02,  1.8785e-02, -3.1818e-02]],
    
             [[ 9.0138e-03, -2.1343e-02,  1.3420e-02],
              [ 4.6611e-02,  2.0769e-02, -1.2019e-02],
              [ 3.8152e-03,  3.2009e-02,  1.7222e-03]],
    
             [[-1.4284e-02, -1.5476e-02,  3.0132e-02],
              [ 1.4093e-03,  5.1971e-03, -2.1814e-02],
              [-6.4795e-03, -2.5254e-03,  6.3007e-05]]],


​    
            [[[ 4.5274e-03,  1.6223e-02,  2.1589e-02],
              [ 2.3792e-02,  4.5018e-02,  3.7072e-02],
              [ 1.1174e-02, -1.0785e-02, -1.5236e-03]],
    
             [[-3.0813e-03, -3.1105e-02, -8.2161e-03],
              [ 1.1115e-02,  2.0689e-02,  8.1091e-03],
              [-3.2117e-02, -1.1782e-02, -1.5681e-02]],
    
             [[ 3.8722e-04,  2.2575e-02, -2.9844e-02],
              [-1.1150e-03, -2.1851e-02, -1.7332e-02],
              [ 9.3539e-03, -2.7910e-02,  7.0806e-03]],
    
             ...,
    
             [[-7.8759e-03, -4.6700e-02,  3.6113e-02],
              [ 2.9170e-02, -1.5087e-02,  1.4162e-02],
              [-4.6622e-02,  5.1858e-03, -1.1309e-02]],
    
             [[ 4.3175e-03, -2.8738e-02, -9.7116e-03],
              [-1.7032e-02, -5.0311e-03, -3.1874e-02],
              [ 4.5853e-03, -1.2478e-02,  2.6720e-02]],
    
             [[-1.6723e-02, -2.2472e-02, -2.0471e-03],
              [ 5.5419e-04,  2.6593e-02,  2.5199e-02],
              [-1.1873e-02, -4.3603e-03,  2.6322e-03]]],


​    
            [[[-1.3595e-02, -2.6590e-02,  2.3901e-02],
              [ 4.7180e-03,  1.3594e-02,  3.0595e-02],
              [ 4.7231e-02, -4.5193e-02, -4.3234e-02]],
    
             [[ 4.5242e-02,  3.8842e-02, -1.0839e-02],
              [ 3.1156e-02, -3.6172e-02, -2.8559e-02],
              [-2.2913e-02,  1.0383e-02, -5.8493e-03]],
    
             [[ 5.0987e-03, -1.1539e-02, -3.0080e-02],
              [-1.4494e-02,  3.7111e-02,  8.3581e-03],
              [-1.6871e-02,  2.6254e-02, -4.2414e-02]],
    
             ...,
    
             [[ 1.3091e-02,  2.3709e-02, -1.7082e-02],
              [-2.3198e-02,  1.1050e-02, -9.1059e-03],
              [-1.9516e-02, -2.6130e-02,  2.1280e-02]],
    
             [[ 1.1603e-02,  1.4547e-02,  3.1231e-02],
              [ 1.0280e-02,  1.3253e-02,  1.0121e-02],
              [-1.6605e-02, -2.2807e-02, -2.4404e-02]],
    
             [[ 6.8024e-03, -1.3941e-02, -1.4979e-02],
              [-1.7372e-02,  1.2247e-02,  9.5539e-04],
              [ 1.1951e-02, -1.0422e-02,  1.7355e-02]]],


​    
            ...,


​    
            [[[ 2.9022e-02,  2.6381e-02, -1.9288e-02],
              [-4.8321e-03,  2.9191e-02,  6.9125e-04],
              [ 1.5155e-02, -3.5044e-03,  1.2359e-04]],
    
             [[ 2.5525e-02, -1.5591e-02,  1.9321e-02],
              [-2.5520e-03,  1.1416e-02,  2.1690e-03],
              [-4.6676e-03, -2.0419e-02,  9.5332e-04]],
    
             [[-2.8312e-02, -9.8543e-04, -2.9239e-02],
              [-6.1804e-05, -3.0979e-02, -7.3537e-03],
              [ 8.8831e-04, -1.3398e-02,  3.4488e-03]],
    
             ...,
    
             [[ 2.4949e-02,  2.8155e-02,  2.3277e-02],
              [ 5.2676e-03, -2.2039e-02,  2.3924e-02],
              [ 3.0675e-02, -2.8452e-03,  1.4936e-02]],
    
             [[ 1.2647e-02,  1.7826e-02,  4.9843e-02],
              [ 2.5734e-02, -2.5533e-02,  9.8916e-03],
              [ 1.9557e-02,  3.8061e-03, -2.9102e-03]],
    
             [[-8.0495e-03, -1.5350e-02, -3.7474e-02],
              [-2.5358e-02, -4.7807e-03, -3.2121e-02],
              [ 1.5920e-02, -1.0264e-02,  1.8845e-02]]],


​    
            [[[-1.4936e-02, -3.2073e-02,  1.4138e-02],
              [ 8.4281e-03, -1.8128e-02, -2.9171e-02],
              [-3.1910e-02, -7.6171e-03, -1.8151e-02]],
    
             [[-2.4884e-03, -1.3325e-02, -4.8013e-02],
              [-4.5844e-03, -5.3668e-03,  2.5114e-02],
              [-8.1003e-03, -1.3105e-02, -2.9609e-02]],
    
             [[ 2.2515e-02,  3.3793e-03,  3.8654e-04],
              [-9.1275e-03,  6.9659e-03, -4.6730e-03],
              [-2.2623e-02,  2.3079e-02, -4.0595e-03]],
    
             ...,
    
             [[ 1.3057e-02,  5.5646e-03, -1.7202e-02],
              [ 7.1149e-03,  5.9121e-03, -1.4524e-02],
              [-1.0922e-02,  4.2458e-02, -1.8767e-03]],
    
             [[ 2.2265e-02, -3.3623e-02, -1.1373e-02],
              [ 1.0413e-02,  5.7937e-03, -3.0215e-03],
              [-1.7297e-02,  2.1106e-03,  3.6974e-02]],
    
             [[-1.6561e-02,  1.1785e-02,  3.7390e-02],
              [ 1.4489e-02,  1.1884e-02,  6.2340e-03],
              [-9.3383e-03,  1.7597e-02,  1.3923e-02]]],


​    
            [[[ 3.0113e-03, -1.0260e-02,  2.6663e-02],
              [ 6.3192e-03, -1.3597e-02, -2.4875e-02],
              [-1.7034e-02, -4.9622e-03, -2.4227e-02]],
    
             [[ 2.8849e-02,  1.7336e-02,  5.2462e-03],
              [-1.4524e-02,  1.4109e-02,  2.0550e-02],
              [ 4.3476e-03,  2.5029e-02, -1.2926e-02]],
    
             [[-3.0400e-02, -2.2211e-02,  2.8638e-02],
              [ 6.5878e-03, -2.2371e-02,  1.7473e-03],
              [-1.4824e-02, -5.1557e-03, -3.2254e-02]],
    
             ...,
    
             [[-2.6154e-02,  1.6258e-02, -1.9958e-02],
              [ 1.1572e-02, -1.5528e-02,  1.4332e-02],
              [-2.0849e-02,  1.9083e-02,  2.4401e-02]],
    
             [[ 1.2214e-02, -6.1253e-03, -2.1794e-02],
              [-6.6523e-03, -1.1222e-02, -4.7902e-04],
              [-3.7102e-02, -9.2181e-03,  1.0906e-02]],
    
             [[ 3.4334e-03,  8.6663e-03, -2.1325e-02],
              [ 5.3908e-02, -9.8666e-04, -2.3395e-03],
              [-6.9018e-04, -3.1469e-02, -1.6748e-02]]]])), ('layer4.0.bn2.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1.])), ('layer4.0.bn2.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer4.0.bn2.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer4.0.bn2.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1.])), ('layer4.0.bn2.num_batches_tracked', tensor(0)), ('layer4.0.downsample.0.weight', tensor([[[[-0.1354]],
    
             [[ 0.0602]],
    
             [[ 0.0289]],
    
             ...,
    
             [[ 0.0753]],
    
             [[ 0.0123]],
    
             [[-0.0188]]],


​    
            [[[ 0.0252]],
    
             [[ 0.0575]],
    
             [[ 0.0006]],
    
             ...,
    
             [[-0.1084]],
    
             [[ 0.0502]],
    
             [[-0.0238]]],


​    
            [[[-0.0509]],
    
             [[-0.0502]],
    
             [[ 0.0187]],
    
             ...,
    
             [[-0.1324]],
    
             [[-0.0758]],
    
             [[ 0.0340]]],


​    
            ...,


​    
            [[[-0.0054]],
    
             [[ 0.0149]],
    
             [[-0.0721]],
    
             ...,
    
             [[-0.0617]],
    
             [[-0.0071]],
    
             [[-0.1242]]],


​    
            [[[-0.0233]],
    
             [[ 0.0655]],
    
             [[-0.0293]],
    
             ...,
    
             [[ 0.0699]],
    
             [[-0.0081]],
    
             [[ 0.1189]]],


​    
            [[[-0.1025]],
    
             [[ 0.0254]],
    
             [[-0.0530]],
    
             ...,
    
             [[-0.0620]],
    
             [[ 0.0768]],
    
             [[-0.0705]]]])), ('layer4.0.downsample.1.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1.])), ('layer4.0.downsample.1.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer4.0.downsample.1.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer4.0.downsample.1.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1.])), ('layer4.0.downsample.1.num_batches_tracked', tensor(0)), ('layer4.1.conv1.weight', tensor([[[[-0.0325,  0.0232,  0.0008],
              [ 0.0086,  0.0065, -0.0044],
              [ 0.0280,  0.0054,  0.0270]],
    
             [[-0.0493,  0.0004,  0.0107],
              [-0.0045,  0.0231,  0.0146],
              [-0.0070, -0.0103,  0.0178]],
    
             [[-0.0034,  0.0108, -0.0015],
              [-0.0314,  0.0683,  0.0057],
              [ 0.0080, -0.0100, -0.0121]],
    
             ...,
    
             [[ 0.0033, -0.0216,  0.0006],
              [-0.0278, -0.0087, -0.0185],
              [ 0.0188,  0.0032, -0.0023]],
    
             [[-0.0338,  0.0129, -0.0048],
              [ 0.0084,  0.0305,  0.0142],
              [-0.0139,  0.0383,  0.0406]],
    
             [[-0.0686, -0.0325,  0.0332],
              [ 0.0020, -0.0054, -0.0112],
              [-0.0077, -0.0010,  0.0078]]],


​    
            [[[-0.0243,  0.0150,  0.0318],
              [ 0.0119, -0.0112,  0.0148],
              [ 0.0383,  0.0199,  0.0068]],
    
             [[ 0.0013, -0.0315, -0.0010],
              [-0.0203,  0.0103,  0.0094],
              [-0.0151, -0.0121, -0.0203]],
    
             [[-0.0341, -0.0234, -0.0026],
              [ 0.0350, -0.0224, -0.0248],
              [ 0.0159, -0.0219, -0.0127]],
    
             ...,
    
             [[ 0.0058,  0.0068, -0.0087],
              [-0.0037,  0.0230,  0.0233],
              [-0.0366, -0.0026, -0.0189]],
    
             [[-0.0141,  0.0222, -0.0415],
              [-0.0459,  0.0135,  0.0050],
              [ 0.0079,  0.0372, -0.0156]],
    
             [[-0.0328, -0.0193, -0.0409],
              [ 0.0119,  0.0201,  0.0032],
              [-0.0222, -0.0082,  0.0064]]],


​    
            [[[ 0.0242, -0.0264,  0.0295],
              [ 0.0264,  0.0230,  0.0105],
              [-0.0207, -0.0118, -0.0314]],
    
             [[-0.0118,  0.0110, -0.0088],
              [ 0.0255,  0.0467, -0.0211],
              [-0.0334,  0.0163, -0.0048]],
    
             [[ 0.0081,  0.0102, -0.0042],
              [-0.0248,  0.0082, -0.0155],
              [-0.0041,  0.0004, -0.0073]],
    
             ...,
    
             [[-0.0181, -0.0056,  0.0169],
              [ 0.0221,  0.0088,  0.0321],
              [ 0.0145, -0.0208,  0.0133]],
    
             [[-0.0042, -0.0308, -0.0267],
              [-0.0023, -0.0079,  0.0153],
              [-0.0004, -0.0119, -0.0183]],
    
             [[-0.0288, -0.0120,  0.0191],
              [-0.0142,  0.0351,  0.0207],
              [ 0.0019, -0.0065, -0.0266]]],


​    
            ...,


​    
            [[[ 0.0199,  0.0022,  0.0199],
              [-0.0171, -0.0496,  0.0146],
              [ 0.0214, -0.0213,  0.0159]],
    
             [[-0.0022,  0.0159,  0.0062],
              [-0.0030, -0.0032, -0.0108],
              [-0.0481, -0.0035, -0.0002]],
    
             [[-0.0147,  0.0247, -0.0213],
              [ 0.0371,  0.0003, -0.0025],
              [-0.0174, -0.0376, -0.0072]],
    
             ...,
    
             [[ 0.0167, -0.0006,  0.0055],
              [ 0.0026, -0.0105,  0.0006],
              [-0.0327, -0.0348,  0.0079]],
    
             [[-0.0455,  0.0031,  0.0042],
              [ 0.0278,  0.0231,  0.0283],
              [ 0.0002,  0.0230, -0.0246]],
    
             [[ 0.0163, -0.0081, -0.0103],
              [ 0.0099,  0.0322,  0.0340],
              [-0.0082, -0.0066, -0.0001]]],


​    
            [[[-0.0228,  0.0106,  0.0129],
              [-0.0187, -0.0123, -0.0146],
              [ 0.0074, -0.0158,  0.0317]],
    
             [[ 0.0152,  0.0024, -0.0054],
              [-0.0139,  0.0057, -0.0325],
              [ 0.0182, -0.0334, -0.0137]],
    
             [[-0.0183, -0.0541,  0.0039],
              [-0.0131, -0.0093,  0.0177],
              [ 0.0458,  0.0213,  0.0033]],
    
             ...,
    
             [[ 0.0030, -0.0052, -0.0268],
              [-0.0207, -0.0085, -0.0107],
              [ 0.0027,  0.0044,  0.0266]],
    
             [[-0.0208,  0.0075, -0.0117],
              [-0.0437, -0.0346, -0.0385],
              [ 0.0183, -0.0017,  0.0119]],
    
             [[-0.0279, -0.0053, -0.0061],
              [-0.0021,  0.0037,  0.0264],
              [-0.0009,  0.0113,  0.0025]]],


​    
            [[[-0.0166,  0.0277, -0.0019],
              [-0.0078,  0.0233,  0.0373],
              [ 0.0110, -0.0101,  0.0001]],
    
             [[ 0.0176,  0.0080,  0.0250],
              [-0.0071,  0.0051,  0.0149],
              [ 0.0005,  0.0050,  0.0129]],
    
             [[-0.0034, -0.0264, -0.0136],
              [-0.0303, -0.0102, -0.0298],
              [-0.0090,  0.0044, -0.0005]],
    
             ...,
    
             [[ 0.0032, -0.0098, -0.0129],
              [ 0.0101,  0.0107, -0.0183],
              [ 0.0147,  0.0256,  0.0420]],
    
             [[-0.0143,  0.0458, -0.0042],
              [ 0.0407, -0.0196, -0.0276],
              [-0.0008,  0.0726, -0.0162]],
    
             [[-0.0229, -0.0024,  0.0218],
              [ 0.0104,  0.0120, -0.0031],
              [-0.0102, -0.0265, -0.0226]]]])), ('layer4.1.bn1.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1.])), ('layer4.1.bn1.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer4.1.bn1.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer4.1.bn1.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1.])), ('layer4.1.bn1.num_batches_tracked', tensor(0)), ('layer4.1.conv2.weight', tensor([[[[-2.3662e-02, -2.3513e-02, -1.5573e-02],
              [-4.7517e-02,  5.9073e-03, -1.3614e-03],
              [ 6.0780e-03, -3.8839e-02, -1.7578e-02]],
    
             [[-2.1779e-02,  4.0308e-02,  6.1145e-03],
              [-5.9949e-03, -5.6621e-03,  1.4169e-02],
              [ 6.7464e-03,  1.3252e-02, -3.0208e-03]],
    
             [[ 4.5111e-02, -2.0537e-02,  1.5476e-02],
              [-5.1928e-03, -2.4852e-03, -2.5094e-02],
              [ 1.7014e-02,  3.2177e-02,  1.0081e-02]],
    
             ...,
    
             [[ 1.6567e-02,  2.2133e-05, -1.3088e-02],
              [-1.8307e-02, -1.1853e-02, -2.9698e-03],
              [-2.6929e-02,  2.2143e-02,  6.0593e-03]],
    
             [[ 3.9457e-02,  3.0668e-02, -1.2497e-02],
              [ 5.0543e-03, -1.1222e-02,  1.6106e-02],
              [ 1.5617e-02,  9.0761e-04, -1.2669e-02]],
    
             [[-9.1282e-03,  3.6710e-02,  4.3944e-03],
              [-8.8752e-03,  1.4686e-02,  1.0446e-02],
              [-1.8295e-02, -2.2993e-02, -2.3620e-02]]],


​    
            [[[ 1.6531e-03,  3.4567e-03, -1.9429e-02],
              [-4.6900e-03,  1.0452e-02, -2.5580e-02],
              [-1.4039e-03,  7.5230e-03, -3.8997e-03]],
    
             [[-1.0146e-02, -1.2023e-03, -6.3622e-03],
              [-5.0349e-03,  1.8959e-02, -2.4189e-02],
              [-1.9850e-02,  1.4023e-02, -1.9445e-02]],
    
             [[ 9.0402e-03, -3.0418e-02,  8.1724e-03],
              [ 3.7860e-02, -1.2944e-02,  9.7460e-03],
              [-1.5397e-02, -2.0535e-02,  2.0626e-02]],
    
             ...,
    
             [[-3.3120e-03, -1.3496e-02,  1.1123e-02],
              [-3.2833e-02, -4.7743e-04, -3.2559e-02],
              [ 4.8121e-03, -1.0991e-02,  1.1493e-04]],
    
             [[-2.0775e-02,  3.9096e-03, -3.9679e-03],
              [ 2.9163e-02,  1.8294e-02, -3.9337e-03],
              [-5.7133e-04,  2.3218e-03, -8.7641e-03]],
    
             [[ 3.8756e-03, -3.3386e-02,  4.3730e-02],
              [-3.0961e-02,  1.6407e-02,  1.0110e-02],
              [-1.6114e-02,  3.1494e-02,  5.9744e-03]]],


​    
            [[[-1.1329e-02,  3.3126e-03, -4.6241e-03],
              [-2.7044e-02,  6.5909e-03,  7.0866e-03],
              [ 5.2363e-02,  1.5017e-02, -1.7317e-02]],
    
             [[-1.3637e-02,  4.2175e-03,  9.4972e-03],
              [ 6.6826e-03, -3.8088e-03,  7.7308e-03],
              [-2.2740e-02, -1.5828e-02, -2.2909e-02]],
    
             [[-4.5026e-03,  4.6851e-03,  4.6103e-02],
              [ 1.2477e-02,  3.2316e-02,  1.1858e-02],
              [-1.6457e-02,  1.1231e-04,  2.8827e-02]],
    
             ...,
    
             [[ 3.1597e-02,  2.6750e-02, -9.7675e-03],
              [ 2.5121e-02,  1.2314e-02, -1.3105e-02],
              [ 1.2193e-02, -5.0444e-02, -3.3051e-02]],
    
             [[-2.4225e-02, -2.7506e-02, -8.8601e-03],
              [ 2.6399e-02,  2.7519e-02, -4.3771e-02],
              [-1.1470e-02, -4.0470e-03,  2.0637e-02]],
    
             [[-6.1738e-03, -3.3873e-02, -1.0252e-02],
              [ 2.0712e-02, -1.3556e-02,  1.4516e-02],
              [ 7.6622e-05,  1.0441e-02, -2.4969e-02]]],


​    
            ...,


​    
            [[[-1.2288e-03, -1.9742e-03, -1.8614e-02],
              [-6.3986e-03, -4.2116e-03,  2.3176e-02],
              [ 1.0561e-02, -1.6420e-02,  3.4507e-02]],
    
             [[-1.9764e-02,  1.0785e-02,  1.6745e-02],
              [-6.5887e-03,  1.5259e-03, -2.6146e-02],
              [ 2.1095e-02, -3.6375e-02, -2.7022e-03]],
    
             [[ 2.9852e-02, -2.4478e-02,  4.6899e-03],
              [ 1.6306e-02, -1.7728e-04, -3.1754e-04],
              [ 5.4857e-03,  3.3921e-02,  2.7303e-03]],
    
             ...,
    
             [[-2.8838e-02, -2.3566e-02, -1.5317e-02],
              [-1.8346e-02, -5.1785e-03,  2.4209e-02],
              [-2.9087e-02, -2.0016e-02,  1.8509e-02]],
    
             [[ 8.3490e-03,  1.7051e-02, -2.7071e-03],
              [-7.8314e-03,  6.2532e-03,  1.2643e-02],
              [-9.9559e-03, -4.6574e-02, -3.2087e-02]],
    
             [[-1.1705e-02,  1.8888e-02,  3.4632e-02],
              [ 5.1837e-02, -3.0410e-02, -1.1340e-02],
              [ 1.4330e-02, -1.1469e-02, -8.6637e-03]]],


​    
            [[[ 3.0953e-03, -2.9400e-02, -5.0638e-03],
              [-1.7371e-02, -5.2132e-03, -1.8046e-02],
              [ 6.5511e-03,  3.0614e-02, -3.1634e-03]],
    
             [[ 3.3873e-02, -1.3166e-02,  1.1041e-02],
              [-3.8080e-02,  2.4966e-02,  4.4413e-02],
              [-7.7000e-03,  2.9861e-02, -2.8791e-02]],
    
             [[ 6.4641e-03,  1.5620e-02,  1.2418e-02],
              [ 1.8234e-02,  9.1384e-03,  1.9138e-02],
              [-6.1269e-02,  6.9481e-03, -1.1410e-02]],
    
             ...,
    
             [[ 1.1664e-02,  4.8882e-03, -1.9306e-03],
              [ 2.6281e-02,  4.4745e-03,  2.5958e-03],
              [-3.7051e-02, -2.4621e-02,  1.3470e-02]],
    
             [[ 1.1898e-02,  3.2344e-02,  2.6440e-02],
              [-1.1959e-02, -4.2995e-02,  2.4245e-02],
              [-1.5575e-02, -1.0455e-02,  1.6821e-02]],
    
             [[ 7.6697e-03,  5.1326e-02,  1.4794e-02],
              [-1.4745e-02, -3.4990e-03, -4.9270e-02],
              [ 1.6760e-02,  3.9897e-03,  1.0378e-02]]],


​    
            [[[ 3.4175e-02,  2.4683e-04,  1.1436e-02],
              [ 1.7215e-02,  1.4469e-02, -1.3782e-02],
              [ 2.6343e-02, -1.7095e-03,  1.0328e-02]],
    
             [[ 1.5810e-02, -6.1180e-03,  2.4916e-03],
              [ 1.6777e-02, -1.3712e-02,  7.2718e-03],
              [ 1.4679e-02, -4.3712e-03,  2.2391e-02]],
    
             [[ 2.1680e-02, -7.1656e-03,  9.3273e-03],
              [ 3.4611e-03, -1.0776e-03, -7.4966e-03],
              [-1.9199e-02, -3.3206e-02,  7.8546e-03]],
    
             ...,
    
             [[-2.0247e-02, -9.2025e-03,  6.8522e-03],
              [ 7.8949e-03,  1.2932e-02, -8.9096e-03],
              [-2.7274e-02,  5.1701e-02,  4.2695e-03]],
    
             [[ 1.3120e-02, -1.8770e-02,  2.2096e-02],
              [ 4.8489e-02, -3.2794e-02, -1.3497e-02],
              [-1.7710e-02,  1.3417e-02,  7.2844e-03]],
    
             [[ 4.3833e-03, -4.3744e-04,  9.6720e-03],
              [ 1.1254e-02,  3.1494e-02,  7.3637e-03],
              [-2.7838e-02, -9.8798e-03, -4.8723e-04]]]])), ('layer4.1.bn2.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1.])), ('layer4.1.bn2.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer4.1.bn2.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer4.1.bn2.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1.])), ('layer4.1.bn2.num_batches_tracked', tensor(0)), ('fc.weight', tensor([[ 0.0363, -0.0408, -0.0379,  ...,  0.0182,  0.0142, -0.0282],
            [-0.0035,  0.0050, -0.0371,  ..., -0.0427,  0.0100, -0.0073],
            [-0.0090, -0.0298, -0.0032,  ..., -0.0061,  0.0058,  0.0414],
            ...,
            [ 0.0085, -0.0284,  0.0028,  ...,  0.0014, -0.0304, -0.0351],
            [ 0.0405, -0.0179, -0.0233,  ...,  0.0163,  0.0255, -0.0224],
            [ 0.0163, -0.0030, -0.0436,  ..., -0.0361, -0.0100,  0.0359]])), ('fc.bias', tensor([-3.3117e-02,  2.1381e-02,  4.5191e-04,  1.9420e-02,  3.1861e-02,
             3.8301e-02,  1.6528e-02, -1.1133e-03,  1.2325e-02,  1.7361e-02,
             2.5049e-02,  1.4298e-02,  2.9451e-02, -1.8848e-02, -1.4263e-02,
             2.3015e-02,  2.6016e-02, -1.1961e-02, -3.4514e-02, -1.1216e-02,
             1.1031e-02,  2.1724e-02,  9.8035e-05, -2.2028e-02, -1.6308e-02,
             3.0718e-02,  8.8301e-03, -3.4585e-02, -7.8474e-03,  1.5382e-02,
             2.6976e-02,  4.3321e-02, -4.2952e-02, -3.4341e-02,  2.0501e-02,
            -2.9527e-02,  1.3484e-02,  1.6248e-02, -2.2084e-02,  8.7246e-04,
            -2.9450e-02,  3.9827e-02,  1.1267e-02,  1.9803e-02, -2.3606e-02,
            -4.0999e-02, -2.5357e-02,  5.5554e-03, -3.4448e-02, -1.9038e-02,
             3.7538e-02,  7.3885e-04,  1.3478e-02,  1.5418e-03,  1.4757e-02,
            -3.7121e-02, -3.5226e-02,  2.0180e-02, -1.8588e-02,  4.7064e-03,
             2.2375e-02,  1.1967e-02,  1.2655e-02, -3.7507e-02,  1.7396e-02,
             2.9527e-02, -8.1724e-03, -3.5185e-02, -4.3861e-03, -3.8405e-03,
            -3.4795e-02,  7.9800e-03, -3.9885e-02, -3.4188e-02,  3.4676e-02,
            -3.0962e-02, -1.2788e-02, -2.7568e-02,  3.2951e-02, -1.6121e-02,
             2.9050e-02,  6.3742e-03, -3.2456e-02, -2.8273e-02, -7.6407e-03,
             2.2433e-02,  8.2769e-03, -2.9409e-02, -2.5392e-02,  3.3147e-04,
            -4.4186e-05, -4.3394e-02, -2.1682e-02, -5.7963e-03, -2.8437e-02,
            -1.5697e-02,  3.9307e-02, -4.0329e-02, -2.3179e-02,  2.4079e-02,
             2.4714e-02,  2.8919e-02, -4.1711e-02,  2.4463e-02,  3.3694e-02,
            -3.9821e-02,  2.2279e-02,  1.5718e-02, -3.6740e-02,  3.2902e-02,
            -3.3871e-02, -3.2231e-02,  8.6428e-03,  1.6780e-02,  4.1209e-02,
             2.6154e-02, -3.0900e-02, -1.9897e-02, -1.1001e-02, -3.8112e-02,
            -6.0327e-03, -2.9308e-02, -3.2270e-02, -2.9932e-02,  4.0400e-02,
             1.4672e-02, -3.4729e-02, -1.7627e-02,  1.4553e-02,  4.6136e-03,
            -1.4569e-02, -3.8026e-02, -2.1089e-02, -2.5645e-02, -3.5489e-02,
             4.4113e-02,  3.4505e-02, -9.0649e-03, -3.8817e-02, -2.2010e-02,
            -3.4405e-02, -1.2182e-03,  8.8558e-03,  2.9489e-02, -4.2498e-02,
             1.5451e-02,  1.1058e-02,  1.7396e-02, -1.2795e-02, -4.2771e-02,
            -3.1412e-02,  1.6232e-02, -1.6277e-02,  2.2794e-02, -3.7098e-02,
            -5.2723e-03, -2.1384e-02, -2.0074e-02,  9.3457e-03,  1.8284e-02,
             2.6006e-02, -2.6507e-04,  1.1843e-02, -2.1611e-02,  1.1691e-02,
             2.9002e-02,  3.2130e-02, -3.7340e-02,  2.5418e-03,  2.9189e-02,
            -1.6373e-02, -2.9543e-02, -3.2752e-02, -2.8742e-02, -2.8770e-02,
             3.9741e-02, -3.8369e-02, -5.1904e-03,  2.9692e-02,  5.7396e-03,
            -3.7989e-02, -3.0075e-02, -1.8731e-03, -3.8163e-02, -4.0778e-02,
            -7.3722e-03,  4.0741e-02, -2.1000e-03,  2.0497e-03, -2.4638e-02,
            -3.0484e-02, -4.1319e-02,  2.9949e-02, -4.0966e-02, -3.1474e-02,
             3.4290e-02,  1.1897e-02, -2.1155e-02, -2.9783e-02,  3.4406e-02,
             2.8104e-02,  3.4051e-02,  4.0026e-02, -3.0636e-02,  1.5356e-02,
             2.4776e-02, -3.3609e-02, -4.1504e-02, -2.6813e-03, -2.4690e-02,
             1.3166e-02, -3.7950e-02,  3.1911e-02, -1.6918e-02,  3.1129e-02,
            -3.4166e-02,  4.7001e-03, -4.4326e-03,  2.1315e-02, -3.9182e-02,
             3.3979e-03, -3.6935e-02,  9.7632e-03,  2.5121e-02, -2.7804e-02,
             9.4276e-03,  4.9146e-03,  3.8149e-02,  3.9888e-02,  6.3460e-03,
            -6.0112e-03,  2.4483e-02, -2.9934e-02,  2.3124e-02, -2.8490e-02,
            -8.6811e-03,  2.2993e-02,  4.3186e-03, -2.5025e-02, -4.2579e-02,
             2.7283e-02,  8.4686e-03,  1.7222e-02, -5.4135e-03,  1.3769e-02,
            -3.9609e-02,  2.0245e-02, -2.4285e-02, -5.5607e-05,  3.8660e-02,
             2.5464e-02,  2.2624e-02,  2.4475e-02, -3.2772e-02,  8.8715e-03,
            -1.9322e-03, -3.1793e-02, -3.9052e-02, -3.0444e-02, -3.7943e-02,
            -2.2011e-02, -3.9018e-03,  3.3250e-03,  2.7783e-03, -2.0854e-02,
            -2.6978e-02, -3.4884e-02,  4.1169e-02,  3.9958e-02, -3.9000e-02,
             2.5006e-02, -2.1077e-02, -1.3210e-02, -1.6881e-02,  2.3695e-02,
             1.7328e-03, -2.9827e-02, -6.1723e-03,  3.7020e-02,  2.1721e-02,
             2.8889e-03,  1.1058e-02, -8.7944e-03, -3.4843e-02,  2.9914e-02,
             8.9737e-03,  4.0784e-02, -3.7105e-02,  3.4704e-02, -3.5057e-02,
            -2.9317e-02, -1.7463e-02, -1.6207e-02, -4.1927e-03,  8.3389e-04,
             3.6975e-02, -3.5106e-02, -1.1557e-02,  1.0856e-02,  1.3658e-02,
             5.7960e-03,  2.5348e-02, -1.9471e-02, -4.0130e-02, -1.7840e-02,
             3.6776e-02,  1.0897e-02, -1.1202e-02, -2.6724e-02, -1.2862e-02,
             1.9320e-02, -2.8532e-02,  3.2379e-02, -2.2447e-02, -2.2893e-02,
            -5.2274e-04, -2.6122e-02,  1.4663e-02,  1.6062e-02, -2.1751e-02,
            -2.9038e-02, -2.8538e-02, -3.3143e-02, -4.1589e-02, -2.2380e-02,
             2.4770e-02,  2.9548e-02, -3.1688e-02, -1.4831e-02, -2.9779e-02,
             1.4684e-02, -6.2382e-03,  2.7696e-02, -3.3836e-02, -4.1328e-04,
            -2.1557e-02,  1.1007e-02, -4.3869e-03, -2.4330e-02, -2.5874e-02,
            -1.5663e-03,  3.3370e-02,  1.7552e-02,  1.0547e-02,  1.4226e-02,
            -9.3844e-05, -2.2725e-02, -3.8917e-02, -4.1002e-02, -4.2495e-02,
             3.0813e-02, -1.4693e-02,  1.4772e-02, -3.6432e-02,  1.1980e-02,
             3.2839e-02,  3.5414e-02,  3.3558e-03,  2.3640e-02,  3.8349e-02,
             4.3575e-02, -6.2014e-03,  5.3419e-03, -3.1280e-02,  3.6914e-02,
            -1.5123e-02,  3.3062e-02,  3.0890e-02, -3.9429e-03, -4.1753e-02,
             2.2143e-02, -1.9111e-02,  1.0017e-02, -2.5360e-02,  4.0554e-02,
             1.7179e-02, -2.9932e-02, -1.3201e-02, -1.4634e-02,  2.4067e-02,
            -1.6131e-02, -2.9521e-02,  3.8612e-02,  7.2837e-03,  4.3081e-04,
            -3.7409e-02, -1.1084e-02, -3.2180e-02,  4.7240e-04,  3.7482e-02,
             3.9070e-02, -5.7358e-04, -1.6119e-02,  1.5712e-02, -3.2270e-02,
             8.3880e-03, -1.9446e-02,  1.3668e-02,  1.1749e-02, -1.0522e-02,
             4.3861e-02,  3.0141e-02, -2.9129e-02, -2.4849e-02, -1.2251e-02,
             2.8610e-02, -2.3811e-02, -1.5605e-02,  3.3063e-02,  1.0504e-02,
            -7.7839e-03,  1.6718e-02,  1.2006e-02, -3.0025e-03, -3.3952e-02,
             1.5182e-02,  1.7610e-02, -1.2439e-02,  3.7906e-02,  4.3673e-02,
             1.6637e-02,  1.2565e-02,  4.1261e-02,  1.8202e-02, -3.0006e-02,
             3.7283e-03,  3.4605e-02,  2.7511e-03, -4.4229e-03,  2.7645e-02,
            -1.5111e-02,  2.8954e-02, -4.0425e-02, -1.0690e-02,  3.5759e-02,
             4.2533e-02,  1.3286e-02,  2.1400e-02,  3.2778e-02,  3.3202e-02,
            -1.9615e-02,  3.7081e-02,  4.0888e-02,  3.6719e-02, -2.3201e-02,
            -3.5237e-02,  3.9244e-02, -1.2778e-02,  3.8095e-02, -1.9599e-02,
            -3.6543e-02, -8.3564e-03,  7.5366e-03, -4.3592e-02,  8.1332e-03,
            -3.8283e-02, -1.7549e-02,  4.3591e-02,  1.6252e-02,  3.3563e-02,
            -2.4057e-02,  3.4805e-02,  2.7358e-02,  3.1244e-02, -6.4368e-03,
            -1.5385e-02,  2.9766e-03, -4.0208e-02, -2.9103e-02, -1.9667e-02,
             3.2045e-03, -3.5145e-02, -2.6877e-02,  2.4310e-02,  1.5016e-02,
            -3.7517e-02,  2.7698e-02, -3.5734e-04,  2.9763e-02,  7.0784e-03,
            -2.9078e-02, -2.3531e-02, -3.9588e-02,  1.7974e-02, -4.0546e-02,
             3.5108e-02,  1.3049e-03,  4.0017e-02,  2.1876e-02, -1.4840e-02,
             2.0077e-03,  1.2187e-03, -1.0499e-02, -1.9889e-02,  1.1983e-02,
            -3.3845e-02, -3.1640e-02, -3.9850e-02, -3.6663e-02,  1.2043e-03,
             2.4583e-02, -3.0066e-02, -3.8949e-02, -4.6746e-03, -9.6048e-03,
            -2.0147e-02, -1.3608e-02,  1.0943e-02, -2.6662e-04,  4.1414e-02,
             4.1538e-02,  4.0596e-03, -4.2921e-02, -3.8691e-02,  1.3438e-02,
             2.0696e-02,  3.8188e-03, -3.8685e-02,  3.6010e-04,  2.3068e-03,
             1.8161e-02,  4.0909e-02,  2.5874e-02, -3.9502e-02,  8.9886e-03,
            -1.7827e-02,  2.7839e-02,  1.3329e-02,  4.1513e-02, -4.2627e-02,
            -3.7625e-02, -3.0720e-02, -3.4754e-02,  1.4254e-02, -2.3538e-02,
             3.9863e-03, -2.3081e-02, -3.1748e-02,  1.2053e-02,  6.2845e-03,
            -4.3717e-02,  3.8105e-02, -5.7647e-03, -3.0263e-02,  4.1685e-02,
            -1.1686e-02,  1.6059e-02,  1.9182e-02,  4.0167e-03,  4.0292e-03,
            -5.0742e-03,  1.4587e-02,  1.1656e-02, -2.6953e-02,  1.8458e-02,
            -1.4548e-02,  1.8667e-02,  4.3167e-03, -1.8348e-02,  2.7135e-02,
             1.2274e-03, -4.2907e-02, -2.1928e-03, -3.2981e-03, -1.0747e-02,
            -2.2316e-02,  2.9696e-02,  2.1058e-02,  1.8506e-02, -5.3214e-03,
            -7.3174e-03, -3.6476e-02, -3.5608e-02,  2.4610e-02, -1.6470e-02,
            -1.1781e-02,  1.2980e-02, -2.4338e-02,  3.6172e-02,  3.9334e-02,
            -1.2570e-02, -4.3619e-02,  4.2439e-02,  1.3377e-02, -4.3007e-02,
             2.5111e-02, -1.0785e-02, -2.0383e-02,  1.0891e-02,  7.6394e-03,
             3.2379e-02,  1.0264e-02, -3.7485e-02,  4.2377e-02,  4.4162e-02,
            -2.0059e-02, -1.6613e-02, -2.0599e-02,  1.3724e-02,  1.5848e-02,
            -6.4984e-03, -2.8599e-02, -3.9005e-02, -2.5267e-02, -3.7383e-02,
            -3.7925e-02, -3.0174e-02,  2.5201e-02,  2.7934e-03, -2.1073e-02,
            -1.2213e-02,  3.9711e-02,  2.8561e-02, -2.6600e-02,  3.6301e-03,
            -1.0460e-02, -2.0751e-02, -2.4667e-02,  1.0336e-02, -3.8812e-02,
            -1.6428e-02,  2.7372e-02, -3.0376e-03, -1.0425e-02, -3.6220e-02,
             1.9789e-02, -1.9636e-03,  1.1677e-02, -6.8549e-03, -2.9949e-02,
            -2.1845e-02,  3.3612e-02,  1.4938e-02,  3.7081e-02, -1.9472e-02,
             2.8941e-02,  7.4000e-03, -1.3529e-02, -3.1744e-02, -3.4879e-02,
            -3.7831e-02, -2.3466e-02,  2.0166e-03, -2.0493e-02,  1.9948e-02,
            -3.3814e-02,  2.6432e-02,  3.5718e-02,  3.7295e-02, -1.1016e-02,
             3.0097e-02, -2.7377e-02,  2.8207e-02,  2.2716e-03,  4.2445e-02,
             2.4413e-04, -3.8713e-02,  2.4490e-02, -4.3551e-02,  4.1845e-02,
            -3.4120e-02,  3.1620e-02,  1.8219e-02,  8.5968e-03, -3.6503e-02,
             3.1027e-02, -1.5036e-02, -3.2103e-02,  2.6419e-02,  2.5318e-02,
            -3.0718e-02, -3.0912e-02,  3.1243e-02,  4.1520e-02, -1.0320e-02,
             3.3256e-02, -2.8246e-02, -1.0273e-02, -2.2718e-02,  3.7776e-02,
            -5.7787e-03,  3.4068e-02, -3.2193e-03,  3.2025e-03,  1.0997e-02,
             3.0308e-02,  2.6697e-02,  2.3445e-02,  1.3668e-02,  5.8259e-03,
             1.4710e-02, -1.4076e-02, -9.1691e-03, -2.1146e-02, -1.4024e-02,
             3.6937e-02,  3.3068e-02, -1.0702e-03, -3.4627e-02,  1.3308e-02,
            -9.1929e-03, -1.2564e-02,  5.6581e-03, -3.6414e-02,  3.8871e-02,
             1.1841e-02,  4.2039e-02,  4.3033e-02, -4.2008e-03, -3.0057e-02,
            -4.0631e-02, -1.6801e-02,  4.4721e-04, -1.9978e-02, -2.9911e-02,
            -3.7106e-02,  2.5949e-02,  6.6971e-03,  2.8034e-02, -9.3827e-03,
             3.8626e-02, -3.9483e-02,  3.4315e-02, -1.2284e-02,  4.4942e-03,
            -2.8916e-02, -3.7983e-02, -1.4357e-02, -3.6765e-02, -4.3759e-02,
            -7.6536e-03,  1.7625e-02,  3.2218e-02,  2.9999e-02,  4.3099e-02,
            -1.0266e-03, -3.3792e-02,  6.3643e-03,  3.9727e-02,  2.2211e-02,
            -1.3953e-02, -9.1497e-03, -2.3724e-02,  2.7131e-02, -1.6413e-03,
            -4.1132e-02,  3.4204e-02,  2.1569e-02,  5.0671e-03,  3.7351e-02,
             4.3558e-02,  2.0884e-02,  2.9375e-02,  4.1709e-02,  2.8462e-02,
            -8.8910e-04,  7.5950e-03, -2.8751e-02,  2.7188e-02, -1.1303e-02,
            -8.4523e-03,  1.5901e-02,  5.8650e-03, -3.0121e-02,  1.5643e-02,
            -4.3617e-02,  3.8544e-02, -1.3150e-02, -2.6992e-02, -3.9682e-03,
             3.0088e-02,  3.4921e-02,  3.4194e-02, -9.4354e-03,  3.8235e-02,
            -2.7295e-02,  8.2966e-03,  3.0110e-02,  1.6849e-03,  1.5671e-02,
            -2.0562e-02, -2.0207e-02, -3.5759e-02,  3.4332e-03,  1.2668e-02,
             4.0891e-02, -1.1468e-02, -1.2866e-02, -3.1233e-02, -4.3509e-02,
             2.5003e-02,  5.0937e-03, -1.6270e-02, -7.7638e-03, -2.7409e-02,
             3.3274e-02,  4.1334e-02,  4.3927e-02, -3.7319e-02,  2.0777e-02,
             1.4294e-02, -4.2854e-03,  1.4250e-02, -4.0259e-02,  3.2913e-02,
             2.3235e-02, -4.0119e-02,  3.8079e-02, -2.0954e-02,  9.4767e-03,
            -2.7012e-02, -6.8499e-03, -5.9213e-03,  2.9294e-02,  8.0973e-03,
            -3.5484e-02, -3.8084e-03, -6.2772e-03,  1.9013e-02, -3.4680e-03,
            -3.3491e-02,  2.4230e-02, -3.8064e-02,  3.1807e-02,  3.2885e-02,
            -1.4008e-02, -2.2544e-02, -1.3461e-02, -5.2985e-04,  3.4565e-02,
             2.2960e-02, -1.5113e-02,  1.2364e-02,  2.9868e-02, -8.8774e-03,
            -2.9295e-02,  2.8589e-02, -1.1447e-02, -4.4494e-03,  3.6675e-02,
            -4.0005e-02,  4.0165e-02, -1.8027e-02, -2.6828e-02,  3.4661e-02,
            -4.0563e-02,  1.6194e-02,  4.0141e-02,  3.8339e-02, -3.7378e-02,
            -3.5936e-02, -1.2882e-02,  2.0961e-02, -4.1095e-02, -1.4003e-02,
             1.5604e-03,  2.6667e-03,  2.2384e-03, -1.5536e-02, -6.7732e-04,
             3.6904e-02, -3.4456e-02, -4.0145e-02,  3.2038e-02, -3.1337e-02,
            -2.7019e-03, -2.1952e-02, -3.2226e-02, -3.0426e-02, -2.2712e-02,
             2.1189e-02,  3.0328e-02, -3.2485e-02,  3.7748e-02,  1.1909e-03,
             2.8606e-02, -1.8905e-02, -2.3313e-02,  8.7787e-03, -5.1719e-04,
             2.7339e-03,  2.9143e-02, -3.1501e-02, -8.9803e-03, -4.0308e-02,
             1.6124e-02, -4.1005e-03,  2.4144e-02,  1.1426e-02,  3.6999e-02,
             1.7658e-02, -1.8783e-03,  9.3975e-04, -3.6401e-02, -2.6279e-02,
             2.0788e-02,  2.5431e-03, -2.3389e-02,  2.8518e-02,  2.2169e-02,
             2.7188e-03,  2.5363e-02,  1.5379e-02, -1.8507e-04, -1.0163e-02,
            -5.8487e-03, -2.7887e-02, -2.1268e-02, -4.3728e-02, -1.8425e-02,
             3.6177e-02, -2.5185e-02, -2.7398e-02, -4.0522e-02, -3.2207e-02,
             3.8161e-02, -4.1884e-02,  1.4452e-02, -2.9603e-02,  1.8466e-02,
            -8.1972e-03, -3.5820e-02,  1.9415e-02,  1.5580e-02, -2.1561e-02,
             4.2275e-02, -2.5455e-02, -1.8684e-02,  1.9697e-02,  1.0054e-02,
             1.7712e-02, -3.9250e-02, -2.6507e-04, -2.2316e-02, -6.9716e-03,
             6.9062e-03,  3.9247e-02,  4.6194e-04, -1.9644e-02, -8.0433e-03,
             4.0123e-02,  4.0256e-02,  3.1510e-02, -2.2922e-02, -2.1645e-02,
            -1.2244e-02,  2.9483e-02,  1.2072e-03,  3.2402e-02,  4.0311e-02,
            -3.2406e-03,  4.1794e-02, -4.3893e-02,  1.2676e-02, -1.0208e-02,
             2.8095e-02, -4.2569e-02, -7.5497e-03,  2.3483e-02, -2.5475e-02,
             2.5934e-02, -1.0603e-02,  4.0301e-02,  3.3295e-02, -5.6134e-03,
            -2.9092e-02, -2.0971e-02, -3.8250e-02, -1.3003e-02,  2.5778e-02,
             1.7962e-02, -4.2304e-03,  4.4162e-02,  2.9964e-02,  3.2598e-03,
            -2.2691e-02, -1.2638e-02,  2.3293e-02,  1.7401e-02, -3.6645e-02,
            -1.2597e-02, -3.5031e-02,  4.3186e-02,  6.0022e-03, -2.4539e-02,
             2.1690e-02,  1.4755e-02,  1.6306e-02,  3.9170e-02,  7.2215e-03,
             3.9204e-02,  2.5716e-02,  3.5606e-02,  2.0549e-02, -1.3898e-02,
            -1.3565e-02,  1.7732e-02, -4.3100e-02, -2.8189e-03,  2.8215e-02]))])



```python
print(model2.state_dict())
```

    OrderedDict([('conv1.weight', tensor([[[[ 1.2353e-02, -3.5713e-02, -4.6419e-02,  ..., -5.4487e-04,
               -2.5580e-02,  9.5039e-03],
              [-1.4113e-02, -2.1687e-02, -7.3655e-02,  ..., -2.4545e-02,
                3.7708e-02,  2.4177e-02],
              [-2.6923e-02, -2.2991e-03,  6.8104e-03,  ...,  3.0040e-02,
                2.5846e-02,  1.0737e-03],
              ...,
              [ 1.7352e-02, -2.8815e-02, -3.1425e-02,  ..., -1.4524e-02,
               -1.5380e-02,  1.5295e-02],
              [ 4.0697e-02, -6.9606e-03,  3.0921e-02,  ...,  4.7491e-03,
                2.9805e-03, -2.2234e-02],
              [-3.6224e-02, -1.5144e-02,  1.1833e-02,  ...,  5.6690e-03,
               -6.7935e-03,  7.5249e-03]],
    
             [[ 1.0802e-02, -1.3089e-02, -1.0400e-02,  ...,  5.3511e-03,
                1.5235e-02, -2.3676e-02],
              [ 5.4103e-03,  1.6151e-02,  4.3026e-02,  ...,  3.7497e-02,
                4.3525e-02, -2.6875e-02],
              [ 3.1831e-02,  5.4097e-02, -1.4268e-02,  ..., -3.4951e-02,
                3.2749e-02, -1.7871e-02],
              ...,
              [-5.7626e-02, -1.8986e-02,  6.0225e-03,  ...,  2.7881e-02,
               -7.7839e-03, -6.0318e-03],
              [ 1.9256e-02,  3.0078e-02,  3.5863e-02,  ...,  3.4886e-02,
                2.4019e-02,  8.8592e-03],
              [ 3.7452e-02,  5.7697e-03,  1.0191e-03,  ..., -2.0031e-02,
               -4.0363e-03, -1.6401e-02]],
    
             [[ 2.5891e-02,  1.7239e-02,  3.0213e-02,  ...,  1.0534e-02,
                5.1747e-02, -7.5586e-03],
              [-1.0796e-03,  1.6125e-02, -1.9219e-02,  ..., -2.1511e-02,
                3.4048e-02, -3.2933e-02],
              [ 7.1547e-03,  4.2137e-02, -9.0451e-03,  ...,  6.6442e-03,
                2.5878e-02,  2.4297e-02],
              ...,
              [-2.2563e-03,  7.9977e-04,  2.3270e-02,  ..., -3.2405e-02,
                6.0747e-04, -1.4864e-02],
              [-1.2239e-02, -1.9019e-02,  2.0364e-02,  ..., -4.4587e-02,
               -1.4965e-03, -7.7182e-03],
              [-4.3396e-02,  1.7862e-02, -1.6113e-03,  ...,  2.7045e-02,
                7.1188e-03,  1.3907e-02]]],


​    
            [[[ 5.5103e-03,  1.3440e-02, -5.4135e-02,  ...,  1.5246e-02,
                7.0073e-03, -6.8188e-03],
              [ 2.5464e-02,  1.9965e-03,  1.1401e-02,  ...,  2.2967e-02,
               -2.3158e-02, -4.8028e-02],
              [-3.0861e-02,  4.3261e-02,  2.2991e-04,  ...,  7.2141e-03,
               -2.5534e-03, -1.6390e-02],
              ...,
              [-2.2476e-03,  1.5484e-02, -1.8076e-02,  ...,  4.2129e-02,
               -2.6390e-02,  1.1068e-02],
              [-1.4368e-02, -2.3337e-02, -3.3074e-02,  ..., -4.1228e-02,
               -2.1975e-02,  1.5222e-02],
              [ 1.0343e-03,  1.2913e-02,  1.4365e-02,  ..., -1.7734e-02,
               -2.8202e-02,  1.5239e-02]],
    
             [[-1.9221e-02, -4.1880e-02, -3.4249e-02,  ..., -2.2087e-02,
               -9.4479e-03, -4.2507e-02],
              [ 1.5276e-02, -1.7290e-02, -5.0062e-03,  ..., -3.3725e-02,
                8.4538e-02,  7.4596e-02],
              [ 1.5014e-02,  1.3399e-02, -1.5247e-02,  ...,  6.1760e-03,
                1.0639e-03, -3.7298e-03],
              ...,
              [-1.7920e-02,  4.6874e-03, -1.3490e-02,  ...,  8.3387e-03,
                1.7101e-02, -2.1144e-02],
              [ 1.6083e-02, -2.8346e-02, -1.0181e-02,  ...,  8.7736e-03,
                1.7608e-02,  1.3936e-03],
              [-1.3562e-02, -1.6548e-02, -1.0019e-02,  ..., -3.1214e-03,
               -4.5485e-02,  1.8684e-02]],
    
             [[ 1.6536e-03, -6.2918e-02,  3.2885e-02,  ..., -1.2632e-02,
               -1.9542e-02, -4.5302e-02],
              [-2.3802e-02, -3.7273e-03,  8.8764e-03,  ...,  1.1835e-03,
                3.5745e-02,  8.3474e-03],
              [-1.8022e-02,  1.2632e-02,  4.6297e-02,  ..., -1.1841e-02,
                1.9780e-02, -9.1604e-03],
              ...,
              [ 1.4568e-02, -6.0503e-02,  3.2855e-02,  ..., -1.6141e-02,
               -3.4471e-03,  3.1018e-02],
              [-1.8240e-02, -2.3850e-02, -1.0398e-02,  ..., -3.8220e-02,
               -1.7804e-02, -2.1852e-03],
              [ 1.8869e-03,  3.8349e-02, -6.0193e-03,  ..., -3.2453e-02,
                3.5725e-02,  3.7736e-02]]],


​    
            [[[-3.7067e-02,  2.5998e-02, -1.5807e-03,  ..., -4.3575e-03,
                8.9018e-03, -2.6813e-02],
              [-6.3004e-02,  2.5725e-02,  2.2938e-02,  ..., -6.4212e-03,
               -4.9227e-02,  8.6667e-03],
              [-1.0451e-02,  7.7711e-03,  1.2346e-02,  ...,  2.1020e-03,
                3.6730e-02,  8.9196e-03],
              ...,
              [-1.3248e-02, -1.3267e-02, -1.6068e-02,  ...,  1.0950e-02,
               -5.2188e-03, -4.5454e-04],
              [-1.7372e-02, -1.7221e-02, -3.5878e-02,  ..., -1.0645e-02,
               -5.3494e-03, -2.7517e-02],
              [ 3.0780e-02, -2.9385e-02, -1.5382e-03,  ..., -2.7368e-02,
               -5.5117e-02,  9.2951e-04]],
    
             [[-2.9642e-02,  3.0652e-02, -2.0592e-02,  ...,  3.3554e-03,
               -5.0659e-03,  2.5336e-02],
              [-1.7673e-02,  6.2349e-03,  1.8002e-03,  ...,  4.5638e-02,
                1.4157e-02,  6.8525e-03],
              [ 2.9224e-02, -1.0199e-02,  5.6825e-03,  ..., -4.1673e-03,
                1.3161e-02,  6.3385e-02],
              ...,
              [ 1.3108e-02, -1.1734e-02, -7.6878e-03,  ...,  2.2743e-02,
               -3.8229e-02,  1.6747e-02],
              [ 2.5327e-03,  2.6047e-03,  5.3567e-02,  ..., -2.6054e-03,
               -1.4169e-02,  9.2331e-04],
              [ 1.1231e-03,  4.1346e-02,  7.1592e-03,  ..., -1.8857e-02,
               -2.1284e-02,  3.1787e-03]],
    
             [[-2.2749e-02,  2.8438e-02,  1.7338e-02,  ..., -1.4904e-02,
                8.4677e-03,  2.4564e-02],
              [ 1.9473e-02, -2.8390e-02,  1.3051e-02,  ...,  1.4458e-03,
                6.6112e-03,  2.8940e-02],
              [ 1.2877e-02, -6.3633e-03, -1.5618e-02,  ...,  7.4578e-03,
               -1.8088e-02,  3.8192e-02],
              ...,
              [-6.9419e-03, -2.7709e-02,  4.2880e-02,  ..., -8.6494e-03,
                7.7893e-03,  1.4865e-02],
              [ 3.1384e-03,  3.7284e-03, -1.2248e-02,  ...,  4.7131e-02,
                9.8473e-03,  2.4719e-02],
              [ 9.0937e-03, -9.5125e-03,  3.4045e-02,  ...,  3.5318e-02,
                8.1867e-03, -1.7129e-02]]],


​    
            ...,


​    
            [[[ 1.0484e-04,  4.1647e-02, -4.4440e-02,  ..., -2.5187e-02,
               -2.6889e-02, -2.6403e-03],
              [-1.4718e-02,  3.9153e-02,  2.1888e-02,  ...,  4.7834e-02,
               -4.0136e-02,  1.2690e-02],
              [-2.0801e-02, -4.4061e-03,  9.3861e-03,  ...,  4.2037e-02,
                9.2128e-03, -1.1401e-02],
              ...,
              [ 1.7732e-02,  2.5918e-02,  7.7917e-03,  ..., -5.1466e-02,
               -4.4485e-02,  8.7730e-03],
              [-4.3991e-03,  1.5231e-02,  6.1977e-03,  ..., -2.1098e-02,
               -7.3863e-03,  7.3081e-02],
              [ 2.5606e-02,  2.2408e-02,  1.3843e-02,  ..., -2.2067e-02,
                1.4076e-03,  3.6694e-03]],
    
             [[-4.1789e-02, -1.2482e-02, -6.4251e-03,  ..., -4.3878e-03,
               -7.4479e-03, -1.3693e-02],
              [ 1.2541e-02,  1.0336e-02, -2.9981e-02,  ...,  1.6250e-02,
                1.5679e-02, -2.3492e-03],
              [-1.8501e-02,  3.0069e-03, -2.4815e-02,  ...,  4.0587e-02,
               -4.5073e-02,  1.6156e-02],
              ...,
              [ 1.5642e-02, -5.8200e-04,  5.1297e-03,  ...,  2.7395e-02,
                6.1172e-03,  2.5805e-02],
              [-2.5227e-02,  1.1054e-02, -4.7216e-02,  ..., -2.8864e-02,
                3.1125e-02,  7.3301e-03],
              [ 1.4667e-02, -3.2257e-02,  2.9765e-02,  ...,  1.9084e-02,
               -6.4017e-03,  2.5315e-02]],
    
             [[ 1.7699e-02, -9.3588e-03, -6.6574e-03,  ..., -3.5212e-02,
               -3.1355e-02, -3.9393e-02],
              [ 1.0191e-02,  3.0520e-02, -6.5790e-03,  ...,  1.8725e-02,
                3.6009e-03, -1.1696e-02],
              [-3.4537e-03,  1.8168e-02, -4.9313e-03,  ...,  2.8893e-02,
                1.4810e-02,  1.2410e-02],
              ...,
              [-3.0489e-02,  2.8490e-02, -2.0475e-02,  ...,  2.5340e-02,
               -1.2772e-02,  4.1910e-03],
              [ 1.2047e-02, -1.3051e-02, -1.0325e-02,  ..., -7.2207e-03,
                1.3946e-02, -3.5824e-02],
              [-3.5725e-02,  5.6465e-03, -4.8942e-02,  ...,  2.7280e-02,
               -2.2560e-02,  2.2423e-02]]],


​    
            [[[ 2.5215e-02, -2.6037e-02, -7.1189e-03,  ...,  1.2421e-02,
               -2.3827e-02,  1.2885e-02],
              [ 2.8025e-02,  2.2198e-02, -1.9905e-02,  ..., -2.6249e-02,
                3.4476e-02,  1.1111e-03],
              [ 2.0601e-02,  1.3386e-02,  9.3798e-03,  ...,  1.8897e-02,
                4.5602e-02,  5.2975e-03],
              ...,
              [ 1.6043e-02, -2.6220e-02,  2.8817e-02,  ...,  3.2787e-02,
               -2.0426e-02, -4.5844e-03],
              [ 2.8145e-03,  9.9836e-03,  1.7517e-02,  ...,  9.4083e-03,
               -3.1759e-02,  6.0568e-02],
              [-9.7894e-04,  3.2081e-02,  1.0885e-02,  ..., -2.7470e-02,
                7.8087e-03, -8.2926e-03]],
    
             [[ 1.2586e-02,  1.4858e-02,  2.2889e-02,  ..., -2.5693e-02,
                6.6764e-04, -2.6042e-02],
              [-1.7399e-02, -4.7715e-03,  7.1666e-03,  ...,  4.3989e-02,
               -4.7763e-02, -1.5933e-03],
              [-3.5319e-03,  1.7150e-02, -3.7306e-02,  ...,  2.9311e-02,
                2.1062e-02,  1.7357e-02],
              ...,
              [-1.5538e-02,  3.4753e-03,  1.9574e-02,  ...,  3.0604e-02,
                9.5223e-03,  4.1064e-02],
              [ 1.7160e-02, -1.9599e-02,  3.3547e-04,  ...,  5.6494e-02,
               -3.5979e-02,  8.9146e-03],
              [-1.0057e-03,  7.0559e-03,  5.7961e-03,  ...,  1.1160e-02,
               -2.6502e-02,  8.7206e-03]],
    
             [[-1.1577e-03, -2.7506e-04, -2.8923e-03,  ..., -1.4175e-03,
               -2.3723e-02, -2.3104e-02],
              [ 8.9705e-03,  2.4740e-02, -2.5172e-02,  ..., -8.8374e-03,
               -4.6386e-02, -2.0492e-02],
              [-1.4191e-02, -1.8026e-02, -3.3038e-02,  ..., -2.7241e-02,
                1.5081e-02,  1.5494e-02],
              ...,
              [ 8.4246e-03,  3.8451e-03, -1.7167e-02,  ...,  6.2993e-02,
               -3.7641e-04,  1.7011e-02],
              [-4.9899e-02,  3.7582e-02,  1.6764e-02,  ...,  8.1563e-03,
                5.6094e-03, -2.6289e-03],
              [-1.7065e-02, -8.2094e-03,  2.0225e-02,  ...,  1.5519e-02,
                3.7813e-02,  3.7859e-02]]],


​    
            [[[ 1.1671e-02,  8.9922e-05,  1.3341e-02,  ...,  3.7564e-02,
               -1.7323e-02,  1.5196e-02],
              [-2.3953e-02,  2.9051e-03, -5.9366e-02,  ..., -1.5524e-02,
               -1.0648e-02, -8.8292e-03],
              [ 4.2525e-03, -7.0732e-03, -5.1372e-04,  ..., -1.2803e-02,
                1.0091e-03, -3.1203e-02],
              ...,
              [ 8.3300e-03, -1.3929e-02,  4.4173e-02,  ..., -4.3772e-03,
                4.5299e-02,  6.0889e-03],
              [-2.7856e-02, -1.4683e-02, -1.5261e-02,  ..., -2.5578e-02,
               -1.2986e-02,  8.5385e-03],
              [ 1.8987e-02,  1.7251e-02,  2.2474e-03,  ...,  5.2491e-03,
                2.7538e-02,  1.9099e-02]],
    
             [[-2.1429e-02, -2.1452e-02, -1.8188e-03,  ..., -1.9078e-02,
               -2.6875e-02,  1.6612e-02],
              [-1.4193e-02,  2.2775e-02, -7.5320e-03,  ...,  4.0329e-02,
                2.3638e-03,  1.8719e-02],
              [ 1.2235e-02,  7.1510e-03, -2.9710e-02,  ..., -3.6395e-02,
               -1.5017e-02, -3.0130e-02],
              ...,
              [-4.3853e-03,  1.2023e-02,  1.9581e-02,  ...,  1.8423e-02,
               -1.5625e-02, -3.9582e-02],
              [ 2.1389e-02, -1.0032e-02,  3.3228e-02,  ...,  5.2692e-02,
                1.8387e-02,  1.1849e-02],
              [-5.2513e-02,  3.9669e-02, -3.7646e-02,  ..., -3.4407e-02,
               -3.2430e-02, -5.8756e-03]],
    
             [[-1.5025e-02,  5.4491e-03,  3.3839e-03,  ..., -4.3292e-02,
               -4.1655e-02, -6.3504e-02],
              [-2.1221e-02, -1.5059e-03,  6.1776e-04,  ...,  4.8302e-03,
                2.3602e-02,  2.6395e-02],
              [-2.4673e-02, -7.1872e-03, -1.2602e-02,  ...,  3.5135e-02,
               -4.5383e-03,  2.2941e-03],
              ...,
              [-2.0872e-02,  1.5452e-02, -3.1187e-02,  ...,  4.5221e-03,
               -2.5530e-02, -1.4707e-02],
              [ 4.1107e-02, -3.1206e-02,  2.8682e-02,  ..., -7.0993e-03,
                3.0871e-02, -1.2732e-02],
              [-7.9013e-03,  2.0054e-02, -6.2115e-03,  ...,  3.4874e-02,
                2.2789e-03,  1.9972e-02]]]])), ('bn1.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])), ('bn1.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('bn1.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('bn1.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])), ('bn1.num_batches_tracked', tensor(0)), ('layer1.0.conv1.weight', tensor([[[[ 3.0539e-02,  1.9263e-02, -9.5153e-03],
              [-1.6868e-02,  8.1200e-02, -5.0737e-02],
              [-1.4941e-02,  9.8156e-02,  2.4128e-02]],
    
             [[-3.5668e-03, -1.7678e-03,  1.0533e-02],
              [-1.2527e-01, -4.3889e-02, -1.9012e-02],
              [-1.1522e-02, -5.6890e-02,  3.5555e-02]],
    
             [[-1.3885e-02, -4.6435e-02,  8.2544e-02],
              [ 4.4329e-02,  9.3898e-02,  2.1418e-02],
              [-2.4573e-02, -2.0268e-03,  4.4004e-02]],
    
             ...,
    
             [[-1.6858e-02,  1.8185e-02,  9.6267e-02],
              [ 2.6345e-02, -5.6629e-02,  5.5235e-03],
              [-3.3296e-02,  9.4378e-02, -6.3945e-02]],
    
             [[ 2.9833e-02,  8.4714e-02,  3.4327e-02],
              [ 8.7805e-02,  1.6598e-03,  1.1578e-01],
              [-1.2608e-02, -7.5151e-03,  5.4985e-02]],
    
             [[-1.2946e-01, -3.8235e-02,  2.0858e-02],
              [ 3.6629e-02, -8.0922e-03,  4.9266e-02],
              [ 5.9833e-02, -1.4829e-01, -7.2516e-03]]],


​    
            [[[ 3.1191e-02, -4.3731e-02, -2.6382e-02],
              [ 1.4370e-01, -1.5643e-02,  3.0076e-02],
              [-3.9352e-02,  3.6075e-02,  1.3779e-03]],
    
             [[-9.7976e-02, -4.2582e-02,  4.7527e-02],
              [-6.0998e-02,  1.4464e-02, -4.8669e-02],
              [-5.6894e-02,  8.1457e-02, -2.4365e-02]],
    
             [[ 1.6716e-02,  1.4763e-04, -6.4492e-02],
              [-4.4191e-02,  1.8873e-01,  8.8353e-02],
              [ 2.9974e-02,  1.6756e-02,  2.0861e-02]],
    
             ...,
    
             [[ 1.2659e-01,  5.4396e-02, -3.1809e-02],
              [-2.4254e-02, -2.8308e-02, -5.3568e-02],
              [ 7.3916e-02,  1.2409e-01,  1.6906e-02]],
    
             [[ 8.7342e-02,  4.5410e-02, -1.6811e-02],
              [-1.1112e-02, -3.1272e-02,  5.4239e-02],
              [-3.2455e-02,  4.1273e-02,  5.1288e-03]],
    
             [[-5.1213e-02,  6.2314e-02, -1.4645e-02],
              [ 4.9925e-02, -3.3413e-02, -1.7102e-02],
              [-3.2561e-02,  2.9704e-02,  1.0166e-01]]],


​    
            [[[ 1.7685e-02, -4.6276e-02, -6.1931e-02],
              [-1.6014e-02, -1.3426e-01,  1.3155e-01],
              [-1.0861e-02,  1.0231e-01,  5.3349e-02]],
    
             [[ 4.5723e-02,  4.2911e-02, -8.4300e-03],
              [ 1.5883e-02,  9.9476e-02, -2.7056e-02],
              [ 1.6891e-02, -1.5752e-04, -9.7016e-02]],
    
             [[-1.2807e-01, -5.4600e-02, -2.2462e-02],
              [-6.2221e-03, -1.8678e-02, -9.5671e-02],
              [-7.4823e-02,  2.1588e-02, -4.0575e-02]],
    
             ...,
    
             [[ 5.5803e-03, -1.7282e-02,  3.0226e-02],
              [ 5.1910e-02,  6.6598e-02,  1.0962e-01],
              [-8.4714e-02,  7.3650e-02, -2.5640e-02]],
    
             [[ 8.5122e-02,  4.1985e-02, -1.2679e-03],
              [-1.4886e-01,  4.8878e-02, -2.1148e-02],
              [-8.1039e-02, -7.3524e-02, -3.2161e-02]],
    
             [[-8.3176e-02, -5.2273e-02,  7.0170e-02],
              [-9.2101e-02, -1.6115e-03, -8.0622e-02],
              [ 6.2005e-02,  1.0547e-01, -5.1940e-02]]],


​    
            ...,


​    
            [[[ 4.6587e-02,  3.1479e-02, -3.1340e-02],
              [ 6.5531e-02, -5.0680e-02,  5.0297e-02],
              [-3.1221e-02,  4.4492e-02, -4.1228e-02]],
    
             [[-3.2066e-02, -3.0490e-02,  6.1165e-02],
              [-7.8898e-03, -8.2159e-02, -2.2750e-02],
              [ 3.4592e-02,  1.3200e-01,  8.1046e-02]],
    
             [[-5.5585e-02, -5.0348e-03, -1.4225e-02],
              [ 2.0807e-02,  2.7146e-02, -9.3803e-02],
              [ 4.9110e-02, -1.8202e-02, -6.5716e-02]],
    
             ...,
    
             [[ 6.8095e-02, -8.5798e-02,  5.9580e-02],
              [-2.2132e-03, -6.3908e-02, -3.7729e-02],
              [-3.3713e-02,  4.3529e-02,  2.3565e-02]],
    
             [[-1.2734e-02,  4.0865e-02, -2.0637e-02],
              [ 2.6333e-02,  3.1064e-02, -1.1274e-01],
              [ 4.4891e-02, -5.4836e-02, -1.0355e-01]],
    
             [[ 1.7384e-02,  8.2354e-02,  3.6510e-02],
              [-7.5621e-03,  5.1190e-02, -1.0679e-01],
              [-2.5223e-02,  1.5098e-02, -8.4929e-02]]],


​    
            [[[-2.3767e-02, -1.9151e-02, -1.0226e-01],
              [ 8.5812e-02, -6.7412e-02, -6.5305e-02],
              [-3.5956e-02,  2.2744e-02, -1.7631e-02]],
    
             [[-9.5714e-02, -7.8447e-02, -3.2280e-02],
              [ 7.4174e-02, -4.2331e-02, -9.6931e-02],
              [-1.7788e-02, -1.7947e-02, -2.3771e-02]],
    
             [[-1.7588e-02, -1.0841e-01, -1.7130e-02],
              [-1.1424e-01, -2.8594e-02,  2.8611e-02],
              [ 3.2761e-02, -6.3870e-02,  5.6570e-02]],
    
             ...,
    
             [[-5.4793e-02, -8.1073e-02, -3.8775e-02],
              [ 5.4089e-02,  8.1337e-02, -1.2124e-01],
              [ 9.2851e-02,  1.0174e-02,  1.1949e-03]],
    
             [[ 1.2050e-02,  8.1603e-02, -1.9838e-02],
              [-2.0533e-02,  4.8026e-02,  1.2331e-02],
              [-3.0673e-03,  1.5969e-02, -7.5717e-02]],
    
             [[ 2.3808e-03, -4.0620e-02,  5.3948e-02],
              [-1.8374e-02,  7.5871e-03,  5.1573e-02],
              [-4.2799e-02, -2.3825e-02,  9.0677e-03]]],


​    
            [[[-1.4914e-01,  6.0750e-02, -1.5681e-02],
              [ 8.3451e-02, -1.2266e-02,  1.3369e-02],
              [-1.2846e-01, -3.8551e-02,  9.5243e-03]],
    
             [[ 6.4532e-02, -9.8366e-03,  5.3057e-02],
              [ 8.9660e-02,  1.6758e-03,  3.0947e-02],
              [ 1.0565e-01, -1.0911e-01, -3.9327e-02]],
    
             [[-3.4817e-02,  1.1678e-01, -1.0543e-02],
              [ 4.7430e-02,  2.3337e-02,  6.9885e-02],
              [ 2.7513e-02, -8.1519e-03, -7.1742e-02]],
    
             ...,
    
             [[ 3.2491e-02, -7.0260e-02, -3.3710e-02],
              [ 1.0237e-02,  9.9978e-02,  2.9672e-02],
              [-9.3573e-03,  1.0182e-01,  1.6146e-02]],
    
             [[ 1.6679e-01,  2.2240e-02, -8.0446e-02],
              [-7.6930e-02, -4.4717e-02,  3.0263e-02],
              [ 6.5176e-02,  1.7422e-03, -1.6081e-02]],
    
             [[ 5.1144e-02, -6.7699e-02, -7.4803e-02],
              [-6.8858e-03,  8.3474e-02, -1.4611e-02],
              [ 8.9938e-03, -5.9545e-02,  6.8011e-02]]]])), ('layer1.0.bn1.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])), ('layer1.0.bn1.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer1.0.bn1.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer1.0.bn1.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])), ('layer1.0.bn1.num_batches_tracked', tensor(0)), ('layer1.0.conv2.weight', tensor([[[[ 0.0083, -0.0330,  0.0239],
              [ 0.1444,  0.0189,  0.0167],
              [-0.0436,  0.0294,  0.0107]],
    
             [[-0.1209, -0.0254, -0.0941],
              [-0.0487, -0.0630, -0.0906],
              [ 0.0006,  0.0342,  0.0982]],
    
             [[-0.0041,  0.0920, -0.0070],
              [-0.0273,  0.0331, -0.0145],
              [-0.1147, -0.0142,  0.0629]],
    
             ...,
    
             [[-0.0096, -0.0182, -0.0159],
              [ 0.1239, -0.0338,  0.0627],
              [-0.0305, -0.0864,  0.0739]],
    
             [[-0.0162, -0.0785, -0.1316],
              [ 0.0478,  0.0196,  0.0629],
              [ 0.0621, -0.0325,  0.0875]],
    
             [[ 0.0043, -0.1119,  0.0232],
              [ 0.0625,  0.1038, -0.0756],
              [ 0.0425, -0.0416,  0.0315]]],


​    
            [[[-0.0910, -0.0469,  0.0288],
              [ 0.0337, -0.0080, -0.0733],
              [ 0.0270, -0.0115, -0.0522]],
    
             [[-0.0318,  0.0210,  0.0072],
              [ 0.0030,  0.0356,  0.0582],
              [-0.0695,  0.0288,  0.0182]],
    
             [[-0.0327,  0.0090, -0.0537],
              [-0.0192,  0.0546, -0.0595],
              [-0.0541,  0.0084,  0.0343]],
    
             ...,
    
             [[ 0.0542,  0.0615, -0.0681],
              [-0.0219,  0.1173, -0.0402],
              [ 0.0097, -0.0537,  0.0559]],
    
             [[-0.0379,  0.0716, -0.0035],
              [ 0.0098,  0.0187,  0.0190],
              [-0.0257, -0.0989, -0.0778]],
    
             [[ 0.0420,  0.0235,  0.0480],
              [-0.0723, -0.0717, -0.0880],
              [ 0.0378, -0.0219,  0.0515]]],


​    
            [[[-0.0492,  0.0226,  0.0223],
              [-0.0254, -0.0417,  0.0176],
              [-0.0005, -0.0251, -0.0185]],
    
             [[-0.0328, -0.0242, -0.0487],
              [-0.0183,  0.0093, -0.0453],
              [ 0.0281, -0.0482,  0.0656]],
    
             [[-0.0157, -0.0337,  0.1204],
              [ 0.0717,  0.0375,  0.0046],
              [ 0.1011, -0.0440,  0.0141]],
    
             ...,
    
             [[-0.0673, -0.0259, -0.0246],
              [-0.0129,  0.0625, -0.0236],
              [ 0.0643,  0.0036, -0.0228]],
    
             [[-0.0136,  0.0921, -0.0378],
              [-0.0193, -0.0240, -0.0241],
              [-0.0187, -0.0062, -0.0296]],
    
             [[ 0.1154, -0.0782,  0.0653],
              [ 0.0156, -0.0046,  0.0296],
              [ 0.1194,  0.0948, -0.0432]]],


​    
            ...,


​    
            [[[ 0.0365,  0.0108,  0.0052],
              [-0.0331, -0.0711,  0.0663],
              [ 0.0771,  0.0402,  0.0853]],
    
             [[-0.0210,  0.0714,  0.0206],
              [ 0.0606, -0.0247, -0.0115],
              [-0.0478, -0.0496,  0.1077]],
    
             [[-0.0271,  0.0521,  0.0589],
              [-0.0165,  0.0817, -0.0201],
              [-0.0580, -0.0378, -0.0737]],
    
             ...,
    
             [[ 0.0976,  0.0284,  0.0424],
              [-0.0407, -0.0614, -0.0880],
              [-0.0502, -0.0829,  0.0614]],
    
             [[-0.0403,  0.0362, -0.0232],
              [-0.0203, -0.0789,  0.0276],
              [-0.0240,  0.0607,  0.0470]],
    
             [[-0.0507, -0.0425, -0.0552],
              [ 0.0509,  0.0233, -0.0280],
              [-0.1407, -0.0146, -0.0659]]],


​    
            [[[-0.0501,  0.0019,  0.0723],
              [-0.0173,  0.0494, -0.0947],
              [ 0.0874, -0.0464, -0.0561]],
    
             [[ 0.1013,  0.0606, -0.0128],
              [ 0.0755, -0.0198,  0.0805],
              [-0.0607,  0.0667,  0.0699]],
    
             [[-0.0165,  0.0337, -0.0096],
              [-0.0110, -0.0783,  0.0059],
              [-0.0598, -0.0082, -0.0286]],
    
             ...,
    
             [[ 0.0175,  0.0038,  0.0052],
              [-0.0550, -0.0232, -0.0211],
              [ 0.0114,  0.0307,  0.0737]],
    
             [[-0.0087, -0.0385, -0.0230],
              [-0.0099,  0.0482,  0.0191],
              [-0.0498,  0.0061,  0.0107]],
    
             [[-0.0619, -0.0375, -0.0460],
              [-0.0748,  0.0587,  0.0155],
              [-0.0382, -0.1296,  0.0401]]],


​    
            [[[ 0.1032,  0.0737, -0.0121],
              [-0.0740,  0.0474,  0.1038],
              [ 0.0264,  0.0181, -0.0661]],
    
             [[ 0.0893, -0.1562,  0.0106],
              [-0.0010, -0.0407,  0.0568],
              [-0.0355, -0.0063, -0.0501]],
    
             [[ 0.0838, -0.0432, -0.1053],
              [-0.0758, -0.0197,  0.0002],
              [-0.0392,  0.0217, -0.0483]],
    
             ...,
    
             [[ 0.0440,  0.1094,  0.0016],
              [-0.0285,  0.0497,  0.0423],
              [-0.0162, -0.0153, -0.0864]],
    
             [[ 0.0398,  0.0847,  0.0057],
              [-0.0832, -0.0192,  0.0490],
              [ 0.0811,  0.0741, -0.0071]],
    
             [[-0.0179, -0.1659, -0.0107],
              [ 0.0372,  0.0517,  0.0248],
              [-0.0744,  0.0986, -0.0061]]]])), ('layer1.0.bn2.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])), ('layer1.0.bn2.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer1.0.bn2.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer1.0.bn2.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])), ('layer1.0.bn2.num_batches_tracked', tensor(0)), ('layer1.1.conv1.weight', tensor([[[[-0.1661, -0.0325,  0.0481],
              [ 0.0180,  0.0693,  0.0220],
              [-0.0003, -0.0421,  0.1044]],
    
             [[-0.0187,  0.0235, -0.0151],
              [-0.0813,  0.0226, -0.0949],
              [ 0.0097,  0.0214,  0.0410]],
    
             [[-0.0135,  0.0218,  0.0028],
              [ 0.0019, -0.0467, -0.0480],
              [ 0.0054, -0.0323, -0.0525]],
    
             ...,
    
             [[-0.1421,  0.0737,  0.0375],
              [ 0.0137,  0.0635, -0.0040],
              [ 0.0347, -0.0009, -0.0568]],
    
             [[ 0.0090,  0.0043,  0.0387],
              [-0.0073, -0.0276,  0.0165],
              [ 0.0033, -0.1153, -0.0409]],
    
             [[-0.0409, -0.0183,  0.0113],
              [-0.0867,  0.0243, -0.0858],
              [-0.0934,  0.1038, -0.1234]]],


​    
            [[[-0.0077,  0.0054, -0.0260],
              [ 0.0113,  0.0551,  0.0002],
              [-0.0482, -0.0336,  0.0117]],
    
             [[ 0.0295, -0.0811,  0.0316],
              [-0.0098, -0.0331, -0.0203],
              [-0.0209,  0.0733, -0.0355]],
    
             [[-0.0141,  0.0304,  0.0505],
              [-0.0175,  0.0566, -0.0006],
              [-0.0098,  0.0682,  0.0371]],
    
             ...,
    
             [[ 0.0124, -0.0007, -0.0031],
              [-0.0143,  0.1253, -0.0289],
              [-0.0202, -0.0201, -0.0772]],
    
             [[-0.0043,  0.0408, -0.0973],
              [-0.0419, -0.0009,  0.0605],
              [ 0.0404, -0.0502,  0.0407]],
    
             [[ 0.0026,  0.0231, -0.0167],
              [ 0.0552, -0.0043,  0.0487],
              [ 0.0675,  0.0704, -0.0067]]],


​    
            [[[-0.0854,  0.1115,  0.0739],
              [ 0.0499, -0.1231,  0.0368],
              [-0.0748,  0.0232, -0.0357]],
    
             [[ 0.0052,  0.0502, -0.0465],
              [ 0.0301,  0.0489,  0.0320],
              [ 0.0301, -0.0610,  0.0776]],
    
             [[ 0.0165,  0.0352,  0.0138],
              [ 0.0055,  0.0054, -0.0178],
              [ 0.0606,  0.0947,  0.0279]],
    
             ...,
    
             [[-0.0012, -0.0921,  0.0333],
              [-0.0367, -0.1523, -0.0472],
              [ 0.0398,  0.0314, -0.1023]],
    
             [[ 0.1261, -0.0244, -0.0582],
              [-0.1445, -0.0890, -0.1329],
              [-0.0689,  0.1158,  0.0307]],
    
             [[-0.0027, -0.0194,  0.0237],
              [ 0.0144, -0.0117,  0.0261],
              [-0.0502, -0.0103, -0.0098]]],


​    
            ...,


​    
            [[[ 0.1260,  0.0500, -0.0215],
              [-0.0723, -0.1556, -0.0259],
              [-0.0719, -0.0888, -0.0642]],
    
             [[ 0.0804,  0.0390, -0.0229],
              [-0.0479, -0.0119,  0.0435],
              [ 0.1310,  0.0729, -0.0791]],
    
             [[ 0.0423, -0.1180, -0.0675],
              [-0.0425, -0.1203,  0.0044],
              [ 0.0638,  0.0859,  0.0578]],
    
             ...,
    
             [[ 0.0593,  0.0736, -0.0529],
              [-0.0108, -0.0475, -0.0877],
              [-0.0467,  0.0087,  0.0352]],
    
             [[ 0.2153, -0.1379,  0.0200],
              [-0.0499,  0.0454, -0.1016],
              [ 0.0157, -0.0685,  0.1457]],
    
             [[-0.0687,  0.0630, -0.0864],
              [-0.1005,  0.0715,  0.0174],
              [-0.0171,  0.0036, -0.0891]]],


​    
            [[[-0.0446, -0.0626, -0.0922],
              [-0.0193,  0.0444,  0.0487],
              [ 0.0412,  0.0356, -0.0672]],
    
             [[-0.0293,  0.0417, -0.0478],
              [-0.0681, -0.0323, -0.0284],
              [-0.0575, -0.0768,  0.0582]],
    
             [[-0.0893, -0.0532, -0.0345],
              [-0.0412, -0.0468,  0.0752],
              [-0.0912,  0.0663,  0.0715]],
    
             ...,
    
             [[ 0.1167,  0.0335,  0.0335],
              [-0.0106, -0.1084, -0.0503],
              [ 0.0413,  0.0730, -0.0194]],
    
             [[ 0.1289,  0.0234, -0.0755],
              [ 0.0372,  0.0043,  0.0784],
              [-0.0068,  0.0358,  0.0871]],
    
             [[ 0.0393,  0.0068,  0.0128],
              [ 0.0424, -0.0368, -0.0324],
              [ 0.0648,  0.0259,  0.0306]]],


​    
            [[[ 0.0691,  0.0583,  0.0003],
              [-0.1522, -0.0230, -0.0126],
              [-0.0031,  0.0124,  0.1019]],
    
             [[ 0.0132,  0.0181, -0.0091],
              [ 0.0380, -0.1025, -0.0396],
              [ 0.1248, -0.0299,  0.0515]],
    
             [[-0.0309, -0.0312, -0.0271],
              [ 0.0667,  0.0953, -0.0649],
              [ 0.0105, -0.0209, -0.0163]],
    
             ...,
    
             [[-0.1490,  0.0701, -0.0396],
              [-0.0523,  0.0123, -0.0107],
              [ 0.1744,  0.0265,  0.0721]],
    
             [[ 0.0931,  0.0099,  0.0383],
              [ 0.0188, -0.0077, -0.0936],
              [-0.0108,  0.0174,  0.1103]],
    
             [[-0.0816, -0.0310, -0.0625],
              [ 0.0325, -0.0471,  0.0362],
              [-0.0239, -0.0185,  0.0184]]]])), ('layer1.1.bn1.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])), ('layer1.1.bn1.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer1.1.bn1.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer1.1.bn1.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])), ('layer1.1.bn1.num_batches_tracked', tensor(0)), ('layer1.1.conv2.weight', tensor([[[[-0.0418,  0.0517, -0.0367],
              [-0.1072,  0.0787, -0.0783],
              [ 0.0037, -0.0077,  0.0740]],
    
             [[-0.0156, -0.0413, -0.0214],
              [ 0.0883, -0.1097,  0.0673],
              [ 0.0476, -0.1062,  0.0265]],
    
             [[-0.0783, -0.0212, -0.0339],
              [-0.0383,  0.0644,  0.0198],
              [ 0.0287,  0.0573, -0.0919]],
    
             ...,
    
             [[ 0.0576,  0.0018, -0.0310],
              [-0.0161, -0.1002,  0.0616],
              [-0.0237,  0.0053,  0.0048]],
    
             [[-0.0185, -0.0040,  0.0640],
              [-0.0283,  0.0555,  0.0292],
              [ 0.0827, -0.0251,  0.0036]],
    
             [[ 0.0546,  0.1813, -0.0761],
              [ 0.0566, -0.0218, -0.1569],
              [ 0.1152, -0.0317,  0.0405]]],


​    
            [[[-0.1422,  0.0019, -0.1374],
              [ 0.1368, -0.0819, -0.0020],
              [-0.0233,  0.0156,  0.0488]],
    
             [[-0.0368,  0.0828, -0.0558],
              [ 0.0523, -0.0483, -0.0482],
              [ 0.1053,  0.1068,  0.0187]],
    
             [[ 0.0198, -0.0718, -0.1419],
              [-0.0948,  0.0147,  0.0190],
              [ 0.0108, -0.0117, -0.0271]],
    
             ...,
    
             [[ 0.0678,  0.0224,  0.0125],
              [ 0.0632, -0.0364,  0.0726],
              [ 0.0148, -0.1674, -0.1287]],
    
             [[ 0.0382,  0.0220, -0.0394],
              [-0.0342, -0.0371,  0.0544],
              [-0.0029, -0.0019, -0.0645]],
    
             [[-0.0156, -0.0399, -0.1027],
              [ 0.1213,  0.0268, -0.0537],
              [ 0.0584, -0.0146,  0.0157]]],


​    
            [[[ 0.0946, -0.0692,  0.1115],
              [ 0.0107,  0.0624, -0.0535],
              [-0.0596,  0.0018,  0.0396]],
    
             [[-0.0548,  0.1033,  0.0296],
              [ 0.0219, -0.0318, -0.0630],
              [-0.0180, -0.0005,  0.0548]],
    
             [[ 0.0556,  0.1429,  0.0296],
              [ 0.0171, -0.0699, -0.0222],
              [-0.1353, -0.0404,  0.0345]],
    
             ...,
    
             [[-0.0066, -0.0689, -0.1081],
              [ 0.0955,  0.0188,  0.0044],
              [-0.0013, -0.0772,  0.0168]],
    
             [[-0.0169,  0.0162, -0.0034],
              [-0.0236,  0.0275,  0.0925],
              [-0.0112,  0.0091, -0.0394]],
    
             [[ 0.0500,  0.0612, -0.0636],
              [-0.0369,  0.1176, -0.0574],
              [-0.0291, -0.0182,  0.0071]]],


​    
            ...,


​    
            [[[ 0.1107, -0.0451, -0.0485],
              [ 0.0133, -0.0131,  0.0128],
              [ 0.0743,  0.0387, -0.0319]],
    
             [[-0.0395, -0.0511,  0.0265],
              [ 0.0023,  0.0313,  0.0538],
              [ 0.0274, -0.0821,  0.0272]],
    
             [[ 0.0004,  0.0754, -0.0057],
              [ 0.0763,  0.0108, -0.0086],
              [-0.0390,  0.0788, -0.0507]],
    
             ...,
    
             [[ 0.0911,  0.0784,  0.0418],
              [ 0.0081,  0.0178, -0.0586],
              [ 0.0143,  0.0875, -0.0307]],
    
             [[ 0.1231,  0.0539,  0.0040],
              [ 0.0395, -0.0399, -0.1014],
              [ 0.0648, -0.0134,  0.0969]],
    
             [[-0.0551, -0.0911,  0.0094],
              [-0.0094, -0.1176,  0.0225],
              [ 0.0309, -0.0439, -0.0350]]],


​    
            [[[-0.0802, -0.0111, -0.0389],
              [-0.0039, -0.0396, -0.0477],
              [ 0.0213, -0.0263,  0.0047]],
    
             [[-0.0593, -0.0311, -0.0076],
              [ 0.1850,  0.0092, -0.0523],
              [-0.0179,  0.1118, -0.0099]],
    
             [[-0.0127,  0.0157,  0.0159],
              [ 0.0758, -0.0141, -0.0721],
              [ 0.0239,  0.1099, -0.0094]],
    
             ...,
    
             [[-0.0427,  0.0406,  0.0056],
              [-0.0218, -0.0121, -0.0541],
              [ 0.0533, -0.1114, -0.0181]],
    
             [[-0.0203, -0.0509, -0.0655],
              [ 0.0229,  0.0841,  0.0253],
              [ 0.0395, -0.0941, -0.0103]],
    
             [[-0.0830,  0.0291, -0.0449],
              [-0.0625,  0.0190,  0.0918],
              [-0.0615,  0.0039,  0.0896]]],


​    
            [[[-0.0533,  0.0376, -0.0035],
              [ 0.0514,  0.0254, -0.1093],
              [-0.0729, -0.0984,  0.1304]],
    
             [[-0.0579,  0.0398, -0.0262],
              [-0.0217,  0.0503, -0.0140],
              [-0.0552, -0.0712, -0.0095]],
    
             [[ 0.0142, -0.0578,  0.0958],
              [-0.0318,  0.0626,  0.0492],
              [ 0.0109, -0.0047,  0.0003]],
    
             ...,
    
             [[-0.0039,  0.0532,  0.0530],
              [ 0.0090,  0.0223,  0.0167],
              [-0.0387, -0.0130,  0.0584]],
    
             [[ 0.0535, -0.1143,  0.0704],
              [ 0.0114, -0.0757, -0.0231],
              [ 0.1362, -0.0145, -0.0142]],
    
             [[ 0.0470, -0.0066,  0.0616],
              [ 0.0179,  0.0076,  0.0384],
              [-0.0093, -0.0557, -0.0846]]]])), ('layer1.1.bn2.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])), ('layer1.1.bn2.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer1.1.bn2.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer1.1.bn2.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])), ('layer1.1.bn2.num_batches_tracked', tensor(0)), ('layer2.0.conv1.weight', tensor([[[[ 0.0060, -0.0458,  0.0395],
              [-0.0618, -0.0014, -0.0316],
              [ 0.0437,  0.0058,  0.0027]],
    
             [[-0.0855, -0.0436, -0.0019],
              [-0.0467,  0.0367, -0.0278],
              [-0.0004,  0.0849,  0.0615]],
    
             [[-0.0099,  0.0283,  0.0683],
              [ 0.0167,  0.0170,  0.0051],
              [-0.0412, -0.0289, -0.0280]],
    
             ...,
    
             [[ 0.0478, -0.0383,  0.0187],
              [ 0.0094,  0.0047,  0.0491],
              [ 0.0179,  0.0175, -0.0291]],
    
             [[-0.0653, -0.0411, -0.0138],
              [ 0.1275,  0.0323,  0.0157],
              [-0.0130,  0.0325,  0.0376]],
    
             [[-0.0172, -0.0395,  0.0027],
              [ 0.0210,  0.0518,  0.0195],
              [-0.0436,  0.0678,  0.0457]]],


​    
            [[[-0.0013, -0.0328, -0.0262],
              [-0.0115,  0.0324, -0.0278],
              [-0.0248, -0.0294, -0.0380]],
    
             [[ 0.0403, -0.0017,  0.0553],
              [ 0.0593, -0.0345, -0.0149],
              [ 0.0094,  0.0113,  0.0617]],
    
             [[ 0.0438,  0.0013,  0.0569],
              [ 0.0134,  0.0698,  0.0032],
              [-0.0487,  0.0060, -0.0422]],
    
             ...,
    
             [[-0.0056,  0.0620, -0.0209],
              [-0.0107,  0.0245,  0.0321],
              [-0.0604,  0.0308, -0.0498]],
    
             [[-0.0384,  0.0313,  0.0267],
              [-0.0731,  0.0370,  0.0448],
              [ 0.0489,  0.0586, -0.0123]],
    
             [[-0.0310,  0.0247,  0.0184],
              [ 0.0207, -0.0285, -0.0191],
              [ 0.0201, -0.0094, -0.0130]]],


​    
            [[[-0.0183, -0.0379, -0.0875],
              [-0.0086, -0.0389, -0.0356],
              [ 0.0400, -0.0403,  0.1065]],
    
             [[-0.0492,  0.0258,  0.0319],
              [ 0.0183,  0.0280, -0.0278],
              [-0.0338, -0.1121, -0.0628]],
    
             [[-0.0242, -0.0331, -0.0384],
              [-0.0234, -0.0100, -0.0630],
              [ 0.0317,  0.0313, -0.0515]],
    
             ...,
    
             [[-0.0236, -0.0411,  0.0166],
              [ 0.0699,  0.0918,  0.0101],
              [-0.0005, -0.0006, -0.0425]],
    
             [[-0.0410,  0.0628, -0.0840],
              [ 0.0098,  0.0228, -0.0583],
              [-0.0094,  0.0215, -0.0637]],
    
             [[ 0.0215,  0.0117, -0.0682],
              [-0.0111,  0.0199,  0.0780],
              [ 0.0050,  0.0571,  0.0253]]],


​    
            ...,


​    
            [[[-0.0746, -0.0486, -0.0010],
              [ 0.0341,  0.0851, -0.0946],
              [ 0.0124,  0.0472, -0.0573]],
    
             [[-0.0189,  0.0290, -0.0303],
              [-0.0232, -0.0205, -0.0168],
              [-0.0034,  0.0630,  0.0066]],
    
             [[-0.0389, -0.0413, -0.0489],
              [-0.0304, -0.0109, -0.0292],
              [ 0.0476,  0.0005,  0.0348]],
    
             ...,
    
             [[ 0.0478,  0.0152,  0.0667],
              [ 0.0524, -0.0323,  0.0056],
              [-0.0133, -0.0292,  0.0614]],
    
             [[ 0.0556, -0.0114,  0.0356],
              [-0.0693,  0.0634, -0.0174],
              [ 0.0692,  0.0518, -0.0460]],
    
             [[-0.0132,  0.0179, -0.0121],
              [-0.0056,  0.0573, -0.0743],
              [-0.0128, -0.0058, -0.0049]]],


​    
            [[[ 0.0172,  0.0307,  0.0437],
              [-0.0358, -0.0098,  0.0533],
              [-0.0702, -0.0728,  0.0780]],
    
             [[ 0.0749, -0.0362, -0.0053],
              [ 0.0096, -0.0204, -0.0239],
              [-0.0154, -0.0101, -0.0086]],
    
             [[ 0.0047,  0.0374, -0.0289],
              [-0.0600,  0.0487, -0.0130],
              [-0.0032, -0.0242,  0.0271]],
    
             ...,
    
             [[-0.0029,  0.0010, -0.0515],
              [ 0.0176, -0.0491, -0.0399],
              [-0.0052,  0.0752, -0.0279]],
    
             [[ 0.0449,  0.0155, -0.0454],
              [ 0.0128,  0.0712,  0.0472],
              [-0.0417,  0.0190,  0.0454]],
    
             [[-0.0674,  0.0464,  0.0473],
              [ 0.0133, -0.0986, -0.0194],
              [ 0.0300,  0.0219, -0.0223]]],


​    
            [[[ 0.0609, -0.0621,  0.0276],
              [ 0.0091, -0.0020, -0.0011],
              [ 0.0309, -0.0084, -0.0435]],
    
             [[ 0.0111,  0.0236,  0.0367],
              [ 0.0792,  0.0743, -0.0432],
              [-0.0540,  0.0395,  0.0420]],
    
             [[-0.0225, -0.0245, -0.0029],
              [-0.0392,  0.0383,  0.0899],
              [-0.0118,  0.0049, -0.0263]],
    
             ...,
    
             [[ 0.1031,  0.0167, -0.0020],
              [-0.0125, -0.0907,  0.0373],
              [ 0.0090, -0.0008,  0.0524]],
    
             [[ 0.0812,  0.0085, -0.0226],
              [ 0.0177, -0.0148, -0.0286],
              [-0.0171, -0.0206,  0.0571]],
    
             [[-0.0742,  0.0241,  0.0427],
              [-0.0483, -0.0376, -0.0237],
              [-0.0554, -0.0395, -0.0414]]]])), ('layer2.0.bn1.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1.])), ('layer2.0.bn1.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer2.0.bn1.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer2.0.bn1.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1.])), ('layer2.0.bn1.num_batches_tracked', tensor(0)), ('layer2.0.conv2.weight', tensor([[[[ 0.0540,  0.0808, -0.0557],
              [-0.0042, -0.0145,  0.0696],
              [-0.0208,  0.0225, -0.0438]],
    
             [[ 0.0146,  0.0077, -0.0104],
              [ 0.0063,  0.0570, -0.0525],
              [-0.0059,  0.0452, -0.0325]],
    
             [[ 0.0307,  0.0341, -0.0237],
              [ 0.0053, -0.0322,  0.0116],
              [ 0.0380, -0.0227,  0.0056]],
    
             ...,
    
             [[ 0.0728, -0.0403,  0.0429],
              [ 0.0005,  0.0043,  0.0282],
              [-0.0084, -0.0714, -0.0208]],
    
             [[-0.0540,  0.0761,  0.0295],
              [-0.0189,  0.0028, -0.0063],
              [-0.0500,  0.0112,  0.0140]],
    
             [[ 0.0347,  0.0827,  0.0492],
              [-0.0118, -0.0020,  0.0466],
              [ 0.0434,  0.0436, -0.0186]]],


​    
            [[[ 0.0460,  0.0043,  0.0196],
              [-0.0271, -0.0468,  0.0041],
              [ 0.0331, -0.0697,  0.0376]],
    
             [[ 0.0281, -0.0401,  0.0246],
              [-0.0353, -0.0218,  0.0143],
              [ 0.0669,  0.0624,  0.0319]],
    
             [[-0.0116,  0.0075, -0.0165],
              [ 0.0110, -0.0511,  0.0491],
              [ 0.0134,  0.0530,  0.0903]],
    
             ...,
    
             [[-0.0339,  0.0166,  0.0286],
              [ 0.0027,  0.0117,  0.0407],
              [-0.0431, -0.0342,  0.0097]],
    
             [[-0.0032,  0.0125, -0.0275],
              [-0.0431,  0.0234, -0.0412],
              [ 0.0423,  0.0734, -0.0414]],
    
             [[-0.0598,  0.0072,  0.0379],
              [ 0.0426, -0.0440, -0.0191],
              [-0.0481,  0.0893,  0.0237]]],


​    
            [[[-0.0825,  0.0553,  0.0074],
              [-0.0255, -0.0539,  0.0232],
              [ 0.0644, -0.0174, -0.0372]],
    
             [[ 0.0341, -0.0136,  0.0040],
              [ 0.0033, -0.0074,  0.0289],
              [ 0.0321,  0.0334,  0.0246]],
    
             [[ 0.0643,  0.0417,  0.0225],
              [ 0.0257, -0.0056,  0.0148],
              [ 0.0348,  0.0281, -0.0416]],
    
             ...,
    
             [[ 0.0449,  0.0257, -0.0047],
              [-0.0270,  0.0014, -0.0060],
              [ 0.0515, -0.0391, -0.0946]],
    
             [[ 0.0207,  0.0787,  0.0350],
              [-0.0195,  0.0555,  0.0372],
              [ 0.0180,  0.0108, -0.0047]],
    
             [[-0.0596, -0.0661, -0.0033],
              [ 0.0371,  0.0503, -0.0218],
              [-0.0576, -0.0514,  0.0902]]],


​    
            ...,


​    
            [[[ 0.0294,  0.0230, -0.0115],
              [-0.0338, -0.0647, -0.0426],
              [-0.0279, -0.0551,  0.0729]],
    
             [[ 0.0125,  0.0363,  0.0218],
              [ 0.0022, -0.0080, -0.0459],
              [-0.0155, -0.0217, -0.0062]],
    
             [[ 0.0237, -0.0554,  0.0558],
              [-0.0203,  0.0602, -0.0062],
              [ 0.0857,  0.0023,  0.0523]],
    
             ...,
    
             [[ 0.0596, -0.0441,  0.0076],
              [-0.0520, -0.0061,  0.0128],
              [ 0.0390,  0.0791,  0.0416]],
    
             [[-0.0093, -0.0717, -0.0024],
              [-0.0657, -0.0172, -0.0540],
              [ 0.0390,  0.0569, -0.0246]],
    
             [[-0.0669, -0.0047, -0.0136],
              [-0.0264,  0.0379,  0.0256],
              [ 0.0443, -0.0414,  0.0119]]],


​    
            [[[-0.0158,  0.0465, -0.0227],
              [-0.0108, -0.0593,  0.0290],
              [-0.0309,  0.0075, -0.0199]],
    
             [[-0.0493, -0.0702, -0.0206],
              [-0.0124,  0.0799,  0.0100],
              [-0.0214,  0.0253, -0.0078]],
    
             [[-0.0163,  0.0854,  0.0402],
              [ 0.0191, -0.0416,  0.0141],
              [ 0.0074, -0.0067,  0.0804]],
    
             ...,
    
             [[ 0.0352,  0.0655, -0.0062],
              [ 0.0447,  0.0479, -0.0708],
              [-0.0972, -0.0279,  0.0688]],
    
             [[ 0.0050, -0.0125,  0.0006],
              [-0.0513,  0.0188,  0.0887],
              [-0.0286, -0.0418, -0.0104]],
    
             [[-0.0491,  0.1084,  0.0515],
              [-0.0180, -0.0015,  0.0720],
              [ 0.0138, -0.0039, -0.0229]]],


​    
            [[[-0.0086, -0.0610,  0.0271],
              [ 0.0088,  0.0534, -0.0652],
              [ 0.0101, -0.0364,  0.0920]],
    
             [[ 0.0252, -0.0443, -0.0188],
              [ 0.0025, -0.0267, -0.0080],
              [-0.0067, -0.0207, -0.0606]],
    
             [[-0.0613,  0.0134,  0.0378],
              [ 0.0246,  0.0262, -0.0212],
              [ 0.0537,  0.0398, -0.0308]],
    
             ...,
    
             [[-0.0633, -0.0193, -0.0111],
              [-0.0126,  0.0047,  0.0053],
              [-0.0018,  0.0107, -0.0034]],
    
             [[ 0.0355, -0.0341, -0.0109],
              [-0.0062,  0.0130,  0.0540],
              [-0.0594, -0.0286, -0.0381]],
    
             [[ 0.0183,  0.0292, -0.0305],
              [-0.0375,  0.0597,  0.0681],
              [-0.0246, -0.0031,  0.0534]]]])), ('layer2.0.bn2.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1.])), ('layer2.0.bn2.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer2.0.bn2.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer2.0.bn2.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1.])), ('layer2.0.bn2.num_batches_tracked', tensor(0)), ('layer2.0.downsample.0.weight', tensor([[[[ 0.1134]],
    
             [[-0.1649]],
    
             [[-0.2037]],
    
             ...,
    
             [[ 0.1360]],
    
             [[-0.0981]],
    
             [[ 0.0617]]],


​    
            [[[ 0.0900]],
    
             [[-0.1207]],
    
             [[-0.2714]],
    
             ...,
    
             [[-0.1491]],
    
             [[ 0.1718]],
    
             [[ 0.0035]]],


​    
            [[[-0.1024]],
    
             [[-0.0853]],
    
             [[ 0.1771]],
    
             ...,
    
             [[-0.0016]],
    
             [[-0.1849]],
    
             [[ 0.0911]]],


​    
            ...,


​    
            [[[-0.1319]],
    
             [[ 0.0694]],
    
             [[-0.1359]],
    
             ...,
    
             [[ 0.0161]],
    
             [[ 0.1369]],
    
             [[ 0.1154]]],


​    
            [[[-0.1115]],
    
             [[ 0.1137]],
    
             [[-0.2520]],
    
             ...,
    
             [[ 0.0064]],
    
             [[ 0.0804]],
    
             [[-0.1589]]],


​    
            [[[ 0.0434]],
    
             [[ 0.1527]],
    
             [[-0.1698]],
    
             ...,
    
             [[ 0.0994]],
    
             [[ 0.0780]],
    
             [[ 0.0740]]]])), ('layer2.0.downsample.1.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1.])), ('layer2.0.downsample.1.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer2.0.downsample.1.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer2.0.downsample.1.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1.])), ('layer2.0.downsample.1.num_batches_tracked', tensor(0)), ('layer2.1.conv1.weight', tensor([[[[ 9.4917e-03, -4.3838e-02, -1.4113e-02],
              [-1.9540e-02, -1.5401e-02,  1.4366e-02],
              [-7.2431e-02,  1.1573e-02, -1.3649e-02]],
    
             [[ 3.1076e-02,  1.4257e-02,  3.6470e-03],
              [-4.7784e-02, -5.1621e-02,  3.6865e-02],
              [-3.6935e-02, -2.3126e-02,  2.8439e-02]],
    
             [[ 1.4482e-02,  1.5604e-02, -5.7814e-03],
              [ 2.0195e-02,  3.4757e-03, -8.1251e-02],
              [ 2.8865e-03, -4.2297e-02,  6.1348e-02]],
    
             ...,
    
             [[ 6.1181e-02,  3.1304e-02,  1.2904e-02],
              [-2.4378e-02, -4.7457e-02,  2.4194e-02],
              [ 2.4862e-02, -5.0659e-02, -9.5623e-02]],
    
             [[-3.2832e-02, -4.2360e-02,  2.1370e-02],
              [-1.7944e-03, -1.2385e-01, -4.8749e-02],
              [-3.2802e-02,  8.7864e-02, -2.3640e-02]],
    
             [[ 6.9454e-02, -2.9245e-02, -4.7851e-02],
              [ 3.1639e-02,  1.2180e-02, -5.6808e-02],
              [ 1.1535e-02, -4.1574e-02, -1.1260e-02]]],


​    
            [[[-5.7193e-02,  7.4393e-03,  2.2646e-02],
              [-1.6073e-02, -6.0812e-02,  3.1450e-02],
              [ 1.1325e-02,  7.1660e-03,  1.9514e-02]],
    
             [[-3.6434e-03,  5.9549e-02, -1.9878e-02],
              [ 4.5325e-02,  1.5327e-02,  3.3561e-02],
              [-3.9024e-02,  6.6292e-02, -3.1064e-03]],
    
             [[-5.6671e-03, -1.0653e-02,  1.0467e-01],
              [ 4.3120e-02, -2.2607e-02, -7.7391e-02],
              [ 6.2994e-03, -1.5461e-02, -3.6156e-02]],
    
             ...,
    
             [[ 3.7762e-02,  2.3886e-03, -7.0734e-02],
              [-4.2752e-02,  4.1623e-02,  1.5848e-02],
              [ 1.6811e-02, -8.4648e-02, -8.8035e-03]],
    
             [[ 3.5259e-02,  5.1821e-02, -7.0861e-02],
              [-2.0294e-02,  1.6550e-02,  2.0257e-03],
              [ 6.0949e-02,  1.2421e-02,  7.3805e-02]],
    
             [[ 4.3864e-02, -2.3545e-02,  2.6641e-02],
              [-1.3562e-02, -2.0005e-02, -2.2738e-02],
              [-6.9720e-03,  4.0579e-02,  6.4031e-02]]],


​    
            [[[ 5.4271e-02, -1.2097e-02,  9.9753e-02],
              [ 7.4491e-02,  5.3236e-02,  1.0788e-02],
              [ 4.6727e-03, -1.3132e-02, -5.1397e-03]],
    
             [[ 6.5068e-02, -9.5091e-03, -4.7880e-02],
              [-1.8116e-02,  5.0310e-02, -4.3630e-03],
              [ 3.4612e-03, -4.3647e-02,  1.3044e-02]],
    
             [[ 2.4180e-03,  2.5471e-02,  3.7343e-02],
              [-1.7611e-02, -5.6464e-02, -3.4999e-02],
              [-2.7549e-02, -5.7016e-03, -4.2026e-02]],
    
             ...,
    
             [[-7.9049e-03, -3.4917e-02, -5.0150e-04],
              [-6.1644e-02,  2.9234e-02,  2.4467e-02],
              [ 6.4167e-03,  2.9870e-02,  7.5125e-02]],
    
             [[ 7.6612e-02,  1.1932e-02, -1.4564e-02],
              [-4.4840e-02,  8.0319e-03,  4.2495e-02],
              [-4.8409e-02,  4.5992e-02,  2.3031e-02]],
    
             [[ 3.2587e-02, -5.6621e-02,  6.2170e-02],
              [-3.2940e-02, -1.6148e-02, -7.8749e-03],
              [ 1.5296e-02,  6.6066e-03,  2.1501e-02]]],


​    
            ...,


​    
            [[[ 4.3392e-02,  3.1892e-02, -6.0912e-02],
              [ 3.2236e-02, -6.1438e-02, -4.4012e-02],
              [-3.4353e-02,  6.7961e-02, -5.4611e-02]],
    
             [[ 1.8713e-02, -9.7891e-02, -5.6852e-02],
              [ 2.9484e-02, -4.0038e-02,  5.6397e-02],
              [ 2.2133e-02, -3.3515e-02,  3.2406e-02]],
    
             [[-2.7721e-02,  2.2127e-02,  2.9530e-02],
              [-2.6102e-02, -3.8631e-02,  6.8731e-02],
              [ 1.9735e-02,  2.3008e-02, -2.3933e-02]],
    
             ...,
    
             [[ 4.1398e-02,  2.2786e-02,  2.7265e-03],
              [ 1.0733e-02,  3.9280e-02, -2.9558e-03],
              [-5.1938e-02, -1.9259e-02,  4.2349e-02]],
    
             [[ 7.5985e-03, -9.4925e-02,  2.1317e-02],
              [-1.9697e-02,  3.9288e-02,  1.6268e-02],
              [-8.2106e-02, -5.6089e-03,  9.8829e-02]],
    
             [[ 2.0950e-03, -2.4346e-02,  3.8180e-02],
              [-4.8120e-03,  3.7703e-03,  3.2822e-02],
              [-2.1882e-02, -8.5669e-02, -5.5339e-02]]],


​    
            [[[-3.9782e-02, -2.8178e-02,  2.1350e-02],
              [-1.5101e-02, -6.2741e-02, -4.7504e-02],
              [ 1.9134e-02, -3.2309e-02,  3.7014e-02]],
    
             [[-4.6494e-02,  5.6103e-02,  1.2124e-03],
              [ 1.2678e-02, -2.2464e-02,  3.6343e-02],
              [ 1.7750e-02,  5.7882e-02, -3.4187e-02]],
    
             [[-4.0532e-02, -4.7067e-02, -2.5017e-02],
              [ 3.1092e-02, -2.5320e-02, -4.8343e-02],
              [-7.0592e-03,  6.9279e-02,  4.1107e-03]],
    
             ...,
    
             [[-5.4115e-03, -4.6132e-02, -4.2962e-02],
              [ 2.1316e-02, -2.9461e-02,  8.0669e-02],
              [ 7.3475e-03, -6.2416e-02,  5.8797e-02]],
    
             [[ 2.8009e-02,  9.4438e-02,  2.4128e-02],
              [-5.1240e-03,  3.9849e-02, -1.9139e-05],
              [ 9.9925e-03,  2.3025e-02, -3.0954e-02]],
    
             [[ 1.6193e-02, -5.7257e-02,  4.7540e-03],
              [-5.2892e-02, -2.7952e-02, -2.2088e-02],
              [-2.2044e-02, -5.4004e-02,  5.4337e-02]]],


​    
            [[[-2.6053e-02,  9.5196e-03, -2.1971e-02],
              [ 7.5675e-02, -5.6186e-02, -7.1327e-02],
              [-7.3842e-04, -2.4744e-02,  3.8442e-02]],
    
             [[ 2.0697e-03,  4.5354e-02,  6.5955e-02],
              [ 7.3361e-03,  1.9311e-02,  2.2453e-03],
              [ 1.1895e-02,  1.2448e-02, -1.5129e-02]],
    
             [[-1.0624e-02,  4.9166e-02,  3.1875e-02],
              [ 4.2217e-02,  1.3336e-02, -2.4965e-02],
              [-1.5078e-02, -4.1329e-02,  1.7680e-03]],
    
             ...,
    
             [[ 2.1686e-02, -8.3606e-03,  3.4883e-02],
              [-2.4252e-02, -8.9345e-03,  6.1014e-02],
              [-1.0333e-02, -2.7579e-02,  3.4201e-02]],
    
             [[ 1.1051e-01,  3.1364e-02, -4.1041e-02],
              [-1.1251e-02, -5.9290e-02,  3.4159e-02],
              [-7.5320e-03,  4.0232e-02, -4.2174e-02]],
    
             [[-3.2418e-03, -8.3922e-03,  8.1281e-02],
              [-6.7691e-02,  5.3527e-02, -1.5334e-02],
              [-2.7017e-02,  1.2073e-02,  3.9451e-02]]]])), ('layer2.1.bn1.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1.])), ('layer2.1.bn1.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer2.1.bn1.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer2.1.bn1.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1.])), ('layer2.1.bn1.num_batches_tracked', tensor(0)), ('layer2.1.conv2.weight', tensor([[[[ 0.0438,  0.0212,  0.0536],
              [-0.0553, -0.0061,  0.0488],
              [ 0.0429,  0.0411,  0.0124]],
    
             [[ 0.0867,  0.0072,  0.0142],
              [ 0.0055,  0.0552, -0.0237],
              [ 0.0047,  0.0041,  0.0014]],
    
             [[-0.0873, -0.1168,  0.0350],
              [ 0.0639, -0.0410, -0.0236],
              [ 0.0454,  0.0339,  0.0153]],
    
             ...,
    
             [[ 0.0595, -0.0314,  0.0183],
              [ 0.0088,  0.0639, -0.0579],
              [ 0.0012, -0.0317,  0.0295]],
    
             [[ 0.0010, -0.0198,  0.0331],
              [-0.1408,  0.0007,  0.0637],
              [-0.0242,  0.0030,  0.0096]],
    
             [[ 0.0049, -0.0033,  0.0685],
              [ 0.0282, -0.0911,  0.0314],
              [-0.0009, -0.0623,  0.0361]]],


​    
            [[[-0.0130,  0.0253, -0.0279],
              [ 0.0479, -0.0155,  0.0235],
              [ 0.0929, -0.0080,  0.0621]],
    
             [[ 0.0703,  0.0640, -0.0015],
              [ 0.0293, -0.0201,  0.0015],
              [-0.0222, -0.0073,  0.0475]],
    
             [[ 0.0537, -0.0159,  0.0414],
              [-0.0113, -0.0737,  0.0194],
              [-0.0251, -0.0452,  0.0056]],
    
             ...,
    
             [[ 0.0374,  0.0207, -0.0172],
              [-0.0302, -0.0282, -0.0555],
              [-0.0704,  0.0335,  0.0391]],
    
             [[-0.0483,  0.0278, -0.0649],
              [-0.0218,  0.0291,  0.0120],
              [-0.0715, -0.0882, -0.0135]],
    
             [[-0.0408,  0.0279, -0.0953],
              [-0.0277, -0.0323, -0.0265],
              [-0.0082,  0.0475,  0.0367]]],


​    
            [[[ 0.0643,  0.0171, -0.0050],
              [ 0.0072,  0.0043,  0.0748],
              [-0.0254, -0.1025, -0.0675]],
    
             [[ 0.0136, -0.0239,  0.0070],
              [-0.0154, -0.0906, -0.0549],
              [ 0.0133, -0.0315, -0.0086]],
    
             [[-0.0007,  0.0256,  0.0499],
              [ 0.0102, -0.0533,  0.0108],
              [ 0.0190, -0.0124, -0.0424]],
    
             ...,
    
             [[ 0.0334,  0.0582,  0.0360],
              [ 0.0600, -0.0246,  0.0014],
              [-0.0664, -0.0340, -0.0272]],
    
             [[ 0.0595,  0.0349, -0.0132],
              [ 0.0824, -0.0058,  0.0064],
              [-0.0066,  0.0201, -0.0285]],
    
             [[ 0.0537,  0.0192,  0.0188],
              [ 0.0184,  0.0452,  0.0640],
              [-0.0817,  0.0401, -0.0109]]],


​    
            ...,


​    
            [[[-0.0428, -0.0149, -0.0246],
              [ 0.0046,  0.0200, -0.0761],
              [-0.0081,  0.0070,  0.0307]],
    
             [[-0.0494,  0.0473,  0.0065],
              [-0.0317, -0.0046,  0.0469],
              [ 0.0110, -0.0626, -0.0298]],
    
             [[ 0.0476, -0.0788, -0.0107],
              [-0.0166,  0.0018, -0.0068],
              [ 0.0084,  0.0426,  0.0553]],
    
             ...,
    
             [[ 0.0197,  0.0296, -0.0125],
              [ 0.0059, -0.0097, -0.0440],
              [-0.0721,  0.0200,  0.1105]],
    
             [[ 0.0202,  0.0191,  0.0226],
              [-0.0082, -0.0265,  0.0410],
              [-0.0283,  0.0376, -0.0068]],
    
             [[ 0.0086,  0.0258, -0.0505],
              [ 0.0324, -0.0182, -0.0452],
              [ 0.0141, -0.0192, -0.0145]]],


​    
            [[[ 0.0459, -0.0163,  0.0096],
              [ 0.0127,  0.0464,  0.0216],
              [ 0.0046,  0.0333, -0.0478]],
    
             [[ 0.0362,  0.0332,  0.0251],
              [ 0.0559,  0.0016, -0.0122],
              [-0.0081,  0.0381,  0.0250]],
    
             [[ 0.0286, -0.0459,  0.0419],
              [ 0.0129, -0.0341, -0.0141],
              [ 0.0174, -0.0138, -0.0706]],
    
             ...,
    
             [[ 0.0469, -0.0872, -0.0281],
              [-0.0472, -0.0288, -0.0116],
              [-0.0058,  0.0350, -0.0293]],
    
             [[-0.0359,  0.0015, -0.0225],
              [ 0.0304,  0.0128,  0.0108],
              [ 0.0566, -0.0777,  0.0529]],
    
             [[ 0.0267,  0.0671, -0.0195],
              [-0.0036, -0.0272,  0.0465],
              [ 0.0214, -0.0128,  0.0035]]],


​    
            [[[-0.0081,  0.0398, -0.0227],
              [-0.0413,  0.0186, -0.0158],
              [ 0.0049, -0.0042, -0.0161]],
    
             [[-0.0488, -0.0421, -0.0690],
              [ 0.0391,  0.0251,  0.0164],
              [ 0.0043,  0.0059,  0.0018]],
    
             [[ 0.0076,  0.1163,  0.0076],
              [ 0.0276, -0.0786, -0.0247],
              [ 0.0491, -0.0236,  0.0099]],
    
             ...,
    
             [[-0.0124,  0.0348, -0.0156],
              [-0.0292, -0.0776, -0.0081],
              [ 0.0320, -0.0436,  0.0371]],
    
             [[ 0.0447,  0.0725,  0.0158],
              [-0.1048, -0.0343,  0.0236],
              [-0.0035, -0.0635,  0.0495]],
    
             [[-0.0558, -0.0184,  0.0068],
              [ 0.0774,  0.0130, -0.0256],
              [ 0.0515, -0.0177, -0.0094]]]])), ('layer2.1.bn2.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1.])), ('layer2.1.bn2.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer2.1.bn2.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer2.1.bn2.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1.])), ('layer2.1.bn2.num_batches_tracked', tensor(0)), ('layer3.0.conv1.weight', tensor([[[[ 0.0349,  0.0086,  0.0096],
              [ 0.0413, -0.0102,  0.0055],
              [-0.0163, -0.0677,  0.0016]],
    
             [[ 0.0014, -0.0110,  0.0038],
              [ 0.0337,  0.0020, -0.0030],
              [-0.0236,  0.0057, -0.0509]],
    
             [[ 0.0155,  0.0119,  0.0056],
              [-0.0105, -0.0323,  0.0536],
              [-0.0747, -0.0145, -0.0404]],
    
             ...,
    
             [[ 0.0295, -0.0132,  0.0087],
              [ 0.0296,  0.0195,  0.0187],
              [-0.0496,  0.0062, -0.0463]],
    
             [[-0.0356,  0.0047, -0.0013],
              [ 0.0156, -0.0075, -0.0235],
              [-0.0105, -0.0305,  0.0782]],
    
             [[-0.0231,  0.0091,  0.0305],
              [ 0.0142,  0.0132,  0.0348],
              [ 0.0044,  0.0219,  0.0029]]],


​    
            [[[-0.0212,  0.0364, -0.0290],
              [ 0.0127,  0.0041, -0.0074],
              [-0.0006,  0.0569, -0.0181]],
    
             [[-0.0113,  0.0053, -0.0675],
              [-0.0503, -0.0165, -0.0439],
              [-0.0322, -0.0382, -0.0123]],
    
             [[ 0.0327,  0.0066, -0.0186],
              [-0.0042, -0.0269, -0.0184],
              [-0.0141,  0.0079,  0.0137]],
    
             ...,
    
             [[-0.0125, -0.0250, -0.0081],
              [-0.0542,  0.0288,  0.0271],
              [-0.0183,  0.0235,  0.0012]],
    
             [[ 0.0596, -0.0349,  0.0526],
              [ 0.0047,  0.0208, -0.0436],
              [ 0.0365,  0.0079, -0.0054]],
    
             [[ 0.0479,  0.0087, -0.0030],
              [-0.0075,  0.0429, -0.0259],
              [-0.0032, -0.0156, -0.0009]]],


​    
            [[[-0.0249,  0.0367,  0.0297],
              [ 0.0061, -0.0402, -0.0070],
              [-0.0449, -0.0183, -0.0054]],
    
             [[ 0.0308,  0.0283, -0.0199],
              [ 0.0424, -0.0101,  0.0193],
              [ 0.0449,  0.0070,  0.0582]],
    
             [[-0.0426, -0.0077, -0.0369],
              [ 0.0001, -0.0265, -0.0589],
              [-0.0601, -0.0479, -0.0013]],
    
             ...,
    
             [[-0.0179, -0.0244, -0.0579],
              [-0.0459, -0.0029, -0.0151],
              [ 0.0263, -0.0004, -0.0187]],
    
             [[ 0.0074, -0.0004,  0.0086],
              [ 0.0284,  0.0654, -0.0165],
              [ 0.0116, -0.0059,  0.0304]],
    
             [[ 0.0535, -0.0324, -0.0140],
              [-0.0323,  0.0213,  0.0131],
              [-0.0326, -0.0430,  0.0530]]],


​    
            ...,


​    
            [[[ 0.0449, -0.0052, -0.0313],
              [-0.0396,  0.0049, -0.0056],
              [-0.0410, -0.0122, -0.0070]],
    
             [[ 0.0247,  0.0044, -0.0206],
              [ 0.0302, -0.0333,  0.0366],
              [ 0.0454,  0.0860, -0.0144]],
    
             [[-0.0065, -0.0059, -0.0134],
              [-0.0098,  0.0045, -0.0063],
              [ 0.0162,  0.0272,  0.0029]],
    
             ...,
    
             [[ 0.0081, -0.0118, -0.0031],
              [ 0.0490, -0.0305,  0.0092],
              [-0.0716, -0.0051,  0.0091]],
    
             [[-0.0138,  0.0322,  0.0029],
              [-0.0223,  0.0339,  0.0149],
              [ 0.0173, -0.0205, -0.0313]],
    
             [[-0.0080, -0.0018, -0.0041],
              [ 0.0237,  0.0120, -0.0249],
              [-0.0533, -0.0087,  0.0407]]],


​    
            [[[-0.0691, -0.0210,  0.0125],
              [ 0.0003,  0.0235, -0.0084],
              [ 0.0596, -0.0081, -0.0231]],
    
             [[ 0.0002,  0.0441,  0.0161],
              [ 0.0233, -0.0274,  0.0003],
              [-0.0047,  0.0399,  0.0414]],
    
             [[ 0.0064, -0.0306,  0.0459],
              [-0.0374,  0.0177, -0.0209],
              [-0.0426, -0.0197, -0.0247]],
    
             ...,
    
             [[-0.0193,  0.0245,  0.0153],
              [-0.0099, -0.0507, -0.0386],
              [ 0.0577, -0.0096,  0.0134]],
    
             [[-0.0476,  0.0122, -0.0419],
              [ 0.0218,  0.0256,  0.0191],
              [-0.0145, -0.0224, -0.0050]],
    
             [[ 0.0110,  0.0047, -0.0037],
              [ 0.0061,  0.0668,  0.0475],
              [ 0.0066, -0.0046, -0.0025]]],


​    
            [[[ 0.0083, -0.0538, -0.0203],
              [-0.0113, -0.0147, -0.0328],
              [ 0.0206, -0.0062, -0.0044]],
    
             [[ 0.0410,  0.0431,  0.0183],
              [-0.0123,  0.0068,  0.0033],
              [-0.0031,  0.0313, -0.0670]],
    
             [[ 0.0212, -0.0293, -0.0204],
              [ 0.0063,  0.0166, -0.0234],
              [-0.0406, -0.0198, -0.0353]],
    
             ...,
    
             [[-0.0170, -0.0230, -0.0056],
              [ 0.0055,  0.0298,  0.0040],
              [-0.0243, -0.0092,  0.0778]],
    
             [[-0.0132,  0.0287,  0.0016],
              [ 0.0520,  0.0449,  0.0065],
              [ 0.0165, -0.0415,  0.0275]],
    
             [[-0.0423, -0.0387, -0.0262],
              [ 0.0171, -0.0358, -0.0033],
              [ 0.0362,  0.0118,  0.0392]]]])), ('layer3.0.bn1.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1.])), ('layer3.0.bn1.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer3.0.bn1.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer3.0.bn1.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1.])), ('layer3.0.bn1.num_batches_tracked', tensor(0)), ('layer3.0.conv2.weight', tensor([[[[-0.0204,  0.0307, -0.0113],
              [ 0.0715, -0.0123,  0.0073],
              [ 0.0461, -0.0622, -0.0495]],
    
             [[-0.0041, -0.0623, -0.0655],
              [-0.0062, -0.0174,  0.0399],
              [-0.0096,  0.0002, -0.0127]],
    
             [[ 0.0214, -0.0274,  0.0407],
              [ 0.0415, -0.0221,  0.0415],
              [ 0.0362,  0.0206,  0.0486]],
    
             ...,
    
             [[-0.0109,  0.0002,  0.0343],
              [ 0.0070,  0.0327,  0.0177],
              [ 0.0101, -0.0220,  0.0310]],
    
             [[ 0.0088,  0.0096, -0.0066],
              [-0.0532, -0.0274, -0.0142],
              [ 0.0370, -0.0454, -0.0089]],
    
             [[-0.0248, -0.0109, -0.0664],
              [ 0.0129, -0.0072,  0.0324],
              [-0.0111,  0.0036,  0.0151]]],


​    
            [[[-0.0192, -0.0985, -0.0125],
              [-0.0136, -0.0548, -0.0440],
              [-0.0898, -0.0056, -0.0413]],
    
             [[ 0.0045,  0.0264,  0.0087],
              [ 0.0075, -0.0535, -0.0234],
              [ 0.0048,  0.0244, -0.0081]],
    
             [[ 0.0031,  0.0129, -0.0103],
              [ 0.0397,  0.0222,  0.0207],
              [-0.0562, -0.1118, -0.0240]],
    
             ...,
    
             [[ 0.0356, -0.0236,  0.0706],
              [ 0.0396,  0.0216, -0.0232],
              [-0.0299, -0.0489,  0.0286]],
    
             [[-0.0415, -0.0207, -0.0064],
              [-0.0407,  0.0791,  0.0062],
              [ 0.0288,  0.0222,  0.0014]],
    
             [[ 0.0111,  0.0380, -0.0231],
              [ 0.0161,  0.0108, -0.0158],
              [-0.0293,  0.0718, -0.0129]]],


​    
            [[[ 0.0088, -0.0482, -0.0320],
              [-0.0327,  0.0047, -0.0238],
              [ 0.0105,  0.0399,  0.0064]],
    
             [[ 0.0056, -0.0405, -0.0146],
              [ 0.0072, -0.0119,  0.0366],
              [ 0.0215,  0.0121, -0.0282]],
    
             [[-0.0020, -0.0566, -0.0365],
              [ 0.0665, -0.0455,  0.0041],
              [-0.0060, -0.0327,  0.0613]],
    
             ...,
    
             [[ 0.0061, -0.0231,  0.0126],
              [-0.0126,  0.0249, -0.0173],
              [ 0.0305, -0.0202, -0.0125]],
    
             [[ 0.0108,  0.0124, -0.0241],
              [-0.0519, -0.0344,  0.0101],
              [ 0.0030,  0.0403, -0.0448]],
    
             [[-0.0054, -0.0195, -0.0558],
              [-0.0163,  0.0378,  0.0286],
              [ 0.0061,  0.0207,  0.0359]]],


​    
            ...,


​    
            [[[ 0.0380, -0.0335, -0.0105],
              [ 0.0251,  0.0047,  0.0110],
              [ 0.0437,  0.0054,  0.0125]],
    
             [[ 0.0091,  0.0064, -0.0246],
              [-0.0438,  0.0140,  0.0633],
              [ 0.0193,  0.0032, -0.0254]],
    
             [[-0.0193,  0.0379,  0.0345],
              [ 0.0015,  0.0637,  0.0273],
              [ 0.0088,  0.0133,  0.0551]],
    
             ...,
    
             [[-0.0201,  0.0015, -0.0151],
              [ 0.0344, -0.0493, -0.0246],
              [-0.0080,  0.0391, -0.0078]],
    
             [[ 0.0100, -0.0149,  0.0163],
              [-0.0002,  0.0105,  0.0341],
              [-0.0005, -0.0172,  0.0095]],
    
             [[-0.0250, -0.0026,  0.0116],
              [ 0.0039, -0.0077, -0.0106],
              [-0.0030,  0.0147,  0.0239]]],


​    
            [[[-0.0267, -0.0428,  0.0060],
              [-0.0337,  0.0093,  0.0431],
              [-0.0431, -0.0147, -0.0194]],
    
             [[-0.0112, -0.0124, -0.0457],
              [ 0.0364,  0.0053, -0.0210],
              [ 0.0062, -0.0032, -0.0576]],
    
             [[ 0.0411, -0.0081,  0.0161],
              [ 0.0104, -0.0017,  0.0217],
              [ 0.0425, -0.0259, -0.0102]],
    
             ...,
    
             [[-0.0566,  0.0281,  0.0561],
              [ 0.0386, -0.0370,  0.0405],
              [ 0.0224,  0.0461,  0.0256]],
    
             [[ 0.0308,  0.0206, -0.0410],
              [-0.0365, -0.0139, -0.0191],
              [-0.0479,  0.0091,  0.0462]],
    
             [[ 0.0405,  0.0053, -0.0278],
              [ 0.0221, -0.0220, -0.0342],
              [-0.0027,  0.0194, -0.0425]]],


​    
            [[[ 0.0046,  0.0232, -0.0319],
              [-0.0342,  0.0621, -0.0501],
              [-0.0247, -0.0112,  0.0576]],
    
             [[-0.0053,  0.0199, -0.0020],
              [-0.0033,  0.0155, -0.0357],
              [ 0.0627,  0.0041,  0.0158]],
    
             [[-0.0559, -0.0015, -0.0111],
              [-0.0504, -0.0118,  0.0309],
              [-0.0191,  0.0127,  0.0020]],
    
             ...,
    
             [[-0.0005, -0.0383, -0.0425],
              [ 0.0177,  0.0012,  0.0175],
              [ 0.0022, -0.0020, -0.0347]],
    
             [[ 0.0148,  0.0054,  0.0406],
              [-0.0098,  0.0169, -0.0666],
              [-0.0345,  0.0198, -0.0046]],
    
             [[-0.0136, -0.0206,  0.0022],
              [-0.0020,  0.0172, -0.0251],
              [-0.0197,  0.0181, -0.0603]]]])), ('layer3.0.bn2.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1.])), ('layer3.0.bn2.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer3.0.bn2.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer3.0.bn2.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1.])), ('layer3.0.bn2.num_batches_tracked', tensor(0)), ('layer3.0.downsample.0.weight', tensor([[[[ 5.1302e-02]],
    
             [[-1.2627e-01]],
    
             [[-1.3083e-01]],
    
             ...,
    
             [[-1.6266e-02]],
    
             [[ 1.1629e-01]],
    
             [[ 1.3444e-01]]],


​    
            [[[-6.1614e-02]],
    
             [[-6.8547e-02]],
    
             [[ 5.8207e-02]],
    
             ...,
    
             [[ 1.1938e-02]],
    
             [[ 2.0041e-02]],
    
             [[ 2.1884e-04]]],


​    
            [[[ 8.9777e-03]],
    
             [[-4.8123e-02]],
    
             [[ 3.0489e-02]],
    
             ...,
    
             [[-4.8910e-02]],
    
             [[ 8.1343e-02]],
    
             [[-8.4297e-03]]],


​    
            ...,


​    
            [[[ 7.5705e-02]],
    
             [[ 1.9363e-01]],
    
             [[ 8.0216e-02]],
    
             ...,
    
             [[ 9.8609e-03]],
    
             [[-2.6596e-01]],
    
             [[-5.2704e-03]]],


​    
            [[[-1.1560e-01]],
    
             [[-1.1692e-01]],
    
             [[ 4.2977e-02]],
    
             ...,
    
             [[ 4.9820e-02]],
    
             [[-1.2323e-01]],
    
             [[ 1.6390e-01]]],


​    
            [[[ 4.4047e-02]],
    
             [[ 7.3217e-02]],
    
             [[ 2.5563e-01]],
    
             ...,
    
             [[-1.6249e-03]],
    
             [[-1.0374e-02]],
    
             [[-7.1804e-03]]]])), ('layer3.0.downsample.1.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1.])), ('layer3.0.downsample.1.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer3.0.downsample.1.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer3.0.downsample.1.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1.])), ('layer3.0.downsample.1.num_batches_tracked', tensor(0)), ('layer3.1.conv1.weight', tensor([[[[-3.4275e-03, -3.2765e-02, -4.8360e-02],
              [ 2.6047e-02,  2.4244e-02,  8.3237e-03],
              [ 8.2318e-03,  2.0020e-02, -2.9594e-04]],
    
             [[-1.2054e-03,  3.2403e-02,  2.0327e-02],
              [ 3.0869e-02,  2.3991e-02, -4.9800e-03],
              [-2.4696e-02, -2.1574e-02,  3.2120e-02]],
    
             [[ 2.4407e-03,  4.5500e-02,  1.7414e-02],
              [-1.5649e-02,  1.2575e-02,  2.1246e-02],
              [-2.9785e-02,  2.3647e-03, -4.6126e-03]],
    
             ...,
    
             [[-8.2805e-05, -3.3302e-02,  1.2111e-02],
              [-4.4140e-02,  2.9691e-02,  2.3848e-02],
              [ 1.4394e-02, -2.0125e-02, -2.3710e-02]],
    
             [[ 1.1411e-02, -2.1530e-02,  4.1833e-02],
              [-3.2720e-02, -2.6466e-03,  5.9094e-02],
              [ 1.2959e-02, -4.3469e-03, -1.8603e-02]],
    
             [[-4.6260e-03,  2.6415e-02,  4.3674e-02],
              [-5.0332e-02,  1.0870e-02,  2.7126e-02],
              [ 2.2240e-02,  5.7367e-02,  1.8207e-02]]],


​    
            [[[-2.3301e-02, -4.3126e-02, -2.3901e-02],
              [-1.3154e-02, -1.4424e-02, -1.7940e-02],
              [-3.6215e-02,  4.5187e-02, -5.8700e-03]],
    
             [[ 4.2955e-03,  4.3265e-02, -2.2313e-03],
              [ 7.0994e-02, -8.8121e-03,  1.3649e-02],
              [-1.0897e-02, -2.3306e-03,  5.2530e-02]],
    
             [[-1.2553e-02,  9.9124e-04, -7.9737e-02],
              [ 4.8922e-03,  3.4057e-02,  1.0713e-02],
              [ 7.8766e-02,  2.7916e-02, -3.0844e-02]],
    
             ...,
    
             [[ 1.7683e-02,  1.9431e-03,  1.0674e-02],
              [-2.0575e-02,  9.2218e-03, -2.6168e-03],
              [ 7.8172e-03, -4.8062e-02, -2.3314e-02]],
    
             [[ 3.6147e-02,  2.6981e-02, -2.2957e-04],
              [-1.5095e-02,  3.7287e-03,  1.0717e-03],
              [-1.3981e-02, -2.9080e-02, -9.6914e-03]],
    
             [[ 1.4152e-02,  4.3945e-02,  1.7273e-02],
              [-4.5323e-02,  4.5672e-03,  6.1765e-02],
              [-3.4651e-02,  1.6901e-02,  3.5999e-03]]],


​    
            [[[ 3.2745e-02, -2.1093e-02,  2.6111e-02],
              [ 3.9376e-02, -4.1046e-03, -6.1104e-03],
              [ 1.3050e-02,  6.6005e-02, -1.5202e-02]],
    
             [[-3.6875e-02,  3.7787e-02,  1.5600e-02],
              [-1.3169e-02, -3.4448e-03,  5.3856e-03],
              [-2.4479e-03,  1.6841e-02,  1.8229e-02]],
    
             [[-1.3233e-02, -3.7768e-02, -1.8962e-02],
              [-2.0258e-02, -9.4328e-03,  2.3798e-02],
              [-4.5409e-02,  8.9546e-03,  9.7676e-03]],
    
             ...,
    
             [[-3.8426e-02,  3.6415e-02, -2.1356e-02],
              [-6.9219e-02,  5.6381e-03, -1.0655e-02],
              [-5.3993e-02, -1.0081e-02,  1.2257e-02]],
    
             [[-6.5070e-02, -3.2924e-03,  5.2459e-02],
              [-2.5407e-02, -3.0754e-02,  5.7905e-03],
              [-3.2969e-02, -4.4555e-02, -1.8686e-02]],
    
             [[-1.5556e-02,  1.0232e-02, -1.9892e-02],
              [ 3.1916e-02, -5.5386e-02, -4.2912e-02],
              [ 3.9086e-02, -1.7610e-02,  3.8135e-02]]],


​    
            ...,


​    
            [[[ 8.9523e-03,  9.0386e-03,  3.7514e-04],
              [-2.3940e-03, -1.2129e-02,  1.1425e-02],
              [ 5.1709e-02,  1.4023e-02, -1.3509e-02]],
    
             [[-2.8031e-02,  2.5869e-02,  4.1954e-03],
              [-2.0250e-03,  8.6634e-03,  3.2324e-03],
              [ 6.4992e-02, -8.3147e-03, -4.1640e-03]],
    
             [[-1.9022e-02, -5.3721e-03, -4.4217e-02],
              [-2.2197e-02, -2.5634e-02, -8.3819e-03],
              [ 3.4498e-02, -3.6383e-02,  4.7910e-03]],
    
             ...,
    
             [[-1.2622e-02, -5.1117e-02, -6.9676e-03],
              [-3.2503e-02, -7.8702e-03, -2.7234e-02],
              [-1.7722e-02, -1.9462e-03, -3.1503e-02]],
    
             [[-1.8082e-02,  1.1581e-02, -2.7600e-03],
              [-4.6376e-02, -4.0566e-03,  5.9485e-02],
              [-4.0068e-02, -1.3325e-03,  3.5468e-02]],
    
             [[-2.7588e-02, -3.6860e-03,  2.1761e-02],
              [-1.0829e-02,  1.4175e-02,  9.4780e-03],
              [ 2.0903e-02,  1.3979e-02, -4.8911e-02]]],


​    
            [[[-4.0552e-02,  2.3846e-02,  4.9954e-02],
              [-1.7661e-02,  1.4004e-02, -2.9632e-02],
              [-3.1077e-02,  6.6514e-03,  2.4366e-02]],
    
             [[ 5.1884e-02,  3.1370e-02, -7.4215e-03],
              [-1.8851e-02,  2.6021e-03, -7.0751e-03],
              [-5.2249e-02, -1.9212e-02,  1.6598e-02]],
    
             [[ 2.6262e-02, -5.4647e-04,  2.4515e-02],
              [ 3.4015e-02,  1.3750e-02, -2.9688e-02],
              [ 2.9974e-02,  3.1654e-02,  2.4101e-02]],
    
             ...,
    
             [[ 8.5331e-03, -2.7333e-02,  2.1504e-02],
              [-2.8443e-02,  2.2886e-02,  5.2746e-02],
              [-3.3169e-02,  6.6165e-02,  1.9914e-02]],
    
             [[ 1.1873e-03, -2.0247e-03,  1.8708e-02],
              [ 1.6251e-02, -1.1317e-02, -2.6039e-02],
              [ 9.7906e-03,  2.3926e-02, -5.7490e-02]],
    
             [[ 6.5744e-02, -1.4836e-02, -3.7426e-02],
              [-8.7107e-03, -2.1662e-02,  6.9513e-03],
              [ 7.7147e-04,  2.2458e-02,  4.4468e-02]]],


​    
            [[[-3.1892e-02, -2.4033e-02,  2.3010e-03],
              [ 3.7565e-02,  1.0014e-02, -2.2165e-02],
              [-2.6159e-03,  2.6453e-02, -4.9073e-02]],
    
             [[ 2.9138e-02, -2.6143e-02, -1.7554e-02],
              [-2.6148e-03,  2.1903e-02, -5.5489e-03],
              [ 5.0457e-02,  2.1847e-02, -4.6261e-02]],
    
             [[-2.3481e-02, -1.0568e-03, -3.2478e-02],
              [-1.7053e-03,  2.4769e-02,  2.6193e-02],
              [ 1.6752e-02, -1.1499e-02,  4.0311e-02]],
    
             ...,
    
             [[-1.3489e-02,  5.3589e-02, -2.1646e-04],
              [ 2.0271e-02, -1.2201e-02,  1.7955e-02],
              [ 4.1370e-02,  2.8870e-02, -3.5185e-02]],
    
             [[-1.3736e-02,  8.2726e-03, -5.6179e-02],
              [ 2.3764e-02, -3.3681e-02,  2.2471e-02],
              [ 1.2665e-02,  3.0401e-02, -2.2962e-02]],
    
             [[-1.3585e-02, -5.9026e-03, -2.0017e-02],
              [ 1.4092e-02,  3.8301e-02, -2.9398e-02],
              [-5.5344e-03,  4.3024e-02,  1.2914e-02]]]])), ('layer3.1.bn1.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1.])), ('layer3.1.bn1.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer3.1.bn1.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer3.1.bn1.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1.])), ('layer3.1.bn1.num_batches_tracked', tensor(0)), ('layer3.1.conv2.weight', tensor([[[[ 3.0450e-02, -1.4764e-02,  1.2484e-02],
              [-3.1183e-02,  3.3724e-02,  1.9898e-02],
              [ 5.9339e-02, -1.7990e-03, -1.9772e-02]],
    
             [[-1.5687e-03, -4.2487e-02, -3.7112e-02],
              [-1.4221e-02, -2.5835e-02,  3.9170e-03],
              [ 6.1574e-03, -1.5033e-02, -1.6639e-02]],
    
             [[ 1.4094e-02,  1.0711e-02,  1.0824e-02],
              [ 2.0790e-02, -2.7543e-02,  1.6675e-02],
              [-1.1795e-02,  1.4660e-02, -1.7106e-02]],
    
             ...,
    
             [[-1.4501e-02,  2.8201e-02, -7.0925e-02],
              [ 3.4783e-02,  1.3036e-02,  1.5069e-02],
              [-4.9781e-02, -1.2876e-02, -5.8367e-02]],
    
             [[ 1.3604e-02,  5.0310e-03,  1.9656e-02],
              [ 1.2169e-02, -1.0567e-02, -3.1374e-02],
              [-3.7885e-02, -3.9021e-02, -4.5678e-04]],
    
             [[-7.7496e-02, -3.9379e-02,  3.9398e-02],
              [ 1.6972e-02,  5.6611e-02, -1.7317e-03],
              [ 3.0593e-02,  6.3763e-02, -2.7644e-03]]],


​    
            [[[-1.3291e-02, -1.5206e-02,  5.2298e-03],
              [ 1.1955e-02, -8.3960e-03, -2.1701e-02],
              [ 2.8219e-03,  3.8049e-02,  6.9573e-03]],
    
             [[ 1.7531e-02, -7.3248e-03,  2.9376e-02],
              [ 2.9584e-02,  5.8657e-02, -2.7732e-02],
              [ 3.5657e-02, -5.7662e-02,  3.0640e-02]],
    
             [[-2.1569e-02,  5.9803e-02,  3.7876e-02],
              [ 3.3871e-02,  4.0264e-02,  1.2637e-02],
              [ 5.3023e-02, -1.1335e-02,  1.7939e-02]],
    
             ...,
    
             [[-1.2867e-02, -6.5142e-03, -2.3125e-02],
              [-3.9135e-02,  1.6539e-02, -3.0539e-02],
              [ 2.1629e-02, -3.8552e-02,  1.1575e-02]],
    
             [[ 3.0371e-02,  5.6315e-03,  1.2514e-04],
              [-8.9490e-03, -5.3495e-02,  1.2492e-02],
              [-3.3766e-02,  6.2749e-02, -3.1363e-03]],
    
             [[ 6.9556e-03,  4.1174e-02,  1.4969e-02],
              [-1.3804e-02,  3.0142e-02,  7.5959e-03],
              [-6.8422e-03,  3.4523e-02, -3.5308e-02]]],


​    
            [[[-1.3718e-02,  2.6032e-02, -3.5351e-02],
              [ 1.2929e-02,  1.9278e-02,  2.6253e-02],
              [-4.4458e-03, -3.0676e-02,  6.2885e-03]],
    
             [[-2.3253e-02,  5.8394e-02, -2.7177e-04],
              [ 9.8116e-05, -3.4065e-02,  8.9029e-03],
              [ 9.4137e-03, -3.1040e-02,  5.1619e-04]],
    
             [[-4.7903e-02,  1.4733e-02,  3.7089e-02],
              [-5.0217e-03,  4.9756e-02, -1.6572e-02],
              [ 3.3901e-03, -6.9980e-03,  1.0569e-02]],
    
             ...,
    
             [[ 1.0835e-02,  1.4543e-02, -2.7965e-02],
              [-2.9713e-03, -5.1880e-02, -3.5625e-03],
              [ 3.2518e-03,  1.9563e-02, -6.8342e-03]],
    
             [[-1.7425e-02,  4.1145e-02, -1.6075e-02],
              [-1.2845e-03, -4.9576e-03,  6.3727e-03],
              [ 2.9496e-04,  1.0430e-02,  9.8068e-03]],
    
             [[ 3.5511e-02,  2.3129e-02,  2.8021e-02],
              [ 1.4639e-02,  4.2938e-03,  1.4175e-02],
              [-1.7044e-03, -3.6358e-02,  4.8874e-02]]],


​    
            ...,


​    
            [[[ 8.3247e-03,  5.5425e-02,  8.3526e-03],
              [-2.4693e-02,  9.9390e-04,  3.3968e-02],
              [-4.3386e-03,  8.9296e-04, -1.1349e-02]],
    
             [[ 5.2219e-03, -3.1748e-02,  7.4649e-04],
              [-7.1650e-03,  6.8017e-03,  7.7711e-02],
              [-2.1689e-02, -2.5007e-02,  5.9812e-02]],
    
             [[-2.8304e-02,  2.6397e-02,  2.8205e-02],
              [ 8.4211e-02,  1.1275e-02,  4.8635e-03],
              [ 1.1111e-02,  2.4489e-02, -2.2332e-03]],
    
             ...,
    
             [[-2.5757e-02,  5.5498e-03, -2.1972e-02],
              [-1.3406e-02, -2.0665e-02, -2.7517e-03],
              [-2.4359e-02,  2.7043e-03,  2.5349e-02]],
    
             [[ 5.1658e-03, -2.9786e-02,  1.2704e-02],
              [-1.8020e-02,  8.5598e-02,  6.6740e-04],
              [-3.1628e-03,  2.3645e-02, -6.4903e-02]],
    
             [[ 3.9627e-03, -5.2094e-03,  1.3886e-02],
              [-3.7860e-02,  1.8379e-02,  6.1846e-02],
              [ 8.3205e-03,  2.6255e-02,  3.3783e-02]]],


​    
            [[[-2.0964e-02,  2.5370e-02, -1.7020e-02],
              [ 1.0891e-02,  8.4425e-03,  1.8108e-02],
              [-2.9098e-02,  8.4492e-03, -1.9419e-02]],
    
             [[ 1.1922e-03, -5.1480e-02,  3.8803e-03],
              [-3.6808e-02, -1.7441e-02, -3.5299e-02],
              [-1.3415e-02,  1.5315e-02,  2.6672e-02]],
    
             [[ 9.0886e-03, -2.7243e-03,  3.2336e-02],
              [-5.1367e-03,  2.2698e-02, -2.7158e-02],
              [ 1.0612e-02,  9.1343e-04,  1.0016e-02]],
    
             ...,
    
             [[ 8.2980e-03,  2.2242e-02,  1.6844e-02],
              [-4.2482e-02, -2.4660e-02,  9.3187e-03],
              [-2.8374e-02,  1.1788e-02, -1.8709e-02]],
    
             [[-5.1907e-02,  4.9372e-02, -4.9451e-02],
              [ 1.4267e-02,  3.3285e-02, -2.5228e-02],
              [-6.8552e-03, -1.6252e-03,  8.3553e-03]],
    
             [[-1.4484e-03,  2.2049e-02,  2.0003e-02],
              [-4.5934e-02, -7.7408e-03,  4.2970e-02],
              [ 2.1615e-02,  2.7941e-02, -1.1171e-02]]],


​    
            [[[ 5.4081e-02,  3.3310e-03, -1.5288e-02],
              [ 2.5752e-02,  5.8497e-03, -1.9194e-02],
              [-7.3654e-03,  3.2929e-02,  3.5923e-02]],
    
             [[-4.3991e-02,  2.6179e-02,  3.3552e-02],
              [-3.3413e-02,  2.3890e-02, -2.4189e-02],
              [ 1.3159e-02, -2.3851e-02,  1.2658e-02]],
    
             [[ 3.1942e-02,  1.4981e-03,  2.0295e-03],
              [ 4.3999e-02,  4.4064e-02,  1.1983e-03],
              [-2.9893e-02,  3.2847e-02,  3.0525e-03]],
    
             ...,
    
             [[ 3.1411e-02,  3.0203e-02, -5.8562e-02],
              [ 3.5739e-02,  2.6084e-02, -6.9919e-02],
              [ 1.5704e-02, -2.5062e-02,  4.1598e-02]],
    
             [[ 2.2958e-03,  3.9184e-02, -1.4012e-02],
              [-2.3085e-02,  5.7279e-02,  2.8511e-02],
              [ 4.4926e-02, -2.0987e-02, -4.3386e-03]],
    
             [[ 5.2276e-02, -1.1884e-03,  1.3984e-02],
              [ 2.3814e-02, -6.6492e-02, -3.9726e-02],
              [-2.3462e-03,  6.1467e-02,  5.5971e-02]]]])), ('layer3.1.bn2.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1.])), ('layer3.1.bn2.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer3.1.bn2.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), ('layer3.1.bn2.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1.])), ('layer3.1.bn2.num_batches_tracked', tensor(0)), ('layer4.0.conv1.weight', tensor([[[[-3.4426e-02, -2.0650e-02, -7.9379e-03],
              [ 1.4907e-02,  9.9145e-03,  1.6605e-02],
              [ 4.1901e-02, -3.8256e-03,  4.6165e-02]],
    
             [[-1.5916e-02, -1.4827e-02, -1.6131e-02],
              [-2.8016e-02,  2.7889e-03, -7.1107e-03],
              [-1.0538e-02, -5.5127e-02,  2.8052e-02]],
    
             [[-1.2081e-02, -2.3949e-02, -2.0703e-02],
              [ 2.6517e-04, -1.4399e-02,  2.0012e-02],
              [-1.8030e-02, -2.1231e-03,  6.6343e-03]],
    
             ...,
    
             [[-1.7246e-02,  2.4135e-02,  4.2051e-03],
              [ 2.6897e-02, -1.6369e-04, -1.1309e-02],
              [-1.5673e-02, -8.3443e-03,  4.1823e-03]],
    
             [[ 4.3984e-02, -1.7815e-02, -1.4942e-02],
              [-5.1513e-02, -8.1108e-04, -1.5165e-02],
              [-3.6811e-02,  5.4820e-03, -2.3470e-02]],
    
             [[ 1.8343e-02,  1.3291e-02,  3.3124e-03],
              [ 1.5544e-02,  8.9084e-03, -2.0378e-04],
              [-2.5889e-02, -1.6304e-03,  1.8099e-02]]],


​    
            [[[-3.4747e-02, -9.6332e-04,  3.2263e-02],
              [-1.2929e-02,  3.1741e-03,  8.1934e-04],
              [ 1.9111e-02, -2.2590e-02, -3.7585e-02]],
    
             [[-1.3702e-02, -4.8879e-02,  9.8150e-03],
              [ 2.5271e-02,  1.1344e-02,  2.6328e-02],
              [-2.1827e-02,  2.6530e-02, -3.6134e-02]],
    
             [[ 6.9475e-03,  8.4351e-03,  1.2942e-02],
              [ 1.0882e-02, -1.3784e-02,  9.6637e-04],
              [ 3.6921e-02, -3.2078e-02,  2.7090e-03]],
    
             ...,
    
             [[-5.9892e-03, -1.2388e-02,  9.6632e-04],
              [ 1.9213e-04, -7.2601e-03, -7.1639e-03],
              [-9.3838e-03, -4.8896e-03, -3.3557e-02]],
    
             [[-4.0732e-02, -1.1723e-02, -2.5068e-02],
              [ 3.5916e-02, -2.2435e-02, -3.1604e-02],
              [ 3.6318e-02,  5.6179e-03, -1.8664e-02]],
    
             [[-2.7697e-02,  2.9838e-02,  7.5057e-03],
              [ 5.4007e-03,  1.8735e-02, -2.6802e-02],
              [-2.9978e-02,  3.4745e-02,  4.1988e-02]]],


​    
            [[[-1.0869e-03,  5.6779e-03,  2.2907e-02],
              [ 1.4753e-02,  2.7519e-02,  4.3633e-03],
              [-8.5497e-04, -2.6687e-02,  1.4616e-04]],
    
             [[-4.4301e-02, -1.5542e-02, -1.8199e-02],
              [-1.7307e-02, -2.4363e-02,  1.7238e-02],
              [ 1.8318e-02, -2.4690e-03, -3.2064e-02]],
    
             [[ 2.7114e-02, -3.1437e-03, -2.7026e-02],
              [ 5.1856e-02, -2.4643e-02,  3.7309e-02],
              [ 2.6793e-02, -1.8849e-02, -1.9014e-02]],
    
             ...,
    
             [[ 6.0458e-03,  2.8492e-02, -1.6163e-03],
              [-1.1290e-02,  1.5984e-02,  2.1983e-02],
              [ 1.5031e-02,  3.2189e-02, -2.7406e-02]],
    
             [[-5.8427e-03, -4.1422e-02, -1.9051e-02],
              [-6.2273e-03,  1.3993e-02, -9.1453e-03],
              [-1.4653e-02,  2.3670e-02,  2.3815e-02]],
    
             [[-2.4436e-03,  1.2416e-02, -2.6396e-02],
              [-2.7468e-02,  9.6938e-03,  8.4615e-04],
              [ 4.1347e-04,  7.8903e-03, -1.8545e-02]]],


​    
            ...,


​    
            [[[-2.0768e-02,  1.2019e-02, -2.8248e-02],
              [ 1.9572e-02,  2.1437e-02,  1.7574e-02],
              [ 1.0294e-02,  2.8026e-02,  2.3316e-02]],
    
             [[-3.8005e-02, -2.4392e-02, -4.8353e-04],
              [-2.5244e-02,  1.8302e-02,  2.2747e-02],
              [-8.1275e-03, -2.3235e-02,  2.9404e-03]],
    
             [[-6.3621e-03,  2.6156e-02,  3.4662e-03],
              [-1.3349e-02,  1.1119e-02,  8.4367e-03],
              [-1.6682e-02, -7.1614e-03,  2.2505e-02]],
    
             ...,
    
             [[ 1.1390e-02,  2.0807e-02, -9.0026e-03],
              [-7.5694e-03, -1.2274e-03,  6.5852e-03],
              [-9.6348e-03,  7.7879e-03, -3.0931e-02]],
    
             [[-1.5925e-02,  5.9851e-04,  2.2808e-03],
              [ 3.4779e-02, -3.0730e-03, -1.7820e-02],
              [ 4.7367e-03,  3.5228e-03,  1.0464e-02]],
    
             [[-1.5196e-02, -1.9263e-02,  3.0209e-02],
              [ 4.8933e-03, -1.1437e-02,  2.8073e-02],
              [-3.4256e-03,  8.6841e-06,  1.0678e-03]]],


​    
            [[[-3.0823e-03,  1.4280e-03, -6.2772e-03],
              [ 9.2778e-04,  5.1217e-03,  9.3867e-03],
              [-1.0111e-02, -2.2206e-02, -2.5241e-03]],
    
             [[-1.1326e-02,  5.2809e-03,  2.0117e-02],
              [-3.3499e-02,  3.8196e-02,  1.9622e-02],
              [ 4.1473e-03, -2.8321e-02, -7.3684e-03]],
    
             [[-3.1827e-03, -2.1147e-02, -7.2015e-03],
              [-1.2156e-02,  3.5554e-03, -1.0348e-02],
              [-1.7716e-02, -3.9093e-02,  1.4604e-02]],
    
             ...,
    
             [[-2.9968e-02,  1.0996e-02,  1.4550e-02],
              [ 3.8767e-02,  2.4672e-02,  1.1089e-03],
              [ 7.6358e-03, -4.3723e-03,  9.2216e-03]],
    
             [[ 3.4968e-02,  2.2976e-03,  7.0608e-03],
              [-1.1412e-02,  4.5701e-03,  1.3914e-03],
              [-3.1444e-02,  4.0768e-02,  6.1653e-03]],
    
             [[ 1.7533e-02, -3.0494e-02,  1.0099e-02],
              [ 3.1681e-02, -1.6135e-02, -2.6885e-02],
              [ 1.1824e-02,  2.4516e-02,  1.0581e-02]]],


​    
            [[[-3.2708e-04,  5.9305e-03, -1.2218e-02],
              [ 1.7925e-02,  3.7259e-02,  1.1810e-02],
              [ 3.8261e-03,  2.1109e-02, -1.2404e-02]],
    
             [[-2.2040e-02, -7.9393e-03,  3.8319e-02],
              [-2.2752e-02, -7.0236e-03, -2.0490e-02],
              [-1.4765e-02,  2.1094e-04, -5.3436e-03]],
    
             [[ 1.1482e-02, -1.5475e-02,  8.2683e-03],
              [-3.4944e-03,  5.6585e-04, -9.3152e-03],
              [ 1.1439e-02,  1.9590e-02,  6.4645e-03]],
    
             ...,
    
             [[ 3.8313e-02,  2.4285e-02, -1.1131e-02],
              [-1.0242e-04,  4.0969e-02, -2.9994e-02],
              [ 7.1082e-03,  2.2107e-02,  6.5453e-03]],
    
             [[-9.8455e-03,  7.9057e-04,  4.8128e-02],
              [ 2.9453e-02, -2.8000e-02, -8.5380e-03],
              [-2.4060e-02, -2.8138e-02, -4.4874e-02]],
    
             [[-3.5345e-02, -1.9844e-02, -9.3805e-04],
              [-1.2879e-02, -1.6884e-02, -1.4061e-02],
              [ 2.4718e-02,  1.6698e-02,  5.1346e-03]]]])), ('layer4.0.bn1.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1.])), ('layer4.0.bn1.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer4.0.bn1.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer4.0.bn1.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1.])), ('layer4.0.bn1.num_batches_tracked', tensor(0)), ('layer4.0.conv2.weight', tensor([[[[ 3.9987e-02,  1.6377e-02,  1.1537e-02],
              [ 1.5429e-04,  3.3687e-03,  2.6769e-02],
              [ 4.5581e-02, -2.2978e-02,  2.2034e-02]],
    
             [[-2.7881e-02,  6.1255e-03, -1.2566e-02],
              [-1.0602e-02,  2.0413e-02,  3.6084e-02],
              [ 3.5311e-02, -1.3656e-02, -1.1612e-02]],
    
             [[ 2.2937e-02,  1.9162e-02, -5.4952e-03],
              [ 1.1285e-02,  1.8485e-02,  1.6047e-02],
              [ 2.0785e-02, -2.7041e-02, -1.5259e-02]],
    
             ...,
    
             [[ 4.8956e-03,  2.0282e-02, -5.3931e-03],
              [ 1.0222e-02,  1.8082e-02, -1.3942e-02],
              [ 2.6756e-02,  1.8785e-02, -3.1818e-02]],
    
             [[ 9.0138e-03, -2.1343e-02,  1.3420e-02],
              [ 4.6611e-02,  2.0769e-02, -1.2019e-02],
              [ 3.8152e-03,  3.2009e-02,  1.7222e-03]],
    
             [[-1.4284e-02, -1.5476e-02,  3.0132e-02],
              [ 1.4093e-03,  5.1971e-03, -2.1814e-02],
              [-6.4795e-03, -2.5254e-03,  6.3007e-05]]],


​    
            [[[ 4.5274e-03,  1.6223e-02,  2.1589e-02],
              [ 2.3792e-02,  4.5018e-02,  3.7072e-02],
              [ 1.1174e-02, -1.0785e-02, -1.5236e-03]],
    
             [[-3.0813e-03, -3.1105e-02, -8.2161e-03],
              [ 1.1115e-02,  2.0689e-02,  8.1091e-03],
              [-3.2117e-02, -1.1782e-02, -1.5681e-02]],
    
             [[ 3.8722e-04,  2.2575e-02, -2.9844e-02],
              [-1.1150e-03, -2.1851e-02, -1.7332e-02],
              [ 9.3539e-03, -2.7910e-02,  7.0806e-03]],
    
             ...,
    
             [[-7.8759e-03, -4.6700e-02,  3.6113e-02],
              [ 2.9170e-02, -1.5087e-02,  1.4162e-02],
              [-4.6622e-02,  5.1858e-03, -1.1309e-02]],
    
             [[ 4.3175e-03, -2.8738e-02, -9.7116e-03],
              [-1.7032e-02, -5.0311e-03, -3.1874e-02],
              [ 4.5853e-03, -1.2478e-02,  2.6720e-02]],
    
             [[-1.6723e-02, -2.2472e-02, -2.0471e-03],
              [ 5.5419e-04,  2.6593e-02,  2.5199e-02],
              [-1.1873e-02, -4.3603e-03,  2.6322e-03]]],


​    
            [[[-1.3595e-02, -2.6590e-02,  2.3901e-02],
              [ 4.7180e-03,  1.3594e-02,  3.0595e-02],
              [ 4.7231e-02, -4.5193e-02, -4.3234e-02]],
    
             [[ 4.5242e-02,  3.8842e-02, -1.0839e-02],
              [ 3.1156e-02, -3.6172e-02, -2.8559e-02],
              [-2.2913e-02,  1.0383e-02, -5.8493e-03]],
    
             [[ 5.0987e-03, -1.1539e-02, -3.0080e-02],
              [-1.4494e-02,  3.7111e-02,  8.3581e-03],
              [-1.6871e-02,  2.6254e-02, -4.2414e-02]],
    
             ...,
    
             [[ 1.3091e-02,  2.3709e-02, -1.7082e-02],
              [-2.3198e-02,  1.1050e-02, -9.1059e-03],
              [-1.9516e-02, -2.6130e-02,  2.1280e-02]],
    
             [[ 1.1603e-02,  1.4547e-02,  3.1231e-02],
              [ 1.0280e-02,  1.3253e-02,  1.0121e-02],
              [-1.6605e-02, -2.2807e-02, -2.4404e-02]],
    
             [[ 6.8024e-03, -1.3941e-02, -1.4979e-02],
              [-1.7372e-02,  1.2247e-02,  9.5539e-04],
              [ 1.1951e-02, -1.0422e-02,  1.7355e-02]]],


​    
            ...,


​    
            [[[ 2.9022e-02,  2.6381e-02, -1.9288e-02],
              [-4.8321e-03,  2.9191e-02,  6.9125e-04],
              [ 1.5155e-02, -3.5044e-03,  1.2359e-04]],
    
             [[ 2.5525e-02, -1.5591e-02,  1.9321e-02],
              [-2.5520e-03,  1.1416e-02,  2.1690e-03],
              [-4.6676e-03, -2.0419e-02,  9.5332e-04]],
    
             [[-2.8312e-02, -9.8543e-04, -2.9239e-02],
              [-6.1804e-05, -3.0979e-02, -7.3537e-03],
              [ 8.8831e-04, -1.3398e-02,  3.4488e-03]],
    
             ...,
    
             [[ 2.4949e-02,  2.8155e-02,  2.3277e-02],
              [ 5.2676e-03, -2.2039e-02,  2.3924e-02],
              [ 3.0675e-02, -2.8452e-03,  1.4936e-02]],
    
             [[ 1.2647e-02,  1.7826e-02,  4.9843e-02],
              [ 2.5734e-02, -2.5533e-02,  9.8916e-03],
              [ 1.9557e-02,  3.8061e-03, -2.9102e-03]],
    
             [[-8.0495e-03, -1.5350e-02, -3.7474e-02],
              [-2.5358e-02, -4.7807e-03, -3.2121e-02],
              [ 1.5920e-02, -1.0264e-02,  1.8845e-02]]],


​    
            [[[-1.4936e-02, -3.2073e-02,  1.4138e-02],
              [ 8.4281e-03, -1.8128e-02, -2.9171e-02],
              [-3.1910e-02, -7.6171e-03, -1.8151e-02]],
    
             [[-2.4884e-03, -1.3325e-02, -4.8013e-02],
              [-4.5844e-03, -5.3668e-03,  2.5114e-02],
              [-8.1003e-03, -1.3105e-02, -2.9609e-02]],
    
             [[ 2.2515e-02,  3.3793e-03,  3.8654e-04],
              [-9.1275e-03,  6.9659e-03, -4.6730e-03],
              [-2.2623e-02,  2.3079e-02, -4.0595e-03]],
    
             ...,
    
             [[ 1.3057e-02,  5.5646e-03, -1.7202e-02],
              [ 7.1149e-03,  5.9121e-03, -1.4524e-02],
              [-1.0922e-02,  4.2458e-02, -1.8767e-03]],
    
             [[ 2.2265e-02, -3.3623e-02, -1.1373e-02],
              [ 1.0413e-02,  5.7937e-03, -3.0215e-03],
              [-1.7297e-02,  2.1106e-03,  3.6974e-02]],
    
             [[-1.6561e-02,  1.1785e-02,  3.7390e-02],
              [ 1.4489e-02,  1.1884e-02,  6.2340e-03],
              [-9.3383e-03,  1.7597e-02,  1.3923e-02]]],


​    
            [[[ 3.0113e-03, -1.0260e-02,  2.6663e-02],
              [ 6.3192e-03, -1.3597e-02, -2.4875e-02],
              [-1.7034e-02, -4.9622e-03, -2.4227e-02]],
    
             [[ 2.8849e-02,  1.7336e-02,  5.2462e-03],
              [-1.4524e-02,  1.4109e-02,  2.0550e-02],
              [ 4.3476e-03,  2.5029e-02, -1.2926e-02]],
    
             [[-3.0400e-02, -2.2211e-02,  2.8638e-02],
              [ 6.5878e-03, -2.2371e-02,  1.7473e-03],
              [-1.4824e-02, -5.1557e-03, -3.2254e-02]],
    
             ...,
    
             [[-2.6154e-02,  1.6258e-02, -1.9958e-02],
              [ 1.1572e-02, -1.5528e-02,  1.4332e-02],
              [-2.0849e-02,  1.9083e-02,  2.4401e-02]],
    
             [[ 1.2214e-02, -6.1253e-03, -2.1794e-02],
              [-6.6523e-03, -1.1222e-02, -4.7902e-04],
              [-3.7102e-02, -9.2181e-03,  1.0906e-02]],
    
             [[ 3.4334e-03,  8.6663e-03, -2.1325e-02],
              [ 5.3908e-02, -9.8666e-04, -2.3395e-03],
              [-6.9018e-04, -3.1469e-02, -1.6748e-02]]]])), ('layer4.0.bn2.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1.])), ('layer4.0.bn2.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer4.0.bn2.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer4.0.bn2.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1.])), ('layer4.0.bn2.num_batches_tracked', tensor(0)), ('layer4.0.downsample.0.weight', tensor([[[[-0.1354]],
    
             [[ 0.0602]],
    
             [[ 0.0289]],
    
             ...,
    
             [[ 0.0753]],
    
             [[ 0.0123]],
    
             [[-0.0188]]],


​    
            [[[ 0.0252]],
    
             [[ 0.0575]],
    
             [[ 0.0006]],
    
             ...,
    
             [[-0.1084]],
    
             [[ 0.0502]],
    
             [[-0.0238]]],


​    
            [[[-0.0509]],
    
             [[-0.0502]],
    
             [[ 0.0187]],
    
             ...,
    
             [[-0.1324]],
    
             [[-0.0758]],
    
             [[ 0.0340]]],


​    
            ...,


​    
            [[[-0.0054]],
    
             [[ 0.0149]],
    
             [[-0.0721]],
    
             ...,
    
             [[-0.0617]],
    
             [[-0.0071]],
    
             [[-0.1242]]],


​    
            [[[-0.0233]],
    
             [[ 0.0655]],
    
             [[-0.0293]],
    
             ...,
    
             [[ 0.0699]],
    
             [[-0.0081]],
    
             [[ 0.1189]]],


​    
            [[[-0.1025]],
    
             [[ 0.0254]],
    
             [[-0.0530]],
    
             ...,
    
             [[-0.0620]],
    
             [[ 0.0768]],
    
             [[-0.0705]]]])), ('layer4.0.downsample.1.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1.])), ('layer4.0.downsample.1.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer4.0.downsample.1.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer4.0.downsample.1.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1.])), ('layer4.0.downsample.1.num_batches_tracked', tensor(0)), ('layer4.1.conv1.weight', tensor([[[[-0.0325,  0.0232,  0.0008],
              [ 0.0086,  0.0065, -0.0044],
              [ 0.0280,  0.0054,  0.0270]],
    
             [[-0.0493,  0.0004,  0.0107],
              [-0.0045,  0.0231,  0.0146],
              [-0.0070, -0.0103,  0.0178]],
    
             [[-0.0034,  0.0108, -0.0015],
              [-0.0314,  0.0683,  0.0057],
              [ 0.0080, -0.0100, -0.0121]],
    
             ...,
    
             [[ 0.0033, -0.0216,  0.0006],
              [-0.0278, -0.0087, -0.0185],
              [ 0.0188,  0.0032, -0.0023]],
    
             [[-0.0338,  0.0129, -0.0048],
              [ 0.0084,  0.0305,  0.0142],
              [-0.0139,  0.0383,  0.0406]],
    
             [[-0.0686, -0.0325,  0.0332],
              [ 0.0020, -0.0054, -0.0112],
              [-0.0077, -0.0010,  0.0078]]],


​    
            [[[-0.0243,  0.0150,  0.0318],
              [ 0.0119, -0.0112,  0.0148],
              [ 0.0383,  0.0199,  0.0068]],
    
             [[ 0.0013, -0.0315, -0.0010],
              [-0.0203,  0.0103,  0.0094],
              [-0.0151, -0.0121, -0.0203]],
    
             [[-0.0341, -0.0234, -0.0026],
              [ 0.0350, -0.0224, -0.0248],
              [ 0.0159, -0.0219, -0.0127]],
    
             ...,
    
             [[ 0.0058,  0.0068, -0.0087],
              [-0.0037,  0.0230,  0.0233],
              [-0.0366, -0.0026, -0.0189]],
    
             [[-0.0141,  0.0222, -0.0415],
              [-0.0459,  0.0135,  0.0050],
              [ 0.0079,  0.0372, -0.0156]],
    
             [[-0.0328, -0.0193, -0.0409],
              [ 0.0119,  0.0201,  0.0032],
              [-0.0222, -0.0082,  0.0064]]],


​    
            [[[ 0.0242, -0.0264,  0.0295],
              [ 0.0264,  0.0230,  0.0105],
              [-0.0207, -0.0118, -0.0314]],
    
             [[-0.0118,  0.0110, -0.0088],
              [ 0.0255,  0.0467, -0.0211],
              [-0.0334,  0.0163, -0.0048]],
    
             [[ 0.0081,  0.0102, -0.0042],
              [-0.0248,  0.0082, -0.0155],
              [-0.0041,  0.0004, -0.0073]],
    
             ...,
    
             [[-0.0181, -0.0056,  0.0169],
              [ 0.0221,  0.0088,  0.0321],
              [ 0.0145, -0.0208,  0.0133]],
    
             [[-0.0042, -0.0308, -0.0267],
              [-0.0023, -0.0079,  0.0153],
              [-0.0004, -0.0119, -0.0183]],
    
             [[-0.0288, -0.0120,  0.0191],
              [-0.0142,  0.0351,  0.0207],
              [ 0.0019, -0.0065, -0.0266]]],


​    
            ...,


​    
            [[[ 0.0199,  0.0022,  0.0199],
              [-0.0171, -0.0496,  0.0146],
              [ 0.0214, -0.0213,  0.0159]],
    
             [[-0.0022,  0.0159,  0.0062],
              [-0.0030, -0.0032, -0.0108],
              [-0.0481, -0.0035, -0.0002]],
    
             [[-0.0147,  0.0247, -0.0213],
              [ 0.0371,  0.0003, -0.0025],
              [-0.0174, -0.0376, -0.0072]],
    
             ...,
    
             [[ 0.0167, -0.0006,  0.0055],
              [ 0.0026, -0.0105,  0.0006],
              [-0.0327, -0.0348,  0.0079]],
    
             [[-0.0455,  0.0031,  0.0042],
              [ 0.0278,  0.0231,  0.0283],
              [ 0.0002,  0.0230, -0.0246]],
    
             [[ 0.0163, -0.0081, -0.0103],
              [ 0.0099,  0.0322,  0.0340],
              [-0.0082, -0.0066, -0.0001]]],


​    
            [[[-0.0228,  0.0106,  0.0129],
              [-0.0187, -0.0123, -0.0146],
              [ 0.0074, -0.0158,  0.0317]],
    
             [[ 0.0152,  0.0024, -0.0054],
              [-0.0139,  0.0057, -0.0325],
              [ 0.0182, -0.0334, -0.0137]],
    
             [[-0.0183, -0.0541,  0.0039],
              [-0.0131, -0.0093,  0.0177],
              [ 0.0458,  0.0213,  0.0033]],
    
             ...,
    
             [[ 0.0030, -0.0052, -0.0268],
              [-0.0207, -0.0085, -0.0107],
              [ 0.0027,  0.0044,  0.0266]],
    
             [[-0.0208,  0.0075, -0.0117],
              [-0.0437, -0.0346, -0.0385],
              [ 0.0183, -0.0017,  0.0119]],
    
             [[-0.0279, -0.0053, -0.0061],
              [-0.0021,  0.0037,  0.0264],
              [-0.0009,  0.0113,  0.0025]]],


​    
            [[[-0.0166,  0.0277, -0.0019],
              [-0.0078,  0.0233,  0.0373],
              [ 0.0110, -0.0101,  0.0001]],
    
             [[ 0.0176,  0.0080,  0.0250],
              [-0.0071,  0.0051,  0.0149],
              [ 0.0005,  0.0050,  0.0129]],
    
             [[-0.0034, -0.0264, -0.0136],
              [-0.0303, -0.0102, -0.0298],
              [-0.0090,  0.0044, -0.0005]],
    
             ...,
    
             [[ 0.0032, -0.0098, -0.0129],
              [ 0.0101,  0.0107, -0.0183],
              [ 0.0147,  0.0256,  0.0420]],
    
             [[-0.0143,  0.0458, -0.0042],
              [ 0.0407, -0.0196, -0.0276],
              [-0.0008,  0.0726, -0.0162]],
    
             [[-0.0229, -0.0024,  0.0218],
              [ 0.0104,  0.0120, -0.0031],
              [-0.0102, -0.0265, -0.0226]]]])), ('layer4.1.bn1.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1.])), ('layer4.1.bn1.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer4.1.bn1.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer4.1.bn1.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1.])), ('layer4.1.bn1.num_batches_tracked', tensor(0)), ('layer4.1.conv2.weight', tensor([[[[-2.3662e-02, -2.3513e-02, -1.5573e-02],
              [-4.7517e-02,  5.9073e-03, -1.3614e-03],
              [ 6.0780e-03, -3.8839e-02, -1.7578e-02]],
    
             [[-2.1779e-02,  4.0308e-02,  6.1145e-03],
              [-5.9949e-03, -5.6621e-03,  1.4169e-02],
              [ 6.7464e-03,  1.3252e-02, -3.0208e-03]],
    
             [[ 4.5111e-02, -2.0537e-02,  1.5476e-02],
              [-5.1928e-03, -2.4852e-03, -2.5094e-02],
              [ 1.7014e-02,  3.2177e-02,  1.0081e-02]],
    
             ...,
    
             [[ 1.6567e-02,  2.2133e-05, -1.3088e-02],
              [-1.8307e-02, -1.1853e-02, -2.9698e-03],
              [-2.6929e-02,  2.2143e-02,  6.0593e-03]],
    
             [[ 3.9457e-02,  3.0668e-02, -1.2497e-02],
              [ 5.0543e-03, -1.1222e-02,  1.6106e-02],
              [ 1.5617e-02,  9.0761e-04, -1.2669e-02]],
    
             [[-9.1282e-03,  3.6710e-02,  4.3944e-03],
              [-8.8752e-03,  1.4686e-02,  1.0446e-02],
              [-1.8295e-02, -2.2993e-02, -2.3620e-02]]],


​    
            [[[ 1.6531e-03,  3.4567e-03, -1.9429e-02],
              [-4.6900e-03,  1.0452e-02, -2.5580e-02],
              [-1.4039e-03,  7.5230e-03, -3.8997e-03]],
    
             [[-1.0146e-02, -1.2023e-03, -6.3622e-03],
              [-5.0349e-03,  1.8959e-02, -2.4189e-02],
              [-1.9850e-02,  1.4023e-02, -1.9445e-02]],
    
             [[ 9.0402e-03, -3.0418e-02,  8.1724e-03],
              [ 3.7860e-02, -1.2944e-02,  9.7460e-03],
              [-1.5397e-02, -2.0535e-02,  2.0626e-02]],
    
             ...,
    
             [[-3.3120e-03, -1.3496e-02,  1.1123e-02],
              [-3.2833e-02, -4.7743e-04, -3.2559e-02],
              [ 4.8121e-03, -1.0991e-02,  1.1493e-04]],
    
             [[-2.0775e-02,  3.9096e-03, -3.9679e-03],
              [ 2.9163e-02,  1.8294e-02, -3.9337e-03],
              [-5.7133e-04,  2.3218e-03, -8.7641e-03]],
    
             [[ 3.8756e-03, -3.3386e-02,  4.3730e-02],
              [-3.0961e-02,  1.6407e-02,  1.0110e-02],
              [-1.6114e-02,  3.1494e-02,  5.9744e-03]]],


​    
            [[[-1.1329e-02,  3.3126e-03, -4.6241e-03],
              [-2.7044e-02,  6.5909e-03,  7.0866e-03],
              [ 5.2363e-02,  1.5017e-02, -1.7317e-02]],
    
             [[-1.3637e-02,  4.2175e-03,  9.4972e-03],
              [ 6.6826e-03, -3.8088e-03,  7.7308e-03],
              [-2.2740e-02, -1.5828e-02, -2.2909e-02]],
    
             [[-4.5026e-03,  4.6851e-03,  4.6103e-02],
              [ 1.2477e-02,  3.2316e-02,  1.1858e-02],
              [-1.6457e-02,  1.1231e-04,  2.8827e-02]],
    
             ...,
    
             [[ 3.1597e-02,  2.6750e-02, -9.7675e-03],
              [ 2.5121e-02,  1.2314e-02, -1.3105e-02],
              [ 1.2193e-02, -5.0444e-02, -3.3051e-02]],
    
             [[-2.4225e-02, -2.7506e-02, -8.8601e-03],
              [ 2.6399e-02,  2.7519e-02, -4.3771e-02],
              [-1.1470e-02, -4.0470e-03,  2.0637e-02]],
    
             [[-6.1738e-03, -3.3873e-02, -1.0252e-02],
              [ 2.0712e-02, -1.3556e-02,  1.4516e-02],
              [ 7.6622e-05,  1.0441e-02, -2.4969e-02]]],


​    
            ...,


​    
            [[[-1.2288e-03, -1.9742e-03, -1.8614e-02],
              [-6.3986e-03, -4.2116e-03,  2.3176e-02],
              [ 1.0561e-02, -1.6420e-02,  3.4507e-02]],
    
             [[-1.9764e-02,  1.0785e-02,  1.6745e-02],
              [-6.5887e-03,  1.5259e-03, -2.6146e-02],
              [ 2.1095e-02, -3.6375e-02, -2.7022e-03]],
    
             [[ 2.9852e-02, -2.4478e-02,  4.6899e-03],
              [ 1.6306e-02, -1.7728e-04, -3.1754e-04],
              [ 5.4857e-03,  3.3921e-02,  2.7303e-03]],
    
             ...,
    
             [[-2.8838e-02, -2.3566e-02, -1.5317e-02],
              [-1.8346e-02, -5.1785e-03,  2.4209e-02],
              [-2.9087e-02, -2.0016e-02,  1.8509e-02]],
    
             [[ 8.3490e-03,  1.7051e-02, -2.7071e-03],
              [-7.8314e-03,  6.2532e-03,  1.2643e-02],
              [-9.9559e-03, -4.6574e-02, -3.2087e-02]],
    
             [[-1.1705e-02,  1.8888e-02,  3.4632e-02],
              [ 5.1837e-02, -3.0410e-02, -1.1340e-02],
              [ 1.4330e-02, -1.1469e-02, -8.6637e-03]]],


​    
            [[[ 3.0953e-03, -2.9400e-02, -5.0638e-03],
              [-1.7371e-02, -5.2132e-03, -1.8046e-02],
              [ 6.5511e-03,  3.0614e-02, -3.1634e-03]],
    
             [[ 3.3873e-02, -1.3166e-02,  1.1041e-02],
              [-3.8080e-02,  2.4966e-02,  4.4413e-02],
              [-7.7000e-03,  2.9861e-02, -2.8791e-02]],
    
             [[ 6.4641e-03,  1.5620e-02,  1.2418e-02],
              [ 1.8234e-02,  9.1384e-03,  1.9138e-02],
              [-6.1269e-02,  6.9481e-03, -1.1410e-02]],
    
             ...,
    
             [[ 1.1664e-02,  4.8882e-03, -1.9306e-03],
              [ 2.6281e-02,  4.4745e-03,  2.5958e-03],
              [-3.7051e-02, -2.4621e-02,  1.3470e-02]],
    
             [[ 1.1898e-02,  3.2344e-02,  2.6440e-02],
              [-1.1959e-02, -4.2995e-02,  2.4245e-02],
              [-1.5575e-02, -1.0455e-02,  1.6821e-02]],
    
             [[ 7.6697e-03,  5.1326e-02,  1.4794e-02],
              [-1.4745e-02, -3.4990e-03, -4.9270e-02],
              [ 1.6760e-02,  3.9897e-03,  1.0378e-02]]],


​    
            [[[ 3.4175e-02,  2.4683e-04,  1.1436e-02],
              [ 1.7215e-02,  1.4469e-02, -1.3782e-02],
              [ 2.6343e-02, -1.7095e-03,  1.0328e-02]],
    
             [[ 1.5810e-02, -6.1180e-03,  2.4916e-03],
              [ 1.6777e-02, -1.3712e-02,  7.2718e-03],
              [ 1.4679e-02, -4.3712e-03,  2.2391e-02]],
    
             [[ 2.1680e-02, -7.1656e-03,  9.3273e-03],
              [ 3.4611e-03, -1.0776e-03, -7.4966e-03],
              [-1.9199e-02, -3.3206e-02,  7.8546e-03]],
    
             ...,
    
             [[-2.0247e-02, -9.2025e-03,  6.8522e-03],
              [ 7.8949e-03,  1.2932e-02, -8.9096e-03],
              [-2.7274e-02,  5.1701e-02,  4.2695e-03]],
    
             [[ 1.3120e-02, -1.8770e-02,  2.2096e-02],
              [ 4.8489e-02, -3.2794e-02, -1.3497e-02],
              [-1.7710e-02,  1.3417e-02,  7.2844e-03]],
    
             [[ 4.3833e-03, -4.3744e-04,  9.6720e-03],
              [ 1.1254e-02,  3.1494e-02,  7.3637e-03],
              [-2.7838e-02, -9.8798e-03, -4.8723e-04]]]])), ('layer4.1.bn2.weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1.])), ('layer4.1.bn2.bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer4.1.bn2.running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.])), ('layer4.1.bn2.running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1.])), ('layer4.1.bn2.num_batches_tracked', tensor(0)), ('fc.weight', tensor([[ 0.0281, -0.0168, -0.0191,  ...,  0.0396,  0.0266, -0.0092],
            [-0.0200, -0.0314, -0.0091,  ..., -0.0405, -0.0208, -0.0178],
            [-0.0145,  0.0427, -0.0139,  ...,  0.0367, -0.0092, -0.0170],
            ...,
            [ 0.0235, -0.0132,  0.0278,  ...,  0.0387,  0.0171,  0.0375],
            [ 0.0348, -0.0358,  0.0306,  ..., -0.0277,  0.0260,  0.0028],
            [-0.0263,  0.0209,  0.0191,  ...,  0.0144, -0.0108,  0.0325]])), ('fc.bias', tensor([-0.0385,  0.0121,  0.0343,  0.0226,  0.0156,  0.0066, -0.0032, -0.0420,
            -0.0194,  0.0035]))])


# 6. Emsemble

when you trained your model for several times, you can emsemble those models to predict test set.
for example, if you trained three models using the same train set, when a new image arrivied, it will be sent to those models and get three predictions, you can design a voting mechanism to improve the performance of your model.

# class work

please use training skills to train a better resnet model on cifar10, you can change hyper parameters and transforms as you wish, please show your final best accuracy in this page.

<p align="center" style="color:#8E0000"><strong>The final outcome is shown as follows.</strong></p>

![](https://i.loli.net/2019/06/06/5cf8b55e1cb0f99317.png)

![](https://i.loli.net/2019/06/06/5cf8b57538c6976935.png)

Note that the accuracy of test set is above 90%, indicating the effective of our network.


```python
# try a better ResNet18
# set hyper parameter 

batch_size = 128 #batch_size is enabled by GTX1080TI from InplusLab
n_epochs = 300
learning_rate = 1e-1
```


```python
# transform

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
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



```python
# optimizer and loss_fn

model = ResNet18()
model.to(device)
# Cross entropy
loss_fn = torch.nn.CrossEntropyLoss()
# l2_norm can be done in SGD
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4) 
```


```python
# scheduler
def fit(train_loader, val_loader, model, loss_fn, optimizer, n_epochs, device):
    
    train_accs = [] # save train accuracy every epoch
    train_losses = [] # save train loss every epoch
    
    test_accs = []
    test_losses = []
    
    scheduler = lr_scheduler.StepLR(optimizer,step_size=100,gamma=0.1)
    
    for epoch in range(n_epochs): # train for n_epochs 
        
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
# 2nd try
train_accs, train_losses = fit(train_loader, test_loader, model, loss_fn, optimizer, n_epochs, device=device)
```

    Epoch: 1/300. Train set: Average loss: 1.7891, Accuracy: 47.5520



![](http://ww3.sinaimg.cn/large/006tNc79gy1g4qzky0bb9j30ar07q3ye.jpg)



![](http://ww4.sinaimg.cn/large/006tNc79gy1g4qzl1u0m4j30b707qt8n.jpg)



![](http://ww3.sinaimg.cn/large/006tNc79gy1g4qzl630ftj30ar07qt8l.jpg)



![](http://ww2.sinaimg.cn/large/006tNc79gy1g4qzl9o6djj30b007qa9z.jpg)


    Epoch: 1/300. Validation set: Average loss: 1.4275, Accuracy: 47.9400
    Epoch: 2/300. Train set: Average loss: 1.2687, Accuracy: 60.6020



![](http://ww3.sinaimg.cn/large/006tNc79gy1g4qzlg6jeoj30ar07qaa2.jpg)



![](http://ww2.sinaimg.cn/large/006tNc79gy1g4qzljmyxcj30au07qdfu.jpg)



![](http://ww1.sinaimg.cn/large/006tNc79gy1g4qzlmrfpcj30ar07q74a.jpg)



![](http://ww4.sinaimg.cn/large/006tNc79gy1g4qzlpnwb1j30b007qt8q.jpg)


   .......华丽的省略号。。。。。。

    Epoch: 14/300. Validation set: Average loss: 0.5742, Accuracy: 81.8100
    Epoch: 15/300. Train set: Average loss: 0.3028, Accuracy: 86.0620



![](http://ww3.sinaimg.cn/large/006tNc79gy1g4qzmdh716j30ar07q3yj.jpg)



![](http://ww1.sinaimg.cn/large/006tNc79gy1g4qzmg9jwnj30au07qaa2.jpg)



![](http://ww4.sinaimg.cn/large/006tNc79gy1g4qzmizu7fj30ar07qjre.jpg)



![](http://ww4.sinaimg.cn/large/006tNc79gy1g4qzmm4cygj30au07q0sr.jpg)


    Epoch: 15/300. Validation set: Average loss: 0.6817, Accuracy: 79.6800
    Epoch: 16/300. Train set: Average loss: 0.2974, Accuracy: 83.8680


 .......华丽的省略号。。。。。。




    Epoch: 102/300. Validation set: Average loss: 0.3181, Accuracy: 90.0300
    Epoch: 103/300. Train set: Average loss: 0.0354, Accuracy: 99.4940



![](http://ww2.sinaimg.cn/large/006tNc79ly1g4qzomacmsj30ax07qdfw.jpg)



![](http://ww4.sinaimg.cn/large/006tNc79ly1g4qzopnxe8j30b007qglm.jpg)



![](http://ww2.sinaimg.cn/large/006tNc79ly1g4qzosrv0nj30ar07qjrg.jpg)


![](http://ww3.sinaimg.cn/large/006tNc79ly1g4qzovs679j30au07qmxa.jpg)


    Epoch: 103/300. Validation set: Average loss: 0.3211, Accuracy: 90.3200
    Epoch: 104/300. Train set: Average loss: 0.0201, Accuracy: 99.7540



![](http://ww4.sinaimg.cn/large/006tNc79ly1g4qzoz97oyj30ax07qdfw.jpg)



![](http://ww3.sinaimg.cn/large/006tNc79ly1g4qzp2mxxnj30b007qglm.jpg)



![](http://ww3.sinaimg.cn/large/006tNc79ly1g4qzp77cr6j30ar07qjrg.jpg)



![](http://ww2.sinaimg.cn/large/006tNc79ly1g4qzpakvanj30au07qq32.jpg)


 .......华丽的省略号。。。。。。
 
 

    Epoch: 298/300. Validation set: Average loss: 0.2966, Accuracy: 91.3400
    Epoch: 299/300. Train set: Average loss: 0.0017, Accuracy: 100.0000



![](http://ww2.sinaimg.cn/large/006tNc79ly1g4qzr5ja2lj30ax07qglm.jpg)



![](http://ww4.sinaimg.cn/large/006tNc79ly1g4qzr89rctj30b007q0sq.jpg)



![](http://ww3.sinaimg.cn/large/006tNc79ly1g4qzrbakjmj30ar07q74b.jpg)



![](http://ww3.sinaimg.cn/large/006tNc79ly1g4qzre706sj30au07q74c.jpg)


    Epoch: 299/300. Validation set: Average loss: 0.2964, Accuracy: 91.3300
    Epoch: 300/300. Train set: Average loss: 0.0017, Accuracy: 100.0000



![](http://ww1.sinaimg.cn/large/006tNc79ly1g4qzrhatf2j30ax07qglm.jpg)



![](http://ww2.sinaimg.cn/large/006tNc79ly1g4qzrkwokaj30b007q0sq.jpg)



![](http://ww4.sinaimg.cn/large/006tNc79ly1g4qzrnlpayj30ar07q3yj.jpg)



![](http://ww4.sinaimg.cn/large/006tNc79ly1g4qzrq848sj30au07qaa4.jpg)


    Epoch: 300/300. Validation set: Average loss: 0.2963, Accuracy: 91.3100



```python
# to save the parameters of a model
torch.save(model.state_dict(), './params/resnet18_91_params.pt')

# to save the model
torch.save(model, './params/resnet18_91.pt')
```

