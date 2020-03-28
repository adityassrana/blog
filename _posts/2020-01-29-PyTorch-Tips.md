---
title: "PyTorch Tips and Tricks"
summary: "a little-more-than-introductory guide to help people get comfortable with PyTorch functionalities."
layout: post
branch: master
toc: true
categories: [tutorials]
comments: true
category: blog
---

### PyTorch and its Modules

1. Variables are now deprecated. Tensors can use Autograd directly.

2. The forward function in the NN module defines how to get the output from the NN.
the nn.module() has a __ call function 

````python
model = NN()
model(local_batch)#which calls net.forward(local_batch)
````

3. Input: (N,C,H,W). The model and the convolutional layers expect the input tensor to be in this format, so when feeding an image/images to the model, add a dimension for batching.

4. Converting from img-->numpy representation and feeding  the model gives an error because the input is in ByteTensor format. Only float operations are supported for conv-like operations.

````python
img = img.type('torch.DoubleTensor')
````

### Dataset and DataLoader Shenanigans

1. Create a dictionary: partition['train'] and partition['validation'].
2. Save and Read the paths from textfiles using ````ls /* /* .png > train_path.txt````
3. For getting the number of images use ````ls /* /* .png | wc -l````
4. Order of Transform, image processing like crops and resize should be done on the PIL Image and not the tensor
    - Crop/Resize-->toTensor-->Normalize

5. the transforms.ToTensor() or TF.to_tensor(functional version of the same command) separates the PIL Image into 3 channels (R,G,B), converts it to the range (0,1). You can multiply by 255 to get the range (0,255.

6. Using transforms.Normalize(mean=[_ ,_ ,_ ],std = [_ ,_ ,_ ]) subtracts the mean and divides by the standard deviation. It is **important** to apply the specified mean and std when using a **pre-trained model**. This will normalize the image in the range [-1,1]. To get the original image back use

````python
image = ((image * std) + mean)
````

For example, when using a model trained on ImageNet it is common to apply the transformation

````python
transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
````
For image tensors with values in [0, 1] this transformation will standardize it so that the mean of the data should be ~0 and the std ~1. This is also known as a standard score or z-score in the literature and usually helps in training.

7. Data Augmentation happens at the step below. At this point \_\_getitem\_\_ method in the Dataset Class is called, and the transformations are applied.

````python
for data in train_loader():
````
8. torchvision.transforms vs torchvision.transforms.functional.

The functional API is statelessand you can directly pass all the necessary arguments.

Whereas torchvision.transforms are classes, initialized with some default parameters unless specified. 

````python
# Class-based. Define once and use multiple times
transform = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
data = transform(data)

# Functional. Pass parameters each time
data = TF.normalize(data, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
````

9. The functional API is very useful when transforming your data and target with the same random values, e.g. random cropping:

````python
i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(512, 512))
image = TF.crop(image, i, j, h, w)
mask = TF.crop(mask, i, j, h, w)

````
Functional API also allows us to perform identical transform on both image and target
````python
def transform(self, image, mask):
    # Resize
    resize = transforms.Resize(size=(520, 520))
    image = resize(image)
    mask = resize(mask

# Random horizontal flipping
if random.random() > 0.5:
    image = TF.hflip(image)
    mask = TF.hflip(mask)

# Random vertical flipping
if random.random() > 0.5:
    image = TF.vflip(image)
    mask = TF.vflip(mask)

````
10. Example Dataset Class:

````python
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision
import torchvision.transforms.functional as TF #it's not tensorflow
from torchvision import transforms

class Image_Train_Dataset(Dataset): #inherit from Dataset class and overrride the methods __len__ and __getitem__
    def __init__(self,image_paths):
        self.list_id = open(image_paths_list,'r').read().splitlines()
        
    def __len__(self):
        #return size of the Dataset
        return len(self.list_id)

    def transform(self,image):
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(256,256))#allows us to apply the same crop on semantic segmentation if it's used
        image = TF.crop(image, i, j, h, w)
        image = TF.resize(image,size=(128,128))
        image = TF.to_tensor(image)
        return image

    def __getitem__(self, index):
        #generates one sample of data
        image = Image.open(self.list_id[index])
        if image.mode == 'L':
            image = image.convert('RGB')
        image= self.transform(image)
        return image

    def load_img_data(self,index):#for making inference easier
        image = Image.open(self.list_id[index])
        return image

    def load_tensor_data(self,index):
        image = Image.open(self.list_id[index])
        image = self.transform(image)
        return image
````

### Writing your own custom Autograd Functions

1. [PyTorch Examples for Reference Github](https://github.com/pytorch/pytorch/blob/53fe804322640653d2dddaed394838b868ce9a26/torch/autograd/_functions/pointwise.py)

2. [PyTorch official docs](https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html)

3. Gradient returned by the class should have the same shape as the input to the class, to be able to update the input in the optimizer.step() function.

4. Avoid using in-place operations as they cause problems while back-propagation because the way they modify the graph. As a precaution, always clone the input in the forward pass, and clone the incoming gradients before modifying them.

An in-place operation directly modifies the content of a given Tensor without making a copy. Inplace operations in PyTorch are always postfixed with a _, like .add_() or .scatter_(). Python operations like \+ = or \*= are also inplace operations.


````python
grad_input = grad_output.clone()
return grad_input
````

Example

````python
class MyReLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        input = i.clone()
        """ ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

````


Dealing with non-differentiable functions:

w_hard : non-differentiable
w_soft : differentiable proxy for w_hard

````python
w_bar = w_soft + tf.stop_grad(w_hard - w_soft) #in tensorflow
w_bar = w_soft + (w_hard - w_soft).detach()  #in PyTorch
````

    It gets you x_forward in the forward pass, but derivative acts as if you had x_backward
    ````
    y = x_backward + (x_forward - x_backward).detach()
    ````

loss.backward() computes d(loss)/d(w) for every parameter which has requires_grad=True. They are accumulated in w.grad. And the optimizer.step() updates w using w.grad, w += -lr* x.grad

### Saving and Loading Models

Python saves models as a state_dict. You may use either of the two ways

````python
torch.save(model.state_dict(),'final-contours-branch{}.pt'.format(args.expname))
torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'loss':train_loss},'resume_training.tar')
````

On Loading a model, if it shows a message like this, it means there were no missing keys.

````
IncompatibleKeys(missing_keys=[], unexpected_keys=[])
````
Use this when you have added new layers to the architecture which were not present in the model you saved as checkpoint

````python
trained_dict = torch.load('checkpoint.pt')
model = reducio_binarizer.Reducio()
model.load_state_dict(trained_dict, strict=False)
model.to(device)

````

Keyboard interrupt and saving the last state of a model:

````python
try:
    # training code here
except KeyboardInterrupt:
    # save model here
````

### Learning Rate Schedulers

Change LR with increasing epochs. Read [Reduce LR on Plateau](https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html)

### Useful Links
1. [Using _ in Variable Naming](https://dbader.org/blog/meaning-of-underscores-in-python)
2. [Pytorch Coding Conventions](https://discuss.pytorch.org/t/pytorch-coding-conventions/42548)
3. [Fine Tuning etc](https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/)