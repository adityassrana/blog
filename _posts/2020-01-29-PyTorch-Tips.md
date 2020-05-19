---
title: "PyTorch Tips and Tricks"
description: "a little-more-than-introductory guide to help people get comfortable with PyTorch functionalities."
layout: post
toc: true
categories: [tutorials]
comments: true
---
### PyTorch

Input Tensor Format : (N,C,H,W). The model and the convolutional layers expect the input tensor to be of this shape, so when feeding an image/images to the model, add a dimension for batching.

Converting from img-->numpy representation and feeding the model gives an error because the input is in ByteTensor format. Only float operations are supported for conv-like operations.

````py
img = img.type('torch.DoubleTensor')
````

### Dataset and Transforms

- Dataset Class : what data will be input to the model and what augmentations will be applied
- DataLoader Class : how big a minibatch will be, 

 To create our own dataset class in PyTorch we inherit from the Dataset Class and define two main methods, the ``__len__`` and the ``__getitem__``

 ````py
 import torch
 from PIL import Image
 import torchvision
 import torchvision.transforms.functional as TF #it's not tensorflow

 class ImageDataset(torch.utils.data.Dataset):
     """Dataset class for creating data pipeline for images"""

     
     def __init__(self, train_glob, patchsize):
     """"
     train_glob is a Glob pattern identifying training data. 
     This pattern must expand to a list of RGB images
     in PNG format. for eg. "/images/cat/*.png"
     
     patchsize is the crop size you want from the image
     """
       self.list_id = glob.glob(train_glob)
       self.patchsize = patchsize
         
     def __len__(self):
       #denotes total number of samples
       return len(self.list_id)
    
     def __getitem__(self, index):
         #generates one sample of data
         image = Image.open(self.list_id[index])
         # convert to RGB if image is B/W
         if image.mode == 'L':
             image = image.convert('RGB')
         image= self.transform(image)
         return image
    
     def transform(self,image):
         # Fucntional transforms allow us to apply  
         # the same crop on semantic segmentation    
         i, j, h, w = torchvision.transforms.RandomCrop.get_params(image ,
                      output_size = (self.patchsize, self.patchsize))
         image = TF.crop(image, i, j, h, w)
         image = TF.to_tensor(image)
         return image

 ````

Image processing operations like cropping and resizing should be done on the PIL Image and not the tensor
    
    Image --> Crop/Resize --> toTensor --> Normalize

The transforms.ToTensor() or TF.to_tensor (functional version of the same command) separates the PIL Image into 3 channels (R,G,B), converts it to the range (0,1). You can multiply by 255 to get the range (0,255).

Using transforms.Normalize( mean=[_ ,_ ,_ ],std = [_ ,_ ,_ ] ) normalizes the input by  subtracting the mean and dividing by the standard deviation, the output is in the range [-1,1]. It is **important** to apply the specified mean and std when using a **pre-trained model**.  To get the original image back use

````py
image = ((image * std) + mean)
````

For example, when using a model trained on ImageNet it is common to apply this transformation. It normalizes the data to have a mean of ~0 and std of  ~1

````py
transforms.Normalize(mean = [0.485, 0.456, 0.406],
                     std = [0.229, 0.224, 0.225])
````

torchvision.transforms vs torchvision.transforms.functional.

The functional API is stateless and you can directly pass all the necessary arguments. Whereas torchvision.transforms are classes initialized with some default parameters unless specified. 

 ````python
 # Class-based. Define once and use multiple times
 transform = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
 data = transform(data)

 # Functional. Pass parameters each time
 data = TF.normalize(data, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
 ````

The functional API is very useful when transforming your data and target with the same random values, e.g. random cropping:

````python
i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(512, 512))
image = TF.crop(image, i, j, h, w)
mask = TF.crop(mask, i, j, h, w)
````

Functional API also allows us to perform identical transforms on both image and target
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
 
Data Augmentation happens at the step below. At this point, ````__getitem__```` method in the Dataset Class is called, and the transformations are applied.

````python
for data in train_loader():
````

### Writing  Custom Autograd Functions

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
        In the backward pass we receive a Tensor containing the gradient of the loss wrt the output, and we need to compute the gradient of the loss wrt the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

````

[PyTorch Examples for Reference Github](https://github.com/pytorch/pytorch/blob/53fe804322640653d2dddaed394838b868ce9a26/torch/autograd/_functions/pointwise.py)

[PyTorch official docs](https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html)

Gradient returned by the class should have the same shape as the input to the class, to be able to update the input in the optimizer.step() function.

Avoid using in-place operations as they cause problems while back-propagation because of the way they modify the graph. As a precaution, always clone the input in the forward pass, and clone the incoming gradients before modifying them.

An in-place operation directly modifies the content of a given Tensor without making a copy. Inplace operations in PyTorch are always postfixed with a _, like .add_() or .scatter_(). Python operations like \+ = or \*= are also in-place operations.


````python
grad_input = grad_output.clone()
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

PyTorch saves models as a state_dict.

````py
torch.save({  'encoder_state_dict': encoder.state_dict(),
          'decoder_state_dict': decoder.state_dict()
          },os.path.join(args.experiment_dir,"latest_checkpoint.tar"))
````

Use keyword  ````strict````  when you have added new layers to the architecture which were not present in the model you saved as checkpoint

````python
encoder = Encoder()
checkpoint = torch.load('checkpoints/clic.tar')
encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
````

On Loading a model, if it shows a message like this, it means there were no missing keys (it's not an error).

````
IncompatibleKeys(missing_keys=[], unexpected_keys=[])
````

Keyboard interrupt and saving the last state of a model:

````python
try:
    # training code here
except KeyboardInterrupt:
    # save model here
````

### Extra Readings

- [Grokking PyTorch](https://github.com/Kaixhin/grokking-pytorch/blob/master/README.md)
- [Effective PyTorch](https://github.com/vahidk/EffectivePyTorch/blob/master/README.md)
- [The Python Magic Behind PyTorch](https://amitness.com/2020/03/python-magic-behind-pytorch)
- [Python is Cool - ChipHuyen](https://github.com/chiphuyen/python-is-cool/blob/master/README.md)
- [PyTorch StyleGuide](https://github.com/IgorSusmelj/pytorch-styleguide/blob/master/README.md)
- [Clean Code Python](https://github.com/zedr/clean-code-python)
- [Using _ in Variable Naming](https://dbader.org/blog/meaning-of-underscores-in-python)
- [Pytorch Coding Conventions](https://discuss.pytorch.org/t/pytorch-coding-conventions/42548)
- [Fine Tuning etc](https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/)


### More Tutorials
- https://github.com/dsgiitr/d2l-pytorch
- https://github.com/L1aoXingyu/pytorch-beginner
- https://github.com/yunjey/pytorch-tutorial
- https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/README.md


