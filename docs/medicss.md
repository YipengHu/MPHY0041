# Deep Learning for Medical Image Segmentation and Registration
UCL Medical Image Computing Summer School (MedICSS)
11 – 15 July 2022, London, UK


## The tutorials
https://github.com/YipengHu/MPHY0041/tree/main/tutorials/segmentation

### Setup
Follow the instructions in supported [Development Environment](./env.md)

### Choose between TensorFlow and PyTorch
Get started with [TensorFlow](./tensorflow.md)
Get started with [PyTorch](./pytorch.md)

### Things to do in this project
#### Experimental design
Design a strategy for this small-dataset segmentation expeirment.

#### Validation metrics
Consider what metrics can be used to evaluate the performance of the trained network. Implement them. Report, e.g. by printing to the terminal or files, and save the results during and after training.

#### Visualisation
- Plot the loss from the training set against loss from the validation set. 
- Plot the loss and the metrics of interest from the validation set.
- Analyse the results.
- Plot images overlaid with prediction and ground-truth labels.

#### Data augmentation
- Add spatial data augmentation by warping input images using random affine transformations. 
- Consider a reasonable range for the randomly generated affine parameters.
- Observe the change in training and validation performances.

#### The PROMISE12 Challenge
- Go to [the challenge website](https://promise12.grand-challenge.org/)
- Download and process the original data set.
- Submit to the challenge.

[Further reading](./reading.md)

Optional registration tutorial: https://github.com/YipengHu/MPHY0041/tree/main/tutorials/registration