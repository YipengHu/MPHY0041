
# Formative Assessment
The formative assessment is a list of tasks that can be implemented for each tutorial. Design your experiments and discuss the results.

## Validation
### Validation set
Change the data loader to include an additional validation set, such that the validation set will be evaluated during training.

### Validation metrics
Consider what metrics can be used to evaluate the performance of the trained network. Implement them. Report (by printing to the terminal or files) and save the results during and after training, respectively.

### Visualisation
- Plot the loss from the training set against loss from the validation set. 
- Plot the loss and the metrics of interest from the validation set.
- Analyse the results.


## Network architecture
### Convolutional neural networks
Adapt the convolutional kernels to dilated kernels (dilations parameter in TensorFlow and PyTorch), observe the changes in training and validation performances.

### Residual networks
Compare the performance with and without using residual networks.

### Padding
Understand different types of padding and their impact. Test your understanding by running networks using the changed padding methods.


## Training and regularisation
### Parameter norms and batch norm
- Change the weight of L2-norm of the network trainable parameters.
- Add or remove the batch normalisation layer.
- Observe the change in training and validation performances.

### Optimiser
Change the optimiser to _stochastic gradient descent_ and compare the training performance.

### Depth and width
Change the network size and depth, and observe their impact to the training and validation performances.

### Data augmentation
- Add spatial data augmentation by warping input images using random affine transformations. 
- Consider a reasonable range for the randomly generated affine parameters.
- Observe the change in training and validation performances.

### Dropout
Add dropout layers to a) all layers, b) all convolutional layers and c) only the last output layer. Observe the differences between these strategies.
