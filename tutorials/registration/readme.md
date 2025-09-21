# Registration

This tutorial implements unsupervised registration networks for 2D head-and-neck CT images.  

<img src="../../docs/media/registration.jpg" alt="alt text"/>


Use the module [development environments](../../docs/env.md) to run the code, with the tutorial folder as the current directory. The tutorial also uses code adapted from [Tensorflow Examples](https://github.com/tensorflow/examples)


## TensorFlow
```bash
micromamba activate mphy0041-tf 
pip install git+https://github.com/tensorflow/examples.git  # install tensorflow-examples
python data.py  # download data
python tf_train.py
python visualise.py  # save plotted results  
```

## PyTorch
```bash
micromamba activate mphy0041-pt  
python data.py  # download data
python pt_train.py
python visualise.py  # save plotted results  
```


## Other materials
A series tutorials for image registration can be found in the [Learn2Reg tutorial](https://github.com/learn2reg/tutorials2019). 

For 3D medical image registration using deep learning, see [DeepReg](http://deepreg.net) and [MMONAI tutorials](https://github.com/Project-MONAI/tutorials/tree/main/3d_registration). 
