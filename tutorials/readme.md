# Deep Learning Tutorials
The tutorials require a few dependencies, numpy, matplotlib, in addition to one of the two deep learning libraries. Individual tutorials may also require other libraries which will be specified in the readme.md in individual tutorial folders (see links below). Conda is recommended to manage the required dependencies. 

It is not mandatory, in tutorials or assessed coursework, to use any specific development, package or environment management tools. However, [technical support]((https://weisslab.cs.ucl.ac.uk/YipengHu/mphy0030/-/blob/main/docs/dev_env_python.md)) is available for the setups detailed in [Development Environment](../docs/env.md). 

## Deep learning libraries
Module tutorials are implemented in both [TensorFlow](https://www.tensorflow.org/) and [PyTorch](https://pytorch.org/). 

Learning materials for TensorFlow for Medical Imaging are recommended in [Learning TensorFlow for Medical Imaging](../docs/tensorflow.md).

Learning materials for PyTorch for Medical Imaging are recommended in [Learning PyTorch for Medical Imaging](../docs/pytorch.md).

## Get started
To run the tutorial examples, follow the instruction below. 
For the first time only, create a [conda environment `mphy0041`](../docs/env.md).  
>Note: some tutorials may need additional tools, please read readme.md in individual subfolder.

Next, activate the created `mphy0041` environment:
``` bash
conda activate mphy0041
```

Then, `cd` to each individual tutorial subfolder as the working directory, e.g.:
``` bash
cd tutorials/classification  # e.g. `cd segmentation`
```

Usually, run the `data.py` script first to download tutorial data: 
``` bash
python data.py
```

Run one of the training scripts:
``` bash
python tf_train.py  # train using TensorFlow 2
```
or 
``` bash
python pt_train.py  # train using PyTorch
```

Visualise example data and (predicted) labels:
``` bash
python visualise.py
```

## Tutorials

### Image classification
[Anatomical structure classification on 2D ultrasound images](./classification)

### Image segmentation
[Segmentation of organs on 3D MR images](./segmentation)

### Image registration*
[Unsupervised registration of CT image slices](./registration)

### Image synthesis*
[Ultrasound image simulation](./synthesis)


## Formative assessment
A list of tasks are detailed in the [Formative Assessment](../docs/formative.md). Complete them for individual tutorials.
