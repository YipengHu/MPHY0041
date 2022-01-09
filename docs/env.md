
# Development Environment
The [technical support](https://weisslab.cs.ucl.ac.uk/YipengHu/mphy0030/-/blob/main/docs/dev_env_python.md) is available with conda.  

## Conda
- [Install Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).
- Create a module environment `mphy0041`.
```bash
conda create --name mphy0041 -c conda-forge numpy matplotlib tensorflow=2.6 pytorch=1.10
```

## Optional - Install TensorFlow or PyTorch in `mphy0041` with GPU support
>**Note:** the following instructions for installation depend on the available CUDA versions.

For TensorFlow users, see [conda TensorFlow guide](https://docs.anaconda.com/anaconda/user-guide/tasks/tensorflow/), e.g.:
```bash
conda install tensorflow-gpu -c anaconda 
```

For PyTorch users, see [PyTorch install guide](https://pytorch.org/get-started/locally/), e.g.:
```bash
conda install pytorch cudatoolkit=10.1 -c pytorch
```


## Docker 
A [docker image](https://hub.docker.com/repository/docker/yipenghu/ucl-module-ubuntu) is provided with conda and git installed on Ubuntu 20.04. Alternatively, a [Dockerfile](../Dockerfile) is also provided. See more details and support in the [official Docker documentation](https://docs.docker.com/).


## Cloud service
TBC


## Cheat - Use TensorFlow and PyTorch on Google Colab
[Google Colab](https://colab.research.google.com/)