
# Development Environment
The [technical support](https://weisslab.cs.ucl.ac.uk/YipengHu/mphy0030/-/blob/main/docs/dev_env_python.md) is available with conda.  

## Conda
- [Install Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).
- Create a module environment `mphy0041`.
```bash
conda create --name mphy0041 -c conda-forge numpy matplotlib tensorflow=2.10 pytorch=1.12
```

## Jupyter notebooks
Notebook can be installed within the [`mphy0041` conda environment](../docs/conda.md):

``` bash
conda activate mphy0041 \
 && conda install -c conda-forge notebook
```

Alternatively, create a standalone conda environment instead to manage a lighter environment without the deep learning packages.



## Optional - Install TensorFlow or PyTorch in `mphy0041` with GPU support
>**Note:** the following instructions for installation depend on the available CUDA versions.

For TensorFlow users, see [conda TensorFlow guide](https://docs.anaconda.com/anaconda/user-guide/tasks/tensorflow/), e.g.:
```bash
conda install tensorflow-gpu -c anaconda 
```

For PyTorch users, see [PyTorch install guide](https://pytorch.org/get-started/locally/), e.g.:
```bash
conda install pytorch cudatoolkit=10.2 -c pytorch
```


## Docker 
Ubuntu 20.04 [Docker images](https://hub.docker.com/repository/docker/yipenghu/ucl-module-ubuntu) are available. Alternatively, a [Dockerfile](../Dockerfile) is also provided. See more details and support in the [official Docker documentation](https://docs.docker.com/).  

Download the pre-built Docker image:
```bash
sudo docker pull yipenghu/ucl-module-ubuntu:minimal 
```
Create and run a container named `mphy0041` in its interactive bash:
```bash
sudo docker run --name mphy0041 -ti yipenghu/ucl-module-ubuntu:minimal bash
```
Within the container, follow the above to clone this repositry and create the conda environment.


## Cloud service
_TBC_


## Cheat - Use TensorFlow and PyTorch on Google Colab
[Google Colab](https://colab.research.google.com/)
