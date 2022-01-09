
# Supported Development Environment

After you [install Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/), create a module environment:
```bash
conda create --name mphy0041 numpy scipy matplotlib h5py tensorflow pytorch
```
Then activate the environment:
```bash
conda activate mphy0041
```
To return to conda base environment:
```bash
conda deactivate
```

### Optional - Install TensorFlow or PyTorch in `mphy0041` with GPU support
>**Note:** the following instructions for installation depend on the available CUDA versions.

For TensorFlow users, see [conda TensorFlow guide](https://docs.anaconda.com/anaconda/user-guide/tasks/tensorflow/), e.g.:
```bash
conda install tensorflow-gpu -c anaconda 
```

For PyTorch users, see [PyTorch install guide](https://pytorch.org/get-started/locally/):
```bash
conda install pytorch cudatoolkit=10.1 -c pytorch
```

### Cheat - Use TensorFlow and PyTorch on Google Colab
[Google Colab](https://colab.research.google.com/)