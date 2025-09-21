
# Development Environment
The following development tools are supported.

## Operating systems
A Linux/Unix-based operating system: 
- Linux distros: e.g. [Ubuntu](https://ubuntu.com/tutorials/install-ubuntu-desktop), [Debian](https://www.debian.org/releases/stable/installmanual)
- [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install) 
- [ChromeOS with Linux turned on](https://support.google.com/chromebook/answer/9145439?)  
- [MacOS Terminal](https://support.apple.com/guide/terminal)


## Micromamba
- [Install Micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html).
- Create a module PyTorch environment `mphy0041-pt`.
```bash
micromamba create --name mphy0041-pt -c pytorch numpy matplotlib requests pytorch
```
- Create a module TensorFlow environment `mphy0041-tf`.
```bash
micromamba create --name mphy0041-tf -c pytorch numpy matplotlib requests pytorch
```


## Microsoft Visual Studio Code
- [Install VSCode](https://code.visualstudio.com/docs)
- [Step-by-step tutorial](https://code.visualstudio.com/docs/python/python-tutorial)


## Git and GitHub
 _Optional_: Basics use of Git with GitHub repositories: e.g. [Work with Git](https://github.com/YipengHu/MPHY0030/blob/main/docs/dev_env_git.md)
 