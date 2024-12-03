# A Roadmap to Learning Python and Machine Learning by Yourself

_All materials in this Roadmap are freely available online._

## Content
1. Python  
    1.1. [Development environment](#development-environment)  
    1.2. [Basic language tutorials](#basic-language-tutorials)   
    1.3. [Basic language reference](#basic-language-reference)   
    1.4. [Numerical computing packages](#numerical-computing-packages)  
    1.5. [Hands-on experience](#hands-on-experience)  
    1.6. [Search for help](#search-for-help)  

2. Machine learning with Python  
    1.1. [Machine learning books](#machine-learning-books)  
    1.2. [The scikit-learn package](#the-scikit-learn-package)  
    1.3. [Deep learning](#deep-learning)  
    1.4. [TensorFlow or PyTorch](#tensorflow-or-pytorch)  
    1.5. [Example neural networks](#example-neural-networks)  


________
## Part 1 - Python

------------------
### <a name="development-environment"></a>1.1 Development environment  

> Learn basic command line tools.  

You may never need command line tools by using a good IDE (see below), but it is useful to learn some commands for many reasons. And, you can develop Python programmes entirely in command line.  
* On Linux, e.g. [Ubuntu](https://ubuntu.com/tutorials/command-line-for-beginners#1-overview)
* On MacOS, use the app [Terminal](https://support.apple.com/en-gb/guide/terminal/welcome/mac)
* On ChromeOS, use enable [Linux](https://support.google.com/chromebook/answer/9145439), then follow the Linux tutorial above.
* On Windows, there are many options, include 
  * [Windows Subsystem Linux](https://docs.microsoft.com/en-us/windows/wsl/), for a transferable Linux/Unix experience, or 
  * [Command Prompt or PowerShell](https://docs.microsoft.com/en-us/windows-server/administration/windows-commands/windows-commands), after installing a [Python app in the Windows App Store](https://www.microsoft.com/en-gb/search/shop/apps?q=Python)

> Set up your own (integrated) development environment. Cross-platform options include the following - browse a simple tutorial for your choice.  
* [Visual Studio Code](https://code.visualstudio.com/docs/languages/python)
* [PyCharm](https://www.jetbrains.com/pycharm/)
* A code editors and command line  

_Optional:_ Consider to use a virtual environment to manage your multiple projects, e.g. using the [Conda environment](https://docs.conda.io/en/latest/). However, you can skip this now and come back when you grow familiar with Python.  


------------------
### <a name="basic-language-tutorials"></a>1.2 Basic language tutorials
> Complete one basic Python language tutorial.  

Understand the basic syntax, flow control, data structure, modules, IO and classes. Options include:
* [Documentation on the python.org](https://docs.python.org/3/tutorial/)
* [On w3school.com](https://www.w3schools.com/python/)
* [On tutorialspoint.com](https://www.tutorialspoint.com/python/index.htm)

------------------
### <a name="basic-language-reference"></a>1.3 Basic language reference
> Keep this.  
* [Python Language Reference](https://docs.python.org/3/reference/)


------------------
### <a name="numerical-computing-packages"></a>1.4 Numerical computing packages
> Install the following packages and complete the tutorial for each. Keep the references handy.

Virtual environments or cloud services, such as Anaconda and Colab, may have these numerical computing packages pre-installed.  

* NumPy: <[Install](https://numpy.org/install/)> <[Tutorial](https://numpy.org/devdocs/user/quickstart.html)> <[Reference](https://numpy.org/devdocs/reference/index.html#reference)> <[Tutorial for MATLAB Users](https://numpy.org/doc/stable/user/numpy-for-matlab-users.html)>
* SciPy: <[Install](https://www.scipy.org/install.html)> <[Tutorial](https://docs.scipy.org/doc/scipy/reference/tutorial/index.html)> <[Reference](https://docs.scipy.org/doc/scipy/reference/api.html)>
* Matplotlib: <[Install](https://matplotlib.org/users/installing.html)> <[Tutorial](https://matplotlib.org/tutorials/)> <[Reference](https://matplotlib.org/3.3.2/api/index.html)>

------------------
### <a name="hands-on-experience"></a>1.5 Hands-on experience
> Do something you want.

There are many open-source projects, some examples below. Run the code and change the code to do something different. 
* [Tutorials in a UCL Module MPHY0030](https://github.com/YipengHu/MPHY0030/tree/main/tutorials)
* [Examples on geeksforgeeks.org](https://www.geeksforgeeks.org/python-programming-examples/)
* Example code in tutorials, e.g. [those on w3school.com](https://www.w3schools.com/python/python_examples.asp) 


------------------
### <a name="search-for-help"></a>1.6 Search for help
> Search, search and search for more help!
* Google
* [StackOverflow](https://stackoverflow.com/)



________
## Part 2 - Machine learning with Python  

------------------
### <a name="machine-learning-books"></a>2.1 Machine learning books
> Read the books.  

* [Pattern Recognition and Machine Learning](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf) by Chritopher Bishop
* [Deep Learning](https://www.deeplearningbook.org/) by Ian Goodfellow et al., with [Lecture slides](https://www.deeplearningbook.org/lecture_slides.html)
* [Element of Statistical Learning](https://web.stanford.edu/~hastie/Papers/ESLII.pdf) by Trevor Hastie et al.

------------------
### <a name="the-scikit-learn-package"></a>2.2 The scikit-learn package
> Use scikit-learn as not only a Python library, but also a reference for learning machine learning. Get started with the basic topics. 

* [Install scikit-learn](https://scikit-learn.org/stable/install.html)
* [Supervised learning - Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
  * Ordinary least squares
  * Ridge regression and classification
  * Bayesian regression
  * Logistic regression
* [Clustering](https://scikit-learn.org/stable/modules/clustering.html#clustering)
  * K-means
  * Gaussian Mixtures
* [Dimension reduction](https://scikit-learn.org/stable/modules/decomposition.html#decompositions)
  * Principal component analysis (PCA)
  * Factor Analysis
* [Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing)  


------------------
### <a name="deep-learning"></a>2.3 Deep learning  
> Reading blogs and watching YouTube lectures and clips help...

* Read books, such as Goodfellow's book (see above).
* Build your own neural networks (see below).
* Read research papers.

------------------
### <a name="tensorflow-or-pytorch"></a>2.4 TensorFlow or PyTorch
> Either or both.  
* TensorFlow: <[Install](https://www.tensorflow.org/install)> <[Tutorials](https://www.tensorflow.org/tutorials)> <[Reference](https://www.tensorflow.org/api_docs/python/tf)>  
* PyTorch: <[Install](https://pytorch.org/get-started/locally/)> <[Tutorials](https://pytorch.org/tutorials/)> <[Reference](https://pytorch.org/docs/stable/index.html)>  


------------------
### <a name="example-neural-networks"></a>2.5 Example neural networks
> Start coding, if you have not by now.

* Tutorial examples in TensorFlow/PyTorch tutorials.
* Examples in open-source projects, e.g. in medical imaging:
    * [MONAI](https://monai.io/start.html)
    * [DeepReg](https://deepreg.readthedocs.io/en/latest/) and its [Demos](https://deepreg.readthedocs.io/en/latest/demo/introduction.html)
