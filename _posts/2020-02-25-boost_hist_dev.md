---
layout:     post
title:     Boost Histogram for Python - The Developer Mode
subtitle:   
date:       2020-02-28
author:     Nino Lau
header-img: img/isaac-quesada-MspsNfWf3q8-unsplash.jpg
catalog: true
tags:
    - 知识介绍
    - Boost Histogram

---

Once you've mastered boost-hist, it's time to consider becoming a developer to contribute to [Scikit-HEP](https://scikit-hep.org/)! It's a creative and challenging job. So let's see how to build developing environment and become a developer!

### Building from Source 

The first thing we need to do is to clone the source code from the official git.

```
git clone --recursive https://github.com/scikit-hep/boost-histogram.git
cd boost-histogram
```

### Pip Environment

Then let's build our developer environment!

In this part, we need to establish a virtual environment to develop in a good manner. This is how you would set one up with Python 3:

1. Create a python 3 environment named `.env` and activate it. 

   ```bash
   python3 -m venv .env
   source .env/bin/activate
   ```

2. Install modules needed from PyPI with pip3. 

   ```bash 
   pip3 install numpy ipykernel pytest-sugar numba matplotlib
   ```

3. Create a kernel named `boost-hist`. 

   ```bash
   python -m ipykernel install --user --name boost-hist
   ```

4. Specify pip extra requirement -  VCS projects can be installed in [editable mode](https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs) (using the [--editable](https://pip.pypa.io/en/stable/reference/pip_install/#install-editable) option) or not. You can install local projects or VCS projects in “editable” mode:

   ```bash
   pip3 install -e .[test]
   ```

5. Deactivate `.env`. 

   ```bash
   deactivate
   ```

Now, you can run notebooks using your system [Jupyter Lab](https://jupyter.org/) ([the next generation Jupyter Notebook](https://www.sohu.com/a/341219920_487512)), and it will list the environment as available (*view your kernel according to `jupyter kernelspec list`*)!

```bash
jupyter lab
```

![](https://tva1.sinaimg.cn/large/0082zybply1gc7vtlubk2j31nb0u07ev.jpg)

To rebuild, you may need to delete the `/build` directory, and rerun `pip3 install -e .` from the environment.

```
rm -r ./build
pip3 install -e .
```

**You can also build your developer environment using CMake**. I tried it and found it not convinient as pip3.

### Testing

Now we can test our project using `pytest`.





### Q&A

#### CMake Error

When using CMake to build my environment, I met an error: 

```
CMake Error: CMake was unable to find a build program corresponding to "Ninja".
```

I browsed [some solutions](https://stackoverflow.com/questions/38658014/ninja-not-found-by-cmake) but only to find that nothing changed. 

- I guessed maybe `ninja` is not installed in local `usr/bin` and tried to move binary `ninja` to it. But system did not allow for that operation. 
- I also tried to symlink "ninja-build" to "ninja" according to `# ln -s /usr/bin/ninja /usr/bin/ninja-build` OR `# ln -s /usr/local/bin/ninja /usr/local/bin/ninja-build`. But the error still existed.

Considering I don't have to build two test enviroment, I continued my exploration using pip.

#### Rebuild Error

When trying to rebuild in a different director by `pip3 install -e .`, we will meet some problem `boost-hist already exist.` 

- The simplest solution is to change your project folder. You can see the existed installation in the last line of error thrown.

#### Pytest Lacks Module 

When we are testing using `python3 -m pytest`, a normal error is `ModuleNotFoundError: No module named 'pybind11_tests'`. Why this happens? How to deal with it?



