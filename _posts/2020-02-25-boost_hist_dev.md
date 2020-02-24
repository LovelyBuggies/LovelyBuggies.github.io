---
layout:     post
title:     Boost Histogram for Python (Developer Mode)
subtitle:   
date:       2020-02-28
author:     Nino Lau
header-img: img/jp-desvigne-wVRqisV_mPk-unsplash.jpg
catalog: true
tags:
    - 知识介绍
    - Boost Histogram

---

Once you've mastered boost-hist, it's time to consider becoming a developer to contribute to [Scikit-HEP](https://scikit-hep.org/)! It's a creative and challenging job. So let's see how to build developing environment and become a developer!

![](https://tva1.sinaimg.cn/large/0082zybply1gc7sxqmduaj31it0u0h48.jpg)

### Building from Source 

The first thing we need to do is to clone the source code from the official git.

```
git clone --recursive https://github.com/scikit-hep/boost-histogram.git
cd boost-histogram
```

### Developer Environment

Then let's build our developer environment!

In this part, we need to establish a virtual environment to develop in a good manner. This is how you would set one up with Python 3:

1. Create a python 3 environment named `.env` and activate it. 

   ```
   python3 -m venv .env
   source ./.env/bin/activate
   ```

2. Install modules needed from PyPI with pip3. 

   ```
   pip3 install numpy ipykernel pytest-sugar numba matplotlib
   ```

3. Create a kernel named `boost-hist`. 

   ```
   python -m ipykernel install --user --name boost-hist
   ```

4. Specify pip extra requirement -  VCS projects can be installed in [editable mode](https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs) (using the [--editable](https://pip.pypa.io/en/stable/reference/pip_install/#install-editable) option) or not. You can install local projects or VCS projects in “editable” mode:

   ```
   pip3 install -e .[test]
   ```

5. Deactivate `.env`. 

   ```
   deactivate
   ```

Now, you can run notebooks using your system jupyter lab, and it will list the environment as available! See your kernel according to `jupyter kernelspec list`. 

