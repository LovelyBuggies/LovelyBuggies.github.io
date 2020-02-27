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
   python3 -m ipykernel install --user --name boost-hist
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

Just re-activate our environment and run `pytest` is ok. 

*P.S. As a developer, I think you should also have `pytest\*` installed in your pip root sake of convinience. Of course, this is NOT a good developing manner, but sometimes convinience and manner is a tradeoff.*

```bash
python3 -m pytest tests
```

When you see the green 100% progress bar, the testing is completed. 

![](https://tva1.sinaimg.cn/large/0082zybply1gcav8co7chj31gy0u0tjh.jpg)

If you want to benchmark before and after a change, you can use the following commands:

```bash
python3 -m pytest tests --benchmark-enable --benchmark-autosave

# Make some changes

python3 -m pytest tests --benchmark-enable --benchmark-autosave

pip3 pygal pygaljs

pytest-benchmark compare 0001 0002 --sort fullname --histogram
```

Let's see what's shown by adding benchmark to our command.

![](https://tva1.sinaimg.cn/large/0082zybply1gcawhrmoruj322l0u04qp.jpg)

For each `pytest`, there are several tests in benchmarks. The best performance concerning each criteria for a unit test is shown in green, while the worst is in red. To compare the testings, we can run pytest-benchmark again after some modifications (*here I made no changes to the original tests*). 

Then, we can compare the two testings to see the influence of the modifications. (*Note, while the histogram option (`--histogram`) is nice, it does require `pygal` and `pygaljs` to be installed. Feel free to leave it off if not needed.*) Except for the testing display, the benchmarks will show the comparison results by exporting svg figures.

![](https://tva1.sinaimg.cn/large/0082zybply1gcax4nv2ihj314j0u0qt5.jpg)

![](https://tva1.sinaimg.cn/large/0082zybply1gcax4nk2kbj314w0u0ayn.jpg)

![](https://tva1.sinaimg.cn/large/0082zybply1gcax4n5v36j31430u01kx.jpg)

![](https://tva1.sinaimg.cn/large/0082zybply1gcax4mrtw9j313w0u049c.jpg)

![](https://tva1.sinaimg.cn/large/0082zybply1gcax4mim5ej31470u04cl.jpg)

![](https://tva1.sinaimg.cn/large/0082zybply1gcax4m4ffpj314a0u049t.jpg)

### Q&A

#### Q1: CMake was unable to find program "Ninja".

<details><summary>A1 (click to expand)</summary>

When building my environment, I met an error: 

```
CMake Error: CMake was unable to find a build program corresponding to "Ninja".
```

I browsed [some solutions](https://stackoverflow.com/questions/38658014/ninja-not-found-by-cmake) but only to find that nothing changed. 

- I guessed maybe `ninja` is not installed in local `usr/bin` and tried to move binary `ninja` to it. But system did not allow for that operation. 
- I also tried to symlink "ninja-build" to "ninja" according to `# ln -s /usr/bin/ninja /usr/bin/ninja-build` OR `# ln -s /usr/local/bin/ninja /usr/local/bin/ninja-build`. But the error still existed.

If your situation is same as mine and you cannot solve this error by using the ways mention above, I recommend you to use the powerful `pytest` for your unit test.

</details>

#### Q2: Boost-hist already exist in … when rebuilding.

<details><summary>A2 (click to expand)</summary>

When trying to rebuild in a different director by `pip3 install -e .`, we will meet some problem `boost-hist already exist.` 

The simplest solution is to move your project folder to the right place. You can see the existed installation in the last line of error thrown.

</details>

#### Q3: Pytest lacks "pybind11_tests".

<details><summary>A3 (click to expand)</summary>

When we are testing using `python3 -m pytest`, a normal error is `ModuleNotFoundError: No module named 'pybind11_tests'`. I reported this bug and proposed an issue [#312](https://github.com/scikit-hep/boost-histogram/issues/312), Henry gave me the solution timely:

> I guess I habitually run `python3 -m pytest tests`, which forces the `tests` dir to be the only place searched. Without that, we aren't limiting the search locations, so it picks up `extern/pybind11/tests`, which it (obviously) should not pick up.
>
> For now, you can add the `tests` part to your command, and we can add a pytest configuration option to disable searching for tests in `/extern`. I can help add that soon.

</details>

#### Q4: No modules named "pytest" found.

<details><summary>A4 (click to expand)</summary>
If you are puzzled by this issue, you might pip install `pytest` in your virtual environment. Of course, pip root cannot find your `pytest` after your deactivation. Note that you are still in original folder if you activate your virtual environment (`ls -a` can see), so just run `python3 -m pytest tests` is ok, instead of `python3 - m pytest ../tests`, else you will still meet lacks "pybind11_tests" problem, i.e. Q3.

</details>

### Contribution

- **Ideas**: [#311](https://github.com/scikit-hep/boost-histogram/issues/311).
- **Report Bugs**: [#312](https://github.com/scikit-hep/boost-histogram/issues/312) (fixed by Henry [#313](https://github.com/scikit-hep/boost-histogram/pull/313)).
- **Fix Bugs**: [#315](https://github.com/scikit-hep/boost-histogram/pull/315) (merged).

### Conclusion

Testing is a slow process, keep patient and enjoy your time as a developer!



![](https://tva1.sinaimg.cn/bmiddle/0082zybply1gc7gmnmtxij30md096t9u.jpg)

