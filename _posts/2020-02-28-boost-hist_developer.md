---
layout:     post
title:     Boost Histogram for Python - Developing
subtitle:   
date:       2020-02-27
author:     Nino Lau
header-img: img/isaac-quesada-MspsNfWf3q8-unsplash.jpg
catalog: true
tags:
    - 知识介绍
    - Hist

---

Once you've mastered boost-hist, it's time to consider becoming a developer to contribute to [Scikit-HEP](https://scikit-hep.org/)! It's a creative and challenging job. So let's see how to build a developing environment and become a developer!

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

To rebuild, you may need to delete the `/build` directory and rerun `pip3 install -e .` from the environment.

```
rm -r ./build
pip3 install -e .
```

**You can also build your developer environment using CMake**. I tried it and found it not convenient as pip3.

### Testing

Now we can test our project using `pytest`.

Just re-activate our environment and run `pytest` is ok. 

*P.S. As a developer, I think you should also have `pytest` installed in your pip root sake of convenience. Of course, this is NOT a good developing manner, but sometimes convenience and manner is a tradeoff.*

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

pip3 install pygal pygaljs

pytest-benchmark compare 0001 0002 --sort fullname --histogram
```

Let's see what's shown by adding a benchmark to our command.

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

#### Q1: CMake was unable to find the program 'Ninja'.

<details><summary>A1 (click to expand)</summary>
<br>When building my environment, I met an error <code>CMake Error: CMake was unable to find a build program corresponding to 'Ninja'</code>. I browsed <a href="https://stackoverflow.com/questions/38658014/ninja-not-found-by-cmake">some solutions</a>, but only to find that nothing changed. 
<ul>
  <li>I guessed maybe <code>ninja</code> is not installed in local path <code>usr/bin</code> and tried to move <code>ninja</code> binary to it. But system did not allow for that operation.</li> 
 <li>I also tried to symlink "ninja-build" to "ninja" according to 
   <code>ln -s /usr/bin/ninja /usr/bin/ninja-build</code> OR <code>ln -s /usr/local/bin/ninja /usr/local/bin/ninja-build</code>, but the error still existed.</li>
</ul>
If your situation is same as mine and you cannot solve this error by using the ways mention above, I recommend you to use the powerful <code>pytest</code> for your unit test.
</details>
#### Q2: Boost-hist already exist in … when rebuilding.

<details><summary>A2 (click to expand)</summary>
<br>When trying to rebuild in a different director by <code>pip3 install -e .</code>, we will meet the problem: <code>boost-hist already exist</code>.
The simplest solution is to move your project folder to the right place. You can see the existed installation in the last line of error thrown.
</details>

#### Q3: Pytest lacks 'pybind11_tests'.

<details><summary>A3 (click to expand)</summary>
<br>When we are testing using <code>python3 -m pytest</code>, a normal error is <code>ModuleNotFoundError: No module named 'pybind11_tests'</code>. I reported this bug and proposed an issue <a href="https://github.com/scikit-hep/boost-histogram/issues/312">#312</a>, Henry gave me the solution timely:
<blockquote>I guess I habitually run <code>python3 -m pytest tests</code>, which forces the <code>tests</code> dir to be the only place searched. Without that, we aren't limiting the search locations, so it picks up <code>extern/pybind11/tests</code>, which it (obviously) should not pick up. For now, you can add the <code>tests</code> part to your command, and we can add a pytest configuration option to disable searching for tests in <code>extern</code>. I can help add that soon. </blockquote>
</details>


#### Q4: No modules named 'pytest' found.

<details><summary>A4 (click to expand)</summary>
  <br>If you are puzzled by this issue, you might pip install <code>pytest</code> in your virtual environment. Of course, pip root cannot find your <code>pytest</code> after your deactivation. Note that you are still in original folder if you activate your virtual environment (<code>ls -a</code> can see), so just run <code>python3 -m pytest tests</code> is ok, instead of <code>python3 - m pytest ../tests</code>, else you will still meet lacks "pybind11_tests" problem, i.e. Q3.
</details>


### Contribution

- **Ideas**: [#311](https://github.com/scikit-hep/boost-histogram/issues/311).
- **Report Bugs**: [#312](https://github.com/scikit-hep/boost-histogram/issues/312) (fixed by Henry [#313](https://github.com/scikit-hep/boost-histogram/pull/313)).
- **Fix Bugs**: [#315](https://github.com/scikit-hep/boost-histogram/pull/315) (merged), [#317](https://github.com/scikit-hep/boost-histogram/pull/317) (merged).

### Conclusion

Testing is a slow process, keep patient and enjoy your time as a developer!



![](https://tva1.sinaimg.cn/bmiddle/0082zybply1gc7gmnmtxij30md096t9u.jpg)


