---
layout:     post
title:     Boost Histogram for Python - A Quick Start
subtitle:   
date:       2020-02-24
author:     Nino Lau
header-img: img/tomasz-smal-vT_JAucWU00-unsplash.jpg
catalog: true
tags:
    - 知识介绍
    - Boost Histogram

---



If you are a Pythoner, you must have used [Matplotlib](https://matplotlib.org/). [Matplotlib](https://matplotlib.org/) is a Python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments across platforms. It is a toolkit build in Python scripts, so you will meet some problems when you are trying to access the C++-based plotting toolkits. Some projects are aiming to provide Python blinding for plotting toolkits. [Boost-histogram](https://github.com/scikit-hep/boost-histogram) is one of them and it provides Python bindings for the C++14 Boost::Histogram library. Let's get a quick start for this tool and see its amazing effects!

### Install

You can install this library from [PyPI](https://pypi.org/project/boost-histogram/) with pip (here I used pip3 installation and run it on my local Jupyter Notebook):

```bash
python -m pip install boost-histogram
```

or you can use Conda through [conda-forge](https://github.com/conda-forge/boost-histogram-feedstock):

```bash
conda install -c conda-forge boost-histogram
```

All the normal best-practices for Python apply; you should be in a virtual environment, etc.

### Run 

Open your Jupyter Notebook:


```bash
mkdir hist-plot
Jupyter Notebook
```

Then you are able to run and test your code in Jupyer environment!

### Simple 1D Histogram

First, let's draw a 1-D Histogram to warm up!

```python
import numpy as np
import matplotlib.pyplot as plt

# Make a 1D histogram
plt.hist(np.random.normal(size=1_000_000), np.arange(-3, 4, .5))
plt.savefig("simple_1d.png")
```

![](https://tva1.sinaimg.cn/large/0082zybply1gc7abss57kj30c0080748.jpg)

### Simple 2D Histogram 

In this part, I am going to show you how to draw a simple 2D histogram (note that this is not often used in [Matplotlib](https://matplotlib.org/) without the help of [Seaborn](http://seaborn.pydata.org/)). Before we deep into the source code, we need to figure out WHAT IS A 2D HISTOGRAM. This is pretty important for it will help you to understand the basic structure of [boost-histogram](https://github.com/scikit-hep/boost-histogram). 

In a boost-histogram, a histogram is a collection of Axis objects and storage.

![](https://tva1.sinaimg.cn/large/0082zybply1gc6tc19q4aj30ff09qjs3.jpg)

A 2D-histogram has two independent variables (x and y), and one dependent variable. Just like building our city, we neet two independent variables (latitude and longitude ) to figure out the location, and we can build a skyscraper (altitude) in one place.

![](https://tva1.sinaimg.cn/large/0082zybply1gc6thjclaij30hs0b4my8.jpg)

Here, I show a modification of the sample code - [simple_2d.py](https://github.com/LovelyBuggies/boost-histogram/blob/develop/examples/simple_2d.py). 

```python
import boost_histogram as bh
import matplotlib.pyplot as plt
import numpy as np

# Create 2d-histogram with two axes with 20 equidistant bins from -3 to 3
h = bh.Histogram(
    bh.axis.Regular(50, -3, 3, metadata="x"), bh.axis.Regular(50, -3, 3, metadata="y")
)

# Generate some Numpy arrays with data to fill into histogram,
# in this case normal distributed random numbers in x and y
x = np.random.randn(1_000_000)
y = np.random.randn(1_000_000)

# Fill histogram with Numpy arrays, this is very fast
h.fill(x, y)

# Get numpy.histogram compatible representation of the histogram
w, x, y = h.to_numpy()

# Draw the count matrix

fig, ax = plt.subplots()
mesh = ax.pcolormesh(x, y, w.T)
ax.set_xlabel(h.axes[0].metadata)
ax.set_ylabel(h.axes[1].metadata)
fig.colorbar(mesh)
plt.show()
plt.savefig("simple_2d.png")
```

![](https://tva1.sinaimg.cn/large/0082zybply1gc6tyrq940j30c0080mx9.jpg)

Let me get this straight:

1. First, we need to create a `Histogram` object.
2. Then create x and y axes using two np-arrays.
3. Fill the `Histogram` object.
4. Get the x and y scales and weights. Note that this x is NOT the same as the x in 2. (so is y). The difference is the x in 2. is the observation-x; while, this x is the x-scale.
5. Create a subplot.
6. Plot color on the mesh. Note that we need to use TRANSPOSE of weights to get the mesh.
7. *Optional: create a color bar*.
8. Show and/or save the figure.

Follow the steps above and get your simple 2D-histogram now!

### Density Histogram

Let try a density histogram in this part and I used my modification of the [simple_density.py](https://github.com/LovelyBuggies/boost-histogram/blob/develop/examples/simple_density.py). 

```python
import numpy as np
import boost_histogram as bh
import matplotlib.pyplot as plt

# Make a 2D histogram
hist = bh.Histogram(bh.axis.Regular(50, -3, 3), bh.axis.Regular(50, -3, 3))

# Fill with Gaussian random values
hist.fill(np.random.normal(size=1_000_000), np.random.normal(size=1_000_000))

# Compute the areas of each bin
areas = np.prod(hist.axes.widths, axis=0)

# Compute the density
density = hist.view() / hist.sum() / areas

# Get the edges
X, Y = hist.axes.edges

# Make the plot
fig, ax = plt.subplots()
mesh = ax.pcolormesh(X.T, Y.T, density.T)
fig.colorbar(mesh)
plt.savefig("simple_density.png")
```

Then you can get a figure named `simple_density.png`, which is saved in your current folder.

![](https://tva1.sinaimg.cn/large/0082zybply1gc6tyrmzfrj30hs0dc0t1.jpg)

Dense historgram is similar to the simple 2D-histogram. This only difference is we neet to calculate the density of each bin addtionally according to  `density = hist.view() / hist.sum() / areas`.

### Save and Load Histogram

We save our histogram in figure format above. So you must be curious about how to save and load histogram objects?

Here I am going to show how to manipulate bins and use `pickle` to dump and load our histograms. *P.S., make sure you have installed pickle package before running this code.*

```python
import boost_histogram as bh
import pickle
from pathlib import Path

h1 = bh.Histogram(bh.axis.Regular(2, -1, 1))
h2 = h1.copy()

h1.fill(-0.5)
h2.fill(0.5)

# Arithmetic operators
h3 = h1 + h2
h4 = h3 * 2

print(f"{h4[0]}, {h4[1]}")

h4_saved = Path("h4_saved.pkl")

# Now save the histogram
with h4_saved.open("wb") as f:
    pickle.dump(h4, f, protocol=-1)

# And load
with h4_saved.open("rb") as f:
    h5 = pickle.load(f)

assert h4 == h5
print("Succeeded in pickling a histogram!")

# Delete the file to keep things tidy
h4_saved.unlink()
```

```
2.0, 2.0
Succeeded in pickling a histogram!
```

### Accumulator Storage & Flow

In the final part of our exploration, I will introduce the powerful accumulator storage of boost-hist. Boost-hist can not only store basic types like integer, float, and double, it can also store weights and means. I use the modification of sample code [simple_log_weight.py](https://github.com/scikit-hep/boost-histogram/blob/develop/examples/simple_log_weight.py) to show its usage.

```python
import boost_histogram as bh

# make 1-d histogram with 5 logarithmic bins from 1e0 to 1e5
h = bh.Histogram(
    bh.axis.Regular(5, 1e0, 1e5, metadata="x", transform=bh.axis.transform.log),
    storage=bh.storage.Weight(),
)

# fill histogram with numbers
x = (3e0, 3e1, 3e2, 3e3, 3e4)
h.fill(x, weight=4)
h.fill(x, weight=4)
h.fill(x, weight=4)
h.fill(x, weight=4)

# iterate over bins and access bin counter
for idx, (lower, upper) in enumerate(h.axes[0]):
    val = h[idx]
    print(
        "bin {0} [{1:g}, {2:g}): {3} +/- {4}".format(
            idx, lower, upper, val.value, val.variance ** 0.5
        )
    )

# under- and overflow bin
lo, up = h.axes[0][bh.underflow]
print(
    "underflow [{0:g}, {1:g}): {2} +/- {3}".format(
        lo, up, h[bh.underflow].value, h[bh.overflow].variance ** 0.5
    )
)
lo, up = h.axes[0][bh.overflow]
print(
    "overflow  [{0:g}, {1:g}): {2} +/- {3}".format(
        lo, up, h[bh.overflow].value, h[bh.overflow].variance ** 0.5
    )
)
```

```
bin 0 [1, 10): 16.0 +/- 8.0
bin 1 [10, 100): 16.0 +/- 8.0
bin 2 [100, 1000): 16.0 +/- 8.0
bin 3 [1000, 10000): 16.0 +/- 8.0
bin 4 [10000, 100000): 16.0 +/- 8.0
underflow [0, 1): 0.0 +/- 0.0
overflow  [100000, inf): 0.0 +/- 0.0
```

1. First, create a histogram object.
2. Then fill the histogram four times weighted 4.
3. Print each bin.
4. Create overflow and underflow.

Accumulator storages hold more than one number internally. They return a smart view when queried with `.view()`. Here we use the WeightedMean accumulator storage, something confusing here is the weights. The value of a bin is the fill-in value, which can increase with each fill. The variance of a bin is the power of its weight(s). Common misconceptions include:
1. Understanding the variance as the variance of each fill.
2. Understanding the variance as the variance within the range of a bin.

### Q&A

#### Q1: What is the importance of designing boost-Hist package?

<details><summary>A1 (click to expand)</summary>

Since Matplotlib has almost established a uniform drawing specification for Python,  boost-histogram does not provide plotting utilities partially. It’s not that hard to take a boost-histogram and plot it with Matplotlib. But it should be made simpler and easy, and that’s one of the things Hist should do. Also, some of the plots (like pull plots) take quite a few lines of code in Python currently.

The "multidimensional histogram + easy indexing" can redefine histogramming, this is also something we can do that even ROOT (which is a massive dependency with many downsides) cannot do, and it is almost impossible in any other Python tool (save for Physt, but that is quite slow).

</details>

#### Q2: Isn't it better to call boost-hist 2D-histogram a "heatmap"?

<details><summary>A2 (click to expand)</summary>

Yes, a 2D histogram plot is like a heat map or mesh grid in Matplotlib. You can probably see a variety of plots (and other things) that HEP physicists are used to in the ROOT users' guide: https://root.cern.ch/root/htmldoc/guides/users-guide/Histograms.html.

</details>

#### Q3: What's the superiority of boost-hist concerning performance?

<details><summary>A3 (click to expand)</summary>

The superiority of boost-hist (Py) will be shown majorly by comparing with benchmarks, just like boost-hist (C++). We are now expanding the boost-histogram benchmarks. We will make new benchmarks soon.

</details>

### Contribution

- **PR**: [#307](https://github.com/scikit-hep/boost-histogram/pull/307)
- **Issue**: [#306](https://github.com/scikit-hep/boost-histogram/issues/306), [#308](https://github.com/scikit-hep/boost-histogram/issues/308)

### Conclusion

In this article, I show some simple examples of powerful boost-hist, *i.e.*, simple 2-D histogram, 2-D density histogram, histogram save and load, and accumulator storage. In the next article, I am going to show some advanced manipulation of boost-hist. 

"Practice makes perfect!" Try it by yourself!



![](https://tva1.sinaimg.cn/bmiddle/0082zybply1gc7gmnmtxij30md096t9u.jpg)