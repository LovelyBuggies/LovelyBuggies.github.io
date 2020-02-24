---
layout:     post
title:      Boost-hist-Py (elementary)
subtitle:   
date:       2020-02-24
author:     Nino Lau
header-img: img/Snip20190312_61.png
catalog: true
tags:
    - 知识介绍
		- Hist Plotting

---

If you are a Pythoner, you have must used [Matplotlib](https://matplotlib.org/). [Matplotlib](https://matplotlib.org/) is a Python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments across platforms. It is a toolkit build in Python scripts, so you will meet some problems when you are trying to access the C++-based plotting toolkits. Some projects are aiming at to provide Python blinding for plotting toolkits. [Boost-histogram](https://github.com/scikit-hep/boost-histogram) is one of them and it provides Python bindings for the C++14 Boost::Histogram library. Let's get a quickstart for this tool and see its amazing effects!

### Install

You can install this library from [PyPI](https://pypi.org/project/boost-histogram/) with pip (here I used pip3 installation and run it on my local Jupyter Notebook):

```
python -m pip install boost-histogram
```

or you can use Conda through [conda-forge](https://github.com/conda-forge/boost-histogram-feedstock):

```
conda install -c conda-forge boost-histogram
```

All the normal best-practices for Python apply; you should be in a virtual environment, etc.

### Run 

Open your Jupyter Notebook:


```
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

In this part, I am going to show you how to draw a simple 2D histogram (note that this is not offen used in [Matplotlib](https://matplotlib.org/) without help of [Seaborn](http://seaborn.pydata.org/)). Before we deep into the source code, we need to figure out WHAT IS A 2D HISTOGRAM. This is pretty important for it well help you to understand the basic structure of [boost-histogram](https://github.com/scikit-hep/boost-histogram). 

In boost-histogram, a histogram is collection of Axis objects and a storage.

![](https://tva1.sinaimg.cn/large/0082zybply1gc6tc19q4aj30ff09qjs3.jpg)

A 2D-histogram has two independent variables (x and y), and one dependent variable. Just like building our city, we neet two independent variables (latitude and longtitude ) to figure out the location, and we can bulid a skyscraper (altitude) at one place.

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

Let me to get this straight:

1. First, we need to create a `Histogram` object.
2. Then create x and y axes using two np-arrays.
3. Fill the `Histogram` object.
4. Get the x and y scales and weights. Note that this x is NOT same as the x in 2. (so is y). The difference is the x in 2. is the observation-x; while, this x is the x-scale.
5. Create a subplot.
6. Plot color on mesh. Note that we need to use TRANSPOSE of weights to get the mesh.
7. *Optional: create a color bar*.
8. Show and/or save the figure.

Follow the steps above and get your own simple 2D-histogram now!

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

We save our histogram in figure format above. So you must be curious about how to save and load histogram object?

Here I am going to show how to manipulate bins and use pickle to dump and load our histograms. *P.S., make sure you have install pickle package before running this code.*

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

In the final part of our exploration, I will introduce the powerful accumulator storage of boost-hist. Boost-hist can not only store basic type like integer, float, and double, it can also store weights and means. I use the modification of sample code [simple_log_weight.py](https://github.com/scikit-hep/boost-histogram/blob/develop/examples/simple_log_weight.py) to show its usage.

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

#### What is the motivation of design boost-Hist package?

According to boost-hist C++:

> C++ lacks a widely-used, free multi-dimensional histogram class. While it is easy to write a one-dimensional histogram, writing a general multi-dimensional histogram poses more of a challenge. If a few more features required by scientific professionals are added onto the wish-list, then the implementation becomes non-trivial and a well-tested library solution desirable.

Boost-hist Python is a binding for boost-hist C++. In my opinion, they were created mainly for three reasons:

1. C++ has limited number of histogram plotting tools.
2. It is not easy for C++ to plot high-dimentional histograms.
3. We need some performance enhancement for previous tools.

But what is the necessity or importance of this tool?

1. It's true that C++ lacks plotting tools. But Matplotlib have almost established a uniform drawing specification for Python, meaning that this urgent job for C++ may not be an emergency for Python.
2. After thinking about the meaning of 2D-histogram, I find it represent data of 2 independent variables and 1 dependent variable. Other tools can cover this requirements, too.
   - We could use Seaborn to plot a heatmap to represent this kind of data. (Actually, I think it's might be more accurate to call boost-hist 2D-histogram a "heatmap" ).
   - For data of higher-dimension, if we need to show it, we might use matplotlib-3D. Specifically, we could use scatter plot with scatters in different color to show data of 3 independent variables and 1 dependent variable. (However, I don't find ways to represent this kind of data by only using boost-hist without MPL-3D.)
3. Yeah, I know that histogram plotting is not just a matter of drawing but storaging as well, and I know that boost-hist has a powerful Storage function. But I don't know what is the superiority compared with others. *I see that C++ boost-hist have a Doc section named benchmarks. Maybe we can also add this section to our Doc to attract potential users?*

### Contribution

- **PR**: [#307](https://github.com/scikit-hep/boost-histogram/pull/307)
- **Issue**: [#306](https://github.com/scikit-hep/boost-histogram/issues/306), [#308](https://github.com/scikit-hep/boost-histogram/issues/308)

### Conclusion

In this article, I show some simple examples of powerful boost-hist, *i.e.*, simple 2-D histogram, 2-D density histogram, histogram save and load, and accumulator storage. In the next article, I am going to show some advanced manipulation of boost-hist. 

"Practice makes perfect!" Try it by yourself!

![](https://tva1.sinaimg.cn/large/0082zybply1gc7gmnmtxij30md096t9u.jpg)