---
layout:     post
title:      Python3 Type Check
subtitle:   
date:       2020-05-23
author:     Nino Lau
header-img: img/clay-banks-tRXPl_FCK9c-unsplash.jpg
catalog: true
tags:
    - 个人经历
---

## Python 类型检查

### BG

前几天，Henry 安排我调研一下 Python 3 的 typing 和 annotations，他说这是很有用的 feature，尤其是对软件开发的工作而言。我突然发现 LeetCode 平台的 Solution 其实就用了 typing check，于是在这里给大家分享一下这方面的知识。

传统上，Python 解释器以灵活但隐式的方式处理类型。Python 最新的几个版本允许您指定明确的类型进行提示，有些工具可以使用这些提示来帮助您更有效地开发代码。那么最新的几个版本都做了什么呢？Python 3.0 开始支持函数标注，3.5、3.6 开始支持变量标注。从 3.7 开始，`from __future__ import annotation`  已经可以添加 Python 4 风格的纯文本标注了，相信未来 Python4 强大的类型检查功能一定能成为这门语言的巨大优势。

这篇文章主要涉及以下几个方面：

- 类型标注和类型注释是什么？
- 如何为代码添加静态类型检查？
- 怎样运行静态类型检查工具？
- 如何在运行时强制类型转换？

### 动态和静态类型检查

Python 是一个动态类型检查的语言，以灵活但隐式的方式处理类型。Python 解释器仅仅在运行时检查类型是否正确，并且允许在运行时改变变量类型。

```python
>>> if False:
...     1 + "two"  # This line never runs, so no TypeError is raised
... else:
...     1 + 2
...
3

>>> 1 + "two"  # Now this is type checked, and a TypeError is raised
TypeError: unsupported operand type(s) for +: 'int' and 'str'
```

Python 最新的几个版本允许您指定明确的类型进行提示，有些工具可以使用这些提示来帮助您更有效地开发代码。

### 类型提示

既然 Python 是动态类型检查的语言，不可避免有很多隐式的类型转换，如何确保它们按计划运行呢？Python 的类型检查系统主要是用**类型标注**和**类型注释**进行类型提示和检查。

#### 类型提示简例

我们首先通过一个例子来简要说明。假如我们要向函数中添加关于类型的信息，首先需要按如下方式对它的参数和返回值设置类型标注：

```python
# headlines.py

def headline(text: str, align: bool = True) -> str:
    if align:
        return f"{text.title()}\n{'-' * len(text)}"
    else:
        return f" {text.title()} ".center(50, "o")

print(headline("python type checking"))
print(headline("use mypy", centered=True))
```

但是这样添加类型提示没有运行时的效果——如果我们用错误类型的 `align` 参数，程序依然可以在不报错、不警告的情况下正常运行。

```bash
$ python headlines.py
Python Type Checking
--------------------
oooooooooooooooooooo Use Mypy oooooooooooooooooooo
```

因此，我们需要静态检查工具来排除这类错误（例如 [PyCharm](https://www.jetbrains.com/pycharm/) 中就包含这种检查）。最常用的静态类型检查工具是 [Mypy](http://mypy-lang.org/)。

```bash
$ pip install mypy
Successfully installed mypy.

$ mypy headlines.py
Success: no issues found in 1 source file
```

如果没有报错，说明格式检查通过；否则，会提示出问题的地方。*值得注意的是，类型检查向下兼容，比如整数就可以在 Mypy 中通过浮点数类型标注的检查。*

这种检查对于写出可读性较好的代码是十分有帮助的——Bernát Gábor 曾在他的 [The State of Type Hints in Python](https://www.bernat.tech/the-state-of-type-hints-in-python/) 中说过，“类型提示应当出现在任何值得单元测试的代码里”。

接下来我们将会更详细地介绍 Python 的类型检查系统：如何运行静态类型检查工具（主要是 Mypy）、如何检查引用了不含类型提示的静态库的代码、如何在运行时检查类型标注。

#### 类型标注

类型标注是自 Python 3.0 开始引入的特征，是添加类型提示的重要方法。例如这段代码就引入了类型标注，你可以通过调用 `circumference.__annotations__` 来查看函数中所有的类型标注。

```python
import math

def circumference(radius: float) -> float:
    return 2 * math.pi * radius
```

当然，除了函数函数，变量也是可以类型标注的，你可以通过调用 `__annotations__` 来查看函数中所有的类型标注。

```python
pi: float = 3.142

def circumference(radius: float) -> float:
    return 2 * pi * radius
```

变量类型标注赋予了 Python 静态语言的性质，即声明与赋值分离：

```python
>>> nothing: str
>>> nothing
NameError: name 'nothing' is not defined

>>> __annotations__
{'nothing': <class 'str'>}
```

#### 类型注释

如上所述，Python 的类型标注是 3.0 之后才支持的，这说明如果你需要编写支持遗留Python 的代码，就不能使用标注。为了应对这个问题，你可以尝试使用类型注释——一种特殊格式的代码注释——作为你代码的类型提示。

```python
import math

pi = 3.142  # type: float

def circumference(radius):
    # type: (float) -> float
    return 2 * pi * radius
  
def headline(text, width=80, fill_char="-"):
    # type: (str, int, str) -> str
    return f" {text.title()} ".center(width, fill_char)

def headline(
    text,           # type: str
    width=80,       # type: int
    fill_char="-",  # type: str
):                  # type: (...) -> str
    return f" {text.title()} ".center(width, fill_char)

print(headline("type comments work", width=40))
```

这种注释不包含在类型标注中，你无法通过 `__annotations__` 找到它，同类型标注一样，你仍然可以通过 Mypy 运行得到类型检查结果。

#### 类型别名

当然 Python 有更强大的类型提示功能，支持复杂的类型提示，例如 `List[Tuple[str, str]]`。我们可以为这些类型制作别名来让代码可读性更好，`Card = Tuple[str, str]`。