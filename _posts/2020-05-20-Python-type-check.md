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

前几天，Henry 安排我调研一下 Python 3 的 typing 和 annotations，他说这个一个很有用的 feature，尤其是对软件开发的工作而言。我突然发现 LeetCode 平台的 Solution 其实就用了 typing check，于是在这里给大家分享一下这方面的知识。

传统上，Python 解释器以灵活但隐式的方式处理类型。Python 最新的几个版本允许您指定明确的类型进行提示，有些工具可以使用这些提示来帮助您更有效地开发代码。那么最新的几个版本都做了什么呢？Python 3.0 开始支持函数注释，3.5、3.6 开始支持变量注释。从 3.7 开始，`from __future__ import annotation`  已经可以添加 Python 4 风格的纯文本注释了，相信未来 Python4 强大的注释功能一定能成为 Python 这门语言的巨大优势。

这篇文章主要涉及以下几个方面：

- 类型注释和类型提示
- 为代码添加静态类型检查
- 运行静态类型检查工具
- 在运行时强制类型

### 动态和静态类型