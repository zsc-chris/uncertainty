**This software is released under MIT license.**
**本软件在MIT许可下发行**

This module provides `uncertainty` class that helps address error calculations in college-level courses like Basic Physics Lab and Physical Chemistry Lab.

该模块提供`uncertainty`（不确定度）类，帮助解决各高校《基础物理实验》（基物实验）和《物理化学实验》（物化实验）等课程中的误差计算。

Functions功能
=============

1. Provide a class called **`uncertainty`** that has three attributes: `value`, `error`, `name` (optional)
2. Error is calculated on the fly, and **$\LaTeX$ expression that lead to the error is easily generated** on request.
3. Each variable is created independently so that the **correlation induced by repeated presence of the same variable into account**.
4. This class is compatible as a `dtype` with vector operation classes like `numpy.ndarray`, `pandas.DataFrame`, so that **large amount of data can be processed**.
5. **Supports all scientific operations**: power, exponential, logarithm, trigonometric function, etc.
6. Operations that **introduce additional error between multiple repeated measurements** are implemented: mean and least-square linear regression.
<!-- -->
1. 提供一个名为 **`uncertainty`（不确定度）** 的类，该类具有三个属性：`value`（值）、`error`（误差）、`name`（名称，可选）
2. 动态计算误差，并且可以根据要求轻松**生成推导误差的 $\LaTeX$ 表达式**。
3. 每个变量都是独立创建的，以**考虑由相同变量重复存在引起的相关性**。
4. 此类可作为`dtype`与`numpy.ndarray`、`pandas.DataFrame`等向量操作类兼容，**以处理大量数据**。
5. **支持所有科学运算**：幂、指数、对数、三角……
6. 实现了在**多次重复测量之间引入额外误差的操作**：均值和最小二乘线性回归。

Basic Usage用法简介
===================

1. Download `uncertainty.py` and place it under your data folder.
2. Open Python in that folder and input `from uncertainty import uncertainty`
3. Use `uncertainty(value, error, name)` to create an `uncertainty` object "name = value ± error". Use float as constant.
4. Call `.latex()` method on object to generate $\LaTeX$ expression.
5. `uncertainty.mean`, `uncertainty.lsmr` are two class functions that allows you to consider the extra error introduced by multiple measurements.
6. Try using `uncertainty` as `dtype` in NumPy and pandas in order to process large amount of data.
7. Refer to doc by Python `help` whenever necessary. If there is bug, please report to me: [zsc_chris@outlook.com](mailto:zsc_chris@outlook.com). I'll try to solve it with you.
<!-- -->
1. 下载`uncertainty.py`并将其放在您的数据文件夹下。
2. 在该文件夹中打开Python并输入`from uncertainty import uncertainty`
3. 使用`uncertainty(value, error, name)`创建一个`uncertainty`对象name = value ± error。使用浮点数表示常数。
4. 在对象上调用`.latex()`方法生成 $\LaTeX$ 表达式。
5. `uncertainty.mean`、`uncertainty.lsmr`是两个类函数，可让您考虑多次测量引入的额外误差。
6. 尝试在NumPy和pandas中使用`uncertainty`作为`dtype`，以处理大量数据。
7. 有任何需求，请用Python `help`功能查看文档。如果有bug，请联系我：[zsc_chris@outlook.com](mailto:zsc_chris@outlook.com)。我会尝试和您一起解决。

Requirements依赖
================

`torch`*, `sympy`

*just CPU version is OK只要CPU版本即可

Acknowledgement致谢
===================

Thank the teaching assistant Junhan Chang of "Machine Learning and its Applications in Chemistry" course for giving the inspiration of using PyTorch as autograd tool. Thank Zhaoyang Li, a top senior student in last grade, whose single-formula $\LaTeX$ generator [UncertaintyCalculator](https://github.com/FridrichMethod/UncertaintyCalculator) provides me the motivation to develop a more flexible $\LaTeX$ generator protocol.

感谢《机器学习及其在化学中的应用》课程的昌珺涵助教给了我使用PyTorch作为自动求导工具的启发。感谢比我高一级的学霸李昭阳学长，他的单公式 $\LaTeX$ 生成器[UncertaintyCalculator](https://github.com/FridrichMethod/UncertaintyCalculator)给了我开发更灵活的 $\LaTeX$ 生成方式的动力。
