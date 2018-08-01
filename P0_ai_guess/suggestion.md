# 关于 p0 的一些想法



Hello Rita,
    感谢要求 review p0，在做的时候，发现一些问题，记录如下，望指正



- [关于 p0 的一些想法](#关于-p0-的一些想法)
    - [导入库](#导入库)
    - [激活函数名称](#激活函数名称)
    - [无需 batch](#无需-batch)
    - [误差反向传播](#误差反向传播)
    - [三个方程中的 m](#三个方程中的-m)
    - [loss问题](#loss问题)
    - [以上](#以上)




## 导入库
项目使用时，发现图表显示有时有问题，可能需要导入以下库

``` python
from IPython.display import display
% matplotlib inline
```



## 激活函数名称
>任务5: 训练多分类的神经网络
>下列函数会训练二层神经网络。 首先，我们将写一些 helper 函数。
>- Sigmoid 激活函数
>
>$$\sigma(x) = \frac{e^{x_i}} {\sum_{i=1}^{m} e^{x_i}}$$
>

这时似乎是 softmax 函数




## 无需 batch

从 `train_nn` 的代码中，可以发现没有使用 batch 进行训练，我想对于初学者来说这是非常合适的。

``` python
def train_nn(features, targets, test_data, test_label, epochs, learnrate):
    ......
    for e in range(epochs):
        ......
        for x, y in zip(features.values, targets.values):
```

但对应的几个公式也需要一起配合

>任务5: 训练多分类的神经网络
>下列函数会训练二层神经网络。 首先，我们将写一些 helper 函数。
>- Sigmoid 激活函数
>
>$$\sigma(x) = \frac{e^{x_i}} {\sum_{i=1}^{m} e^{x_i}}$$
>
>- 误差函数
>
>$$ loss = \frac{1} {m} {\sum_{i=1}^{m} (y * \log{\hat{y}})}$$
>

- 如果 loss 这里的 m 指的是 batch size，那么这个 m 似乎不需要
- 如果 loss 中的 m 不是指 batch size，而是指 one hot 的每一项的话，y 需要下标
- 而且误差需要为正数，所以还是加上一个负号
- 建议如下：

$$ loss = - {\sum_{i=1}^{m} ({y_i} * \log{\hat{y_i}})}$$

同时，可注明 m 为 one hot 的每一项。另，对于 * 和 ⋅ 我也不太搞定请是矩阵乘法，还是内积乘法，所以可能说错了，请谅解



## 误差反向传播
> 现在轮到你来练习，编写误差项。 记住这是由方程 
> $$  \frac{x  \cdot ({\hat{y} - y})} {m} $$ 给出的。

也同样不需要代表 batch size 的 m 

## 三个方程中的 m
以上三个方程中的 m 分别指
- x 的特征数
- y 的 one hot 数
- batch size 的值

所以可能会有一点混淆，望详细说明 m 代表的意义



## loss问题
如果，接受上面的 `loss = -xxx` 那么 `weights -= learnrate * del_w / n_records` 这时也使用负号。 


## 以上
以上， 是我在做项目时，想到的一些问题，如有考虑不周之处，请及时指出