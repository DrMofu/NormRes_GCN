### 归一残差图卷积网络

作者：杜欣伟

原论文：基于归一残差的图卷积图像检索网络

简介：本项目应用优化后的图卷积网络进行图像检索。本项目在[GSS](https://github.com/layer6ai-labs/GSS)论文的基础上采用了归一残差机制，并在一定程度上提升了图像检索算法的性能。



#### 文件详情
| 文件名 | 说明 |
| --- | --- |
| data.py | 数据读取 |
| evaluate.py | 模型性能评价代码 |
| graph.py | 图构建代码 |
| model.py | 模型构建代码 |
| training.py | 模型训练代码 |
| setting.py | 超参数设置 |
| main.py | Demo，训练并测试模型性能 |

#### 模型性能
![不同模型在ROxford和RParis数据集上的mAP检索效果](./imgs/tabel_1.png)