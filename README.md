# FingerveinRecognition
> 一个实现指静脉识别的整合应用，包括算法，图像处理以及应用GUI⚡

---

最新release：
<h3 align="center"><a href= 'https://github.com/ExcaliburEX/FingerveinRecognition/releases/download/V1.1/FingerveinRecogntion.exe'><img alt="GitHub release (latest by date)" src="https://img.shields.io/github/downloads/ExcaliburEX/FingerveinRecognition/V1.1/total?color=success&flat-square&logo=Cachet"></a></h3>

# 更新日志
- 2021-07-04：因为VGG19下载的预训练模型在Github导致有可能下载失败，因此将模型文件存储到自己的腾讯云OSS桶，通过程序自动下载，手动设置了VGG19的预训练模型的路径，规避了因为网络问题导致的下载失败。

# 使用说明

## 代码说明

1. 源代码在`finger.py`
2. 根据源代码生成的可执行文件为`finger.exe`，可直接运行
3. 编译源代码所需Python环境为3.7，其他库在`requirements.txt`

## 使用方法

1. 可直接运行`finger.exe`
2. 参数说明
   1. `单人手指图片个数`：训练集中一根手指录入的图片数，本例子设为7，因为训练集中总共21张图片，每根手指7张图片，总共三根手指。
   2. `随机选取的图片测试个数`：识别时，在训练集中随机选取的与待识别图片比对的图片数，取值范围应为[1~7]之间，若取5，意义则为从每根手指的7张图片随机选取5张与待识别的图片进行5次比对，求得相似度平均值。
   3. `训练集图片文件夹`：训练集图片所在文件夹，训练完毕后会在该文件夹下生成`trained`文件夹，存储处理完成后的图片，本例子为`HighGuardFinger`。
   4. `待识别图片文件夹`：待识别的图片文件夹，本例子选取了7张图片为训练集，剩下的第8张图片作为测试图片，文件夹在`HighGuardTest`。

3. 首次运行识别时，会下载`VGG16`的预训练模型，大小在50M左右，以后运行无需下载。
4. 先设置上述说列参数，再进行`开始训练`，待训练完毕后，进行`开始识别`，则会将`HighGuardTest`文件夹中的图片依据训练所生成的`feature_data.npz`特征向量，进行比对，比对方式就是计算余弦值，进而获得识别概率。

## 程序逻辑说明

- 主要包含三大模块，`VGG16`模型，图像处理和GUI模块
- 训练流程：会将待训练图片，进行`自适应直方图`—>`直方图均衡化`—>`二值化`—>`Gaborfilter滤波`处理，将最后得到的`Gaborfilter滤波`处理过的图片丢进`VGG16`，获得对应的特征向量，进行存储。
- 识别流程，将以上流程重复一遍，将待识别图片的特征向量，与数据库中所有特征向量进行比较，得出最相似的手指集对应的手指编号。


# 运行截图

<table>
<tr>
<th align = "center" colspan="1">
<div style="text-align: center;">
<h4 align="center">训练</h4>
</div></th>
</tr>
<tr>
<td align = "center"><img src="https://i.loli.net/2021/07/03/9NIfKUvqwjy8coL.gif"></td>
</tr>
</table>

<table>
<tr>
<th align = "center" colspan="1">
<div style="text-align: center;">
<h4 align="center">识别</h4>
</div></th>
</tr>
<tr>
<td align = "center"><img src="https://i.loli.net/2021/07/03/u9zB7fIkoqaylQ5.gif"></td>
</tr>
</table>