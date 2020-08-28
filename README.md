# advGAN-tf2

## 运用tensorflow2实现advGAN。

------

### 准备工作：

1. 准备 **目标模型(TargetModel)** ，将 模型权重文件 放置个人指定的文件夹。注意，本代码适用的模型 .hdf5 文件。

2. 准备训练advGAN的数据集，包括 **训练集** 和 **测试集**，放置在个人指定的文件夹。

   注意，数据集准备包括2部分（ 图片 和 对应标签文件 ）：

   1. 图片的准备。
   2. 标签文件 的准备。标签文件格式 `XXXX.jpg [类别号]`，如`/dataset/XXX/adganoi2XX3rnadfa.jpg 21 `。

3. 修改 **config.py** 文件中的参数。主要包括

   1. `CLASS_NAME`： 目标模型输出的**类别名称**。
   2. `TRAIN_ANNOT_PATH`, `TEST_ANNOT_PATH`：分别为之前已准备 训练集 和 测试集 的**标签文件路径**。
   3. `MODEL_PATH`： **目标模型权重路径**。
   4. `Model parameters` 参数下的各个参数：主要是 目标模型、生成器和判别器 的 **inputs' shape** 。
   5. `Attack algorithm parameters` 参数下的各个参数：用于调节训练GAN和对抗样本的对抗程度，按需修改。

### 训练advGAN：

`python advGAN.py`

### 测试advGAN：

`python attack.py`

------

注意：生成器 和 判别器 是我自己搭建的，可以按需修改，或参照其他GAN模型。

参考：

1. 论文 https://arxiv.org/abs/1801.02610
2. 代码 https://github.com/ctargon/AdvGAN-tf
3. 代码 https://github.com/mathcbc/advGAN_pytorch