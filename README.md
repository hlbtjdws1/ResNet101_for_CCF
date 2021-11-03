# ResNet101_for_CCF(GPU环境)


> **注意：** 该教程的代码是基于`v1.5.0`版本的MindSpore和python3.7.5版本开发完成的。

## 上手指导

### 安装系统库

* 系统库

    ```
    sudo apt install -y unzip
    ```

* Python库

    ```
    pip install opencv-python
    ```

* MindSpore (**v1.5**)

    MindSpore的安装教程请移步至 [MindSpore安装页面](https://www.mindspore.cn/install).

下载商品数据及
```
从以下地址下载https://gas.graviti.cn/dataset/graviti-open-dataset/RP2K从浏览器中下载该数据集，手动解压。
更改数据集名称为RP2K
```

### 模型训练

```
此处名称要对应
python3.7 train.py --dataset_path ./RP2K/ALL/Train
```


### 下载ResNet-101预训练模型（推理任务使用）

您可以直接点击 (https://github.com/hlbtjdws1/ResNet101_for_CCF) 从浏览器中下载预训练模型。

### 模型精度验证

```
python3.7 eval.py –checkpoint_path ./renet-50_7184.ckpt – dataset_path ./RP2K/ALL/Test
```


## 许可证

[Apache License 2.0](../../LICENSE)
