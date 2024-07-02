<p align="center">
  <h1 align="center"><ins>ImageMatch</ins> ⚡️

## Methods
### Step1. 基于retinex理论增强低光图像便于匹配（非必需）
### Step2. 基于LightGlue找到匹配特征点
### Step3. 根据特征点得出单应性变换矩阵并变换图像

## Use for single_image (MSR + LightGlue)
```bash
python single_match.py
```
## Tips
```txt
在lightglue/superpoint.py设置了"remove_borders": 50以忽略图片周围时间文字，但不一定全适用
```

## Installation and demo
We provide a [demo notebook](demo.ipynb) which shows how to perform feature extraction and matching on an image pair.


