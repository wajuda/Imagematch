<p align="center">
  <h1 align="center"><ins>ImageMatch</ins> ⚡️

## Methods
#### Step1. 基于retinex理论增强低光图像便于匹配（非必需）
#### Step2. 基于[LightGlue](https://github.com/cvg/LightGlue)找到匹配特征点
#### Step3. 根据特征点得出单应性变换矩阵并变换图像

## Use for single_image
```bash
python single_match.py
```
## Tips
```txt
在lightglue/superpoint.py设置了"remove_borders": 50以忽略图片周围时间文字，但不一定全适用
```

## Installation and demo
git clone --quiet https://github.com/wajuda/Imagematch/

!pip install --progress-bar off --quiet -e .

## Examples

### E.g 1
![0a](https://github.com/wajuda/Imagematch/assets/112617153/b55e8ff5-7d95-4e74-8f7d-cfcba988bc8c)
![0b](https://github.com/wajuda/Imagematch/assets/112617153/5a28cfc9-51ac-40fa-8cfd-d78d29bed627)
![0c](https://github.com/wajuda/Imagematch/assets/112617153/096887dd-1b66-4876-86ee-d93a597dbd7f)

### E.g 2
![0a (1)](https://github.com/wajuda/Imagematch/assets/112617153/62bb7e3c-45c2-4f0d-9387-f140ba47ac1a)![0_match](https://github.com/wajuda/Imagematch/assets/112617153/ba4804f0-dac4-42fe-b60c-0ef964d13f15)![0b](https://github.com/wajuda/Imagematch/assets/112617153/d0ad830b-2666-4f05-af73-fa8b728c5cea)




