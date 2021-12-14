# PSGLoss
Progressive Self-Guided Loss for Salient Object Detection

This repository contains the reference code for our TIP 2021 paper. The pdf can be found in [this link](https://arxiv.org/pdf/2101.02412.pdf)

If you use any part of our code, or PSGLoss is useful for your research, please consider citing:
```
@ARTICLE{yang2021progressive,
  author={Yang, Sheng and Lin, Weisi and Lin, Guosheng and Jiang, Qiuping and Liu, Zichuan},
  journal={IEEE Transactions on Image Processing}, 
  title={Progressive Self-Guided Loss for Salient Object Detection}, 
  year={2021},
  volume={30},
  number={},
  pages={8426-8438},
  doi={10.1109/TIP.2021.3113794}}
```

model ckpt and results can be found in [this link](https://drive.google.com/file/d/1aVp6oM1Qrv5cO97oFqOGbz3IwDqmgsNE/view?usp=sharing) 

## Requirements
* Pytorch 0.4 !!!
* opencv-python

### Train/Test 
Train:
```bash
 CUDA_VISIBLE_DEVICES=0 python ystrain.py --batchsize 20 --lr 5e-5 --trainsize 352 --loss dicebce --randomflip --psgloss 
```

Test:
```bash
CUDA_VISIBLE_DEVICES=0 python test_ys.py --checkpointfile xxxx --batchsize 20 --lr 5e-5  --loss dicebce --testsize 352
```

## Acknowledgments
Code and data prepration largely benefits from [CPD](https://github.com/wuzhe71/CPD)
