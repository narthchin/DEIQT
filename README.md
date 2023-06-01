# DEIQT

Checkpoints, logs and source code for AAAI-23 paper '[Data-Efficient Image Quality Assessment with Attention-Panel Decoder](https://arxiv.org/abs/2304.04952)'

## Updates

* [06/2023] We released the source code of 'DEIQT', check the code on [GitHub](https://github.com/narthchin/DEIQT)
* [04/2023] We released our work 'DEIQT', the paper is now on [Arxiv](https://arxiv.org/abs/2304.04952).

## To-Dos

* [ ] Checkpoints & Logs
* [x] Initialization

## Dependencies

* Python
* PyTorch

## Usage

### Pre-requisition

#### Weights

Download the Pre-trained ViT-S[224] weights from [DeiT III](https://github.com/facebookresearch/deit/blob/main/README_revenge.md)

#### WandB

This project use [WandB](https://wandb.ai/site) to log information and report. Remember to adjust the code in main.py to suit your research.

### Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 torchrun --nnodes 1 --nproc_per_node 4 --master_port 49935  main.py \
--cfg [CONFIG_PATH] \
--data-path [YOUR_DATA_PATH] \
--output [LOG_PATH] \
--tag [REMARK_TAG] \
--repeat \
--rnum [TARGET_REPEAT_NUM]
```

## Citing DEIQT

If you find this project helpful in your research, please consider citing our papers:

```text
@inproceedings{qin2023deiqt,
  title={Data-Efficient Image Quality Assessment with Attention-Panel Decoder},
  author={Guanyi Qin and Runze Hu and Yutao Liu and Xiawu Zheng and Haotian Liu and Xiu Li and Yan Zhang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year={2023}
}
```

## Acknowledgement

We borrowed some parts from the following open-source projects:

* [HyperIQA](https://github.com/SSL92/hyperIQA)
* [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
* [DeiT III](https://github.com/facebookresearch/deit/blob/main/README_revenge.md)

Many thanks to them.
