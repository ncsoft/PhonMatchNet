# :mag: PhonMatchNet: Phoneme-Guided Zero-Shot Keyword Spotting for User-Defined Keywords

Official implementation of PhonMatchNet: Phoneme-Guided Zero-Shot Keyword Spotting for User-Defined Keywords.

PyTorch version: https://github.com/ncsoft/PhonMatchNet/tree/pytorch

## Requirements

### Datasets

* [LibriPhrase](https://github.com/gusrud1103/libriphrase)

* [Google Speech Commands](https://arxiv.org/abs/1804.03209)

* [Qualcomm Keyword Speech](https://developer.qualcomm.com/project/keyword-speech-dataset)

Download the dataset and prepare it according to each guide. 

## Getting started

### Environment

```bash
cd ./docker
docker build --tag udkws .
```

### Training

```bash
docker run -it --rm --gpus '"device=0,1"' \
    -v /path/to/this/repo:/home/ \
    -v /path/to/prepared/dataset:/home/DB \
    ukws \
    /bin/bash -c \
    "python train.py \
        --epoch 100 \
        --lr 1e-3 \
        --loss_weight 1.0 1.0 \
        --audio_input both \
        --text_input g2p_embed \
        --stack_extractor \
        --comment 'user comments for each experiment'"

```

### Monitoring

```bash
tensorboard --logdir ./log/ --bind_all
```

## CONTRIBUTING

Please post bug reports and new feature suggestions to the Issues and Pull requests tabs of this repo.
