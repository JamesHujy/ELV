# Towards Interpretable Natural Language Understanding with Explanations as Latent Variables

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
### Usage

This repo contains the official code release for NeurIPS 2020 paper [Towards Interpretable Natural Language Understanding with Explanations as Latent Variables](https://arxiv.org/pdf/2011.05268.pdf).

Directory [ELV_re](https://github.com/JamesHujy/ELV/tree/main/ELV_re) and [ELV_sa](https://github.com/JamesHujy/ELV/tree/main/ELV_sa) contain source code for Relation Extraction task and Sentimental Analysis task in supervised setting. There are some tiny differences in training details and pre-processing. Directory [EST_re](https://github.com/JamesHujy/ELV/tree/main/EST_re) and [EST_sa](https://github.com/JamesHujy/ELV/tree/main/EST_sa) contain codes in semi-supervised setting. 

Directory [data](https://github.com/JamesHujy/ELV/tree/main/data) contain data used in our experiments. TACRED is not released because of its copyright, which can be downloaded at [LDC TACRED webpage](https://catalog.ldc.upenn.edu/LDC2018T24). 

To replicate the result of experiment, run the bash script in each directory. For example, to replicate the ELV result on Semeval dataset, just use

```
bash train_semeval.sh
```

### Implementation

We implement Bert classifier based on [Huggingface transformers](https://github.com/huggingface/transformers) and Unilm generator based on [Microsoft Unilm](https://github.com/microsoft/unilm). 

### Citation 
If you find this repo useful, please cite: 
```
@inproceedings{zhou2020towards,
  title={Towards Interpretable Natural Language Understanding with Explanations as Latent Variables},
  author={Zhou, Wangchunshu and Hu, Jinyi and Zhang, Hanlin and Liang, Xiaodan and Sun, Maosong and Xiong, Chenyan and Tang, Jian},
  booktitle={NeurIPS},
  year={2020}
}
```
