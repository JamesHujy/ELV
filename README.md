# ELV

### Usage

Afficial source code for [Towards Interpretable Natural Language Understanding with Explanations as Latent Variables](https://arxiv.org/pdf/2011.05268.pdf).

Directory [ELV_re](https://github.com/JamesHujy/ELV/tree/main/ELV_re) and [ELV_sa](https://github.com/JamesHujy/ELV/tree/main/ELV_sa) is source code for Relation Extraction task and Sentimental Analysis task in supervised setting(Use all classification labels). There are some tiny difference in training details and pre-processing. Directory [EST_re](https://github.com/JamesHujy/ELV/tree/main/EST_re) and [EST_sa](https://github.com/JamesHujy/ELV/tree/main/EST_sa) are codes in semi-supervised setting(Use part of classificatioin labels). 

Directory [data](https://github.com/JamesHujy/ELV/tree/main/data) is data used in our experiment. TACRED is not released because of its copyright. It can be bought and downloaded at [LDC TACRED webpage](https://catalog.ldc.upenn.edu/LDC2018T24). 

To replicate the result of experiment, run the bash script in each directory. For example, to replicate the ELV result on Semeval dataset, just use

```
bash train_semeval.sh
```

### Implementation

We implement Bert classifier based on [Huggingface transformers](https://github.com/huggingface/transformers) and Unilm generator based on [Microsoft Unilm](https://github.com/microsoft/unilm). 

### Citation 

```
@inproceedings{jiang2019language,
  title={Language as an abstraction for hierarchical deep reinforcement learning},
  author={Jiang, Yiding and Gu, Shixiang Shane and Murphy, Kevin P and Finn, Chelsea},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019}
}
```





