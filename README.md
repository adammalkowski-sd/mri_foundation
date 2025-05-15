# Finetune SAM on your customized medical imaging dataset

This is the official code for our paper: [How to build the best medical image segmentation algorithm using foundation models: a comprehensive empirical study with Segment Anything Model](https://arxiv.org/abs/2404.09957), where we propose a new foundation model designed specifically for MRI.

The final weights can be accessed [here](https://drive.google.com/file/d/1nPkTI3H0vsujlzwY8jxjKwAbOCTJv4yW/view?usp=sharing).

explore three popular scenarios when fine-tuning foundation models to customized datasets in the medical imaging field: (1) only a single labeled dataset; (2) multiple labeled datasets for different tasks; and (3) multiple labeled and unlabeled datasets; and we design three common experimental setups, as shown in figure 1.
![Fig1: Overview of general fine-tuning strategies based on different levels of dataset availability.](https://github.com/mazurowski-lab/finetune-SAM/blob/main/finetune_strategy_v9.png)
