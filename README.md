# ConDA-gen-text-detection
Code for the paper: ConDA: Contrastive Domain Adaptation for AI-generated Text Detection accepted at IJCNLP-AACL 2023 [paper link](https://arxiv.org/abs/2309.03992).



## Setup

Make directories for the models, output logs and huggingface model files.

`mkdir models huggingface_repos output_logs`

Download `roberta-base` and/or `roberta-large` and place these repositories in `huggungface_repos`

`contrast_training_with_da.py` is the ConDA training script. The `multi_domain_runner.py` is the runner script for training ConDA models. Update the arguments in `multi_domain_runner.py` to train models as needed. 
