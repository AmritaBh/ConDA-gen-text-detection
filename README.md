# ConDA-gen-text-detection
Code for the paper: ConDA: Contrastive Domain Adaptation for AI-generated Text Detection accepted at IJCNLP-AACL 2023 [paper link](https://arxiv.org/abs/2309.03992).

![ConDA Framework Diagram](https://github.com/AmritaBh/ConDA-gen-text-detection/conda-framework.png)

## Setup

Set up a separate environment and install requirements via `pip install -r requirements.txt`

Make directories for the models, output logs and huggingface model files.

`mkdir models huggingface_repos output_logs`

Download `roberta-base` from [here](https://huggingface.co/roberta-base/tree/main) and/or `roberta-large` from [here](https://huggingface.co/roberta-large/tree/main) and place these repositories in `huggungface_repos`.

`contrast_training_with_da.py` is the ConDA training script. The `multi_domain_runner.py` is the runner script for training ConDA models. Update the arguments in `multi_domain_runner.py` to train models as needed. 

Use the `evaluation.py` script for evaluating models. Change arguments within the `evaluation.py` script as needed.
