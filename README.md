# ConDA-gen-text-detection
Code for the paper: **ConDA: Contrastive Domain Adaptation for AI-generated Text Detection** accepted at IJCNLP-AACL 2023 [paper link](https://arxiv.org/abs/2309.03992).

![ConDA Framework Diagram](https://github.com/AmritaBh/ConDA-gen-text-detection/blob/main/conda-framework.png)

## Setup

Set up a separate environment and install requirements via `pip install -r requirements.txt`

Make directories for the models, output logs and huggingface model files.

`mkdir models huggingface_repos output_logs`

Download `roberta-base` from [here](https://huggingface.co/roberta-base/tree/main) and/or `roberta-large` from [here](https://huggingface.co/roberta-large/tree/main) and place these repositories in `huggungface_repos`.

`contrast_training_with_da.py` is the ConDA training script. The `multi_domain_runner.py` is the runner script for training ConDA models. Update the arguments in `multi_domain_runner.py` to train models as needed. 

Use the `evaluation.py` script for evaluating models. Change arguments within the `evaluation.py` script as needed.

## TuringBench

Link to the dataset website: [link](https://turingbench.ist.psu.edu/)
Link to the TuringBench paper: [link](https://arxiv.org/abs/2109.13296)

Files should be split into 3 jsonl splits: train, valid, test. Each line in the jsonl is a data instance with `text` and `label` fields.

## Links to best performing models for each target generator

Here we provide links to pre-trained ConDA models for the best performing models:

| Target  | Best performing source | Dropbox Link |
| :-----------: | :-----------: | :-----: |
| CTRL  | GROVER_mega  | [link](https://www.dropbox.com/s/h5prhx3j4yndoig/grover_mega_ctrl_syn_rep_loss1.pt?dl=0) |
| FAIR_wmt19  | GPT2_xl  | [link](https://www.dropbox.com/s/h36fh24qu9203pf/gpt2_xl_fair_wmt19_syn_rep_loss1.pt?dl=0) |
| GPT2_xl | FAIR_wmt19  | [link](https://www.dropbox.com/s/mnx5lyg4geebhm6/fair_wmt19_gpt2_xl_syn_rep_loss1.pt?dl=0) |
| GPT3  | GROVER_mega  | [link](https://www.dropbox.com/s/mh09c8kdinocsz9/grover_mega_gpt3_syn_rep_loss1.pt?dl=0) |
| GROVER_mega  | CTRL  | [link](https://www.dropbox.com/s/o0fs8dodywvuda0/ctrl_grover_mega_syn_rep_loss1.pt?dl=0) |
| XLM  | GROVER_mega  | [link](https://www.dropbox.com/s/q6ddq2aop9qw8lo/grover_mega_xlm_syn_rep_loss1.pt?dl=0) |
| ChatGPT  | FAIR_wmt19  | [link](https://www.dropbox.com/s/sgwiucl1x7p7xsx/fair_wmt19_chatgpt_syn_rep_loss1.pt?dl=0) |


# Contact

For any questions, comments, and feedback, contact Amrita Bhattacharjee at abhatt43@asu.edu
