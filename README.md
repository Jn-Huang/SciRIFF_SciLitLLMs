# SciRIFF

**This repo is forked from [SciRIFF](https://github.com/allenai/SciRIFF), with *minor modification* to reproduce the results in [ScILitLLM paper](https://arxiv.org/abs/2408.15545).** **The difference is that this repo use a newer version of** `lm-evaluation-harness` **to handle LLM chat templates.**

**For LLM-based evaluations in MuP and Qasper, we use GPT-4o instead of GPT-3.5, which is different from what was done in the original paper.** 

**Thanks the original authors for their wonderful work!** 


SciRIFF is a collection of 54 tasks targeting instruction-following over scientific literature. Tasks were created by converting existing scientific datasets to a common instruction-following format via expert-written templates. The SciRIFF dataset, as well as the SciTulu models trained on SciRIFF, are available in the Hugging Face [SciRIFF collection](https://huggingface.co/collections/allenai/sciriff-665f61ba7315e1d202e5f6bf). This repository contains code to evaluate the SciTulu models on 9 held-out SciRIFF tasks, as well as details explaining how to use the data to train new models. Shortly, we will add templates for all tasks, as well as code to recreate the dataset using these templates.

**Table of Contents**

- [Setup](#setup)
- [Evaluation](#evaluation)

## Setup

We recommend using a Conda env:

```bash
conda create --name sciriff python=3.11
conda activate sciriff
```

We use the Eleuther harness to handle inference for evaluation. Different from the original repo, we suggest use a specific commit (version 0.4.4 ) of `lm-evaluation-harness` to handle chat templates easily (code below). To use versions after 0.4.5, you will need to modify the dataset setting under `sciriff/eval/eleuther_templates/general`. See [this link](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md#advanced-group-configs) for more information.

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
git checkout 543617f
pip install -e .[vllm]
```

Then, install `sciriff` by `cd`ing back to the project directory and running:

```bash
pip install -e .
```

You may get a warning about incompatible versions of the `typer` package; this can safely be ignored.

For two of our evaluations, we use **GPT-4o** as an LM judge. In order to do these evaluations, you'll need an OpenAI API key:

```bash
export OPENAI_API_KEY=[your_openai_key]
```


## Evaluation

To evaluate, we first use the Eleuther harness to handle inference, and then run evluations on the results. For implementation details, see [evaluation.md](doc/evaluation.md). For examples of each evaluation task, see [evaluation_tasks.md](doc/evaluation_tasks.md)

### Making predictions

Use `predict_eleuther.py` to make predictions for all eval tasks. The example below makes predictions using SciTulu-7B. The results will go in `results/predictions/scitulu-7b`.

For the 7B, you should be fine using a single A6000 gpu. For the 70B, we've generally used 4 80GB A100's or similar, but it may be possible to do with less. Inference on the whole eval set will take a few hours; you can use the `--limit` flag to cap the number of instances per task.

**We add an argument `--apply_chat_template`, so that `lm-evaluation-harness` will apply the chat template automatically.**

```bash
python script/eval/predict_eleuther.py \
  --model vllm \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --chat_template general \
  --gpus 1 \
  --tasks science_adapt \
  --result_base results/predictions/  \
  --apply_chat_template
```

To make predictions with an API model, you can do:

```bash
python script/eval/predict_eleuther.py \
    --model=openai-chat-completions \
    --model_name=gpt-3.5-turbo-1106 \
    --chat_template general \
    --tasks science_adapt \
    --result_base results/predictions \
    --limit 50
```

You can download all the prediction files [here](https://drive.google.com/drive/folders/1OnqT_pCAVlB9ia6W3WX9k-koKKmqvTZB?usp=share_link).

### Computing metrics

Run `compute_science_metrics.py` to compute metrics based on the model predictions.

```bash
python script/eval/compute_science_metrics.py \
  --pred_dir results/predictions \
  --metrics_dir results/metrics
```

If you've run predictions `predict_eleuther.py` on multiple models, this will evaluate all models for which predictions are available under `results/predictions`. Metrics for each model will be saved to `results/metrics/by_model/{model_name}`; there will be a subfolder for each task including cleaned-up predictions and detailed metrics.

Metrics from all models will be collected in `results/metrics/tables`. The file `results/metrics/tables/summary.tsv` provides a summary of the metrics for all tasks.
