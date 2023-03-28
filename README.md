# Dolly
Databricks’ Dolly, a large language model trained on the [Databricks Machine Learning Platform](https://www.databricks.com/product/machine-learning), demonstrates that a two-years-old open source model ([GPT-J](https://huggingface.co/EleutherAI/gpt-j-6B)) can, when subjected to just 30 minutes of fine tuning on a focused corpus of 50k records ([Stanford Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html)), exhibit surprisingly high quality instruction following behavior not characteristic of the foundation model on which it is based.  We believe this finding is important because it demonstrates that the ability to create powerful artificial intelligence technologies is vastly more accessible than previously realized.

Databricks is committed to ensuring that every organization and individual benefits from the transformative power of artificial intelligence. The Dolly model family represents our first steps along this journey, and we’re excited to share this technology with the world.

Please note that while GPT-J 6B is [Apache 2.0 licensed](https://huggingface.co/EleutherAI/gpt-j-6B), the Alpaca dataset is licensed under [Creative Commons NonCommercial (CC BY-NC 4.0)](https://huggingface.co/datasets/tatsu-lab/alpaca). 

**Dolly is intended exclusively for research purposes and is not licensed for commercial use.**

## Model Overview
In the following passages we refer to `dolly-6b`, the first in the Dolly family of models and the model that this repository presently implements. 

`dolly-6b` is a 6 billion parameter causal language model created by [Databricks](https://databricks.com/) that is derived from [EleutherAI’s](https://www.eleuther.ai/) [GPT-J](https://huggingface.co/EleutherAI/gpt-j-6B) (released June 2021) and fine-tuned on a ~52K record instruction corpus ([Stanford Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html)) consisting of question/answer pairs generated using the techniques outlined in the [Self-Instruct](https://arxiv.org/abs/2212.10560) paper.  Dolly was trained using [deepspeed](https://github.com/microsoft/DeepSpeed) [ZeRO 3](https://github.com/microsoft/DeepSpeed/blob/master/docs/code-docs/source/zero3.rst) on the [Databricks Machine Learning Platform](https://www.databricks.com/product/machine-learning) in just 30 minutes using a single [NDasrA100_v4](https://learn.microsoft.com/en-us/azure/virtual-machines/nda100-v4-series) machine with 8x A100 40GB GPUs. 

Like its base model, dolly-6b has six billion parameters consisting of 28 transformer layers with 16 attention heads each. It employs [Rotary Position Embedding](https://arxiv.org/abs/2104.09864) (RoPE) and shares the same tokenizer as GPT-3. GPT-J was trained on [The Pile](https://huggingface.co/datasets/the_pile), a 400B token dataset of diverse documents designed primarily for text generation tasks.

## Limitations
**`dolly-6b` is intended exclusively for research purposes and is not licensed for commercial use.**

`dolly-6b` is not a state-of-the-art generative language model and, though quantitative benchmarking is ongoing, is not intended to perform competitively with more modern model architectures or models subject to larger pretraining corpuses. For example, we expect the [Alpaca model](https://github.com/tatsu-lab/stanford_alpaca), derived from [LLaMA-7B](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) (trained on 1T tokens vs. The Pile's 400B & with years of scientific advances behind it), to be superior in its generative quality relative to Dolly. What's most notable about Dolly is the degree of its instruction following capabilities given that it's based on a freely available open source model anyone can download and use.

The Dolly model family is under active development, and so any list of shortcomings is unlikely to be exhaustive, but we include known limitations and misfires here as a means to document and share our preliminary findings with the community.  In particular, `dolly-6b` struggles with syntactically complex prompts, mathematical operations, factual errors, dates and times, open-ended question answering, hallucination, enumerating lists of specific length, and stylistic mimicry.  

## Get Started Training

* Add the `dolly` repo to Databricks (under Repos click Add Repo, enter `https://github.com/databrickslabs/dolly.git`, then click Create Repo).
* Start a `12.2 LTS ML (includes Apache Spark 3.3.2, GPU, Scala 2.12)` single-node cluster with node type having 8 A100 GPUs (e.g. `Standard_ND96asr_v4` or `p4d.24xlarge`). Note that these instance types may not be available in all regions, or may be difficult to provision. In Databricks, note that you must select the GPU runtime first, and unselect "Use Photon", for these instance types to appear (where supported).
* Open the `train_dolly` notebook in the Repo (which is the `train_dolly.py` file in the Github `dolly` repo), attach to your GPU cluster, and run all cells.  When training finishes, the notebook will save the model under `/dbfs/dolly_training`.

## Running Unit Tests Locally

```
pyenv local 3.8.13
python -m venv .venv
. .venv/bin/activate
pip install -r requirements_dev.txt
./run_pytest.sh
```
