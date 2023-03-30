# Dolly
Databricks’ [Dolly](https://huggingface.co/databricks/dolly-v1-6b), a large language model trained on the [Databricks Machine Learning Platform](https://www.databricks.com/product/machine-learning), demonstrates that a two-years-old open source model ([GPT-J](https://huggingface.co/EleutherAI/gpt-j-6B)) can, when subjected to just 30 minutes of fine tuning on a focused corpus of 50k records ([Stanford Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html)), exhibit surprisingly high quality instruction following behavior not characteristic of the foundation model on which it is based.  We believe this finding is important because it demonstrates that the ability to create powerful artificial intelligence technologies is vastly more accessible than previously realized.

Databricks is committed to ensuring that every organization and individual benefits from the transformative power of artificial intelligence. The Dolly model family represents our first steps along this journey, and we’re excited to share this technology with the world.

Please note that while GPT-J 6B is [Apache 2.0 licensed](https://huggingface.co/EleutherAI/gpt-j-6B), the Alpaca dataset is licensed under [Creative Commons NonCommercial (CC BY-NC 4.0)](https://huggingface.co/datasets/tatsu-lab/alpaca). 

The model is available on Hugging Face as [databricks/dolly-v1-6b](https://huggingface.co/databricks/dolly-v1-6b).

**Dolly is intended exclusively for research purposes and is not licensed for commercial use.**

## Model Overview
In the following passages we refer to `dolly-v1-6b`, the first in the Dolly family of models and the model that this repository presently implements. 

`dolly-v1-6b` is a 6 billion parameter causal language model created by [Databricks](https://databricks.com/) that is derived from [EleutherAI’s](https://www.eleuther.ai/) [GPT-J](https://huggingface.co/EleutherAI/gpt-j-6B) (released June 2021) and fine-tuned on a ~52K record instruction corpus ([Stanford Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html)) consisting of question/answer pairs generated using the techniques outlined in the [Self-Instruct](https://arxiv.org/abs/2212.10560) paper.  Dolly was trained using [deepspeed](https://github.com/microsoft/DeepSpeed) [ZeRO 3](https://github.com/microsoft/DeepSpeed/blob/master/docs/code-docs/source/zero3.rst) on the [Databricks Machine Learning Platform](https://www.databricks.com/product/machine-learning) in just 30 minutes using a single [NDasrA100_v4](https://learn.microsoft.com/en-us/azure/virtual-machines/nda100-v4-series) machine with 8x A100 40GB GPUs. 

Like its base model, dolly-v1-6b has six billion parameters consisting of 28 transformer layers with 16 attention heads each. It employs [Rotary Position Embedding](https://arxiv.org/abs/2104.09864) (RoPE) and shares the same tokenizer as GPT-3. GPT-J was trained on [The Pile](https://huggingface.co/datasets/the_pile), a 400B token dataset of diverse documents designed primarily for text generation tasks.

## Limitations
**`dolly-v1-6b` is intended exclusively for research purposes and is not licensed for commercial use.**

`dolly-v1-6b` is not a state-of-the-art generative language model and, though quantitative benchmarking is ongoing, is not intended to perform competitively with more modern model architectures or models subject to larger pretraining corpuses. For example, we expect the [Alpaca model](https://github.com/tatsu-lab/stanford_alpaca), derived from [LLaMA-7B](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) (trained on 1T tokens vs. The Pile's 400B & with years of scientific advances behind it), to be superior in its generative quality relative to Dolly. What's most notable about Dolly is the degree of its instruction following capabilities given that it's based on a freely available open source model anyone can download and use.

The Dolly model family is under active development, and so any list of shortcomings is unlikely to be exhaustive, but we include known limitations and misfires here as a means to document and share our preliminary findings with the community.  In particular, `dolly-v1-6b` struggles with syntactically complex prompts, mathematical operations, factual errors, dates and times, open-ended question answering, hallucination, enumerating lists of specific length, and stylistic mimicry.  

## Getting Started with Response Generation

If you'd like to simply test the model without training, the model is available on Hugging Face as [databricks/dolly-v1-6b](https://huggingface.co/databricks/dolly-v1-6b).

First, load the model and tokenizer:

```
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)

tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v1-6b", padding_side="left")
model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v1-6b", device_map="auto", trust_remote_code=True)
```

Next, try generating a response:

```
PROMPT_FORMAT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

def generate_response(instruction: str, *, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, 
                      do_sample: bool = True, max_new_tokens: int = 256, top_p: float = 0.92, top_k: int = 0, **kwargs) -> str:
    input_ids = tokenizer(PROMPT_FORMAT.format(instruction=instruction), return_tensors="pt").input_ids.to("cuda")

    # each of these is encoded to a single token
    response_key_token_id = tokenizer.encode("### Response:")[0]
    end_key_token_id = tokenizer.encode("### End")[0]

    gen_tokens = model.generate(input_ids, pad_token_id=tokenizer.pad_token_id, eos_token_id=end_key_token_id,
                                do_sample=do_sample, max_new_tokens=max_new_tokens, top_p=top_p, top_k=top_k, **kwargs)[0].cpu()

    # find where the response begins
    response_positions = np.where(gen_tokens == response_key_token_id)[0]

    if len(response_positions) >= 0:
        response_pos = response_positions[0]
        
        # find where the response ends
        end_pos = None
        end_positions = np.where(gen_tokens == end_key_token_id)[0]
        if len(end_positions) > 0:
            end_pos = end_positions[0]

        return tokenizer.decode(gen_tokens[response_pos + 1 : end_pos]).strip()

    return None

# Sample similar to: "Excited to announce the release of Dolly, a powerful new language model from Databricks! #AI #Databricks"
generate_response("Write a tweet announcing Dolly, a large language model from Databricks.", model=model, tokenizer=tokenizer)
```

## Getting Started with Training

* Add the `dolly` repo to Databricks (under Repos click Add Repo, enter `https://github.com/databrickslabs/dolly.git`, then click Create Repo).
* Start a `12.2 LTS ML (includes Apache Spark 3.3.2, GPU, Scala 2.12)` single-node cluster with node type having 8 A100 GPUs (e.g. `Standard_ND96asr_v4` or `p4d.24xlarge`). Note that these instance types may not be available in all regions, or may be difficult to provision. In Databricks, note that you must select the GPU runtime first, and unselect "Use Photon", for these instance types to appear (where supported).
* Open the `train_dolly` notebook in the Repo (which is the `train_dolly.py` file in the Github `dolly` repo), attach to your GPU cluster, and run all cells.  When training finishes, the notebook will save the model under `/dbfs/dolly_training`.

## Training on Other Instances

A100 instance types are not available in all cloud regions, or can be hard to provision. Training is possible on other GPU instance types, with small modifications to reduce memory usage.
Training will take longer on these instances. These modifications are not necessarily optimal, but are simple to make.

### A10 GPUs

To run on A10 instances (ex: `g5.24xlarge`, 4 x A10 24GB; `Standard_NV72ads_A10_v5`, 2 x A10), make the following changes:

- Modify the deepspeed config file `ds_z3_bf16_config.json` to configure optimizer offload. Within the `"zero_optimization"` section, add:
  ```
  "offload_optimizer": {
    "device": "cpu",
    "pin_memory": true
  },
  ```
- Set the `num_gpus` widget in `train_dolly` to the number of GPUs in your instance, such as 2 or 4, before running

With 4 A10s, an epoch completes in about 7 hours.

### V100 GPUs

To run on V100 instances with 32GB of GPU memory (ex: `p3dn.24xlarge` or `Standard_ND40rs_v2`), make the following changes:

- Modify the deepspeed config to enable optimizer offload, as above
- Modify `trainer.py` to disable `bf16` and enable `fp16` in `TrainingArguments`:
  ```
  ...
  fp16=True,
  bf16=False,
  ...
  ```
- Set the `num_gpus` widget in `train_dolly` to the number of GPUs in your instance, typically 8

With 8 V100s, an epoch completes in about 3.5 hours. Note that the resulting model may be slightly different when trained with `fp16` versus `bf16`.

## Running Unit Tests Locally

```
pyenv local 3.8.13
python -m venv .venv
. .venv/bin/activate
pip install -r requirements_dev.txt
./run_pytest.sh
```
