# Dolly

Databricks’ [Dolly](https://huggingface.co/databricks/dolly-v2-12b) is an instruction-following large language model trained on the Databricks machine learning platform
that is licensed for commercial use. Based on `pythia-12b`, Dolly is trained on ~15k instruction/response fine tuning records
[`databricks-dolly-15k`](https://github.com/databrickslabs/dolly/tree/master/data) generated
by Databricks employees in capability domains from the InstructGPT paper, including brainstorming, classification, closed QA, generation,
information extraction, open QA and summarization. `dolly-v2-12b` is not a state-of-the-art model, but does exhibit surprisingly
high quality instruction following behavior not characteristic of the foundation model on which it is based.

Databricks is committed to ensuring that every organization and individual benefits from the transformative power of artificial intelligence. The Dolly model family represents our first steps along this journey, and we’re excited to share this technology with the world.

The model is available on Hugging Face as [databricks/dolly-v2-12b](https://huggingface.co/databricks/dolly-v2-12b).

## Model Overview

`dolly-v2-12b` is a 12 billion parameter causal language model created by [Databricks](https://databricks.com/) that is derived from
[EleutherAI’s](https://www.eleuther.ai/) [Pythia-12b](https://huggingface.co/EleutherAI/pythia-12b) and fine-tuned
on a [~15K record instruction corpus](https://github.com/databrickslabs/dolly/tree/master/data) generated by Databricks employees and released under a permissive license (CC-BY-SA)


## Known Limitations

### Performance Limitations
**`dolly-v2-12b` is not a state-of-the-art generative language model** and, though quantitative benchmarking is ongoing, is not designed to perform
competitively with more modern model architectures or models subject to larger pretraining corpuses.

The Dolly model family is under active development, and so any list of shortcomings is unlikely to be exhaustive, but we include known limitations and misfires here as a means to document and share our preliminary findings with the community.
In particular, `dolly-v2-12b` struggles with: syntactically complex prompts, programming problems, mathematical operations, factual errors,
dates and times, open-ended question answering, hallucination, enumerating lists of specific length, stylistic mimicry, having a sense of humor, etc.
Moreover, we find that `dolly-v2-12b` does not have some capabilities, such as well-formatted letter writing, present in the original model.

### Dataset Limitations
Like all language models, `dolly-v2-12b` reflects the content and limitations of its training corpuses.

- **The Pile**: GPT-J’s pre-training corpus contains content mostly collected from the public internet, and like most web-scale datasets,
it contains content many users would find objectionable. As such, the model is likely to reflect these shortcomings, potentially overtly
in the case it is explicitly asked to produce objectionable content, and sometimes subtly, as in the case of biased or harmful implicit
associations.

- **`databricks-dolly-15k`**: The training data on which `dolly-v2-12b` is instruction tuned represents natural language instructions generated
by Databricks employees during a period spanning March and April 2023 and includes passages from Wikipedia as references passages
for instruction categories like closed QA and summarization. To our knowledge it does not contain obscenity, intellectual property or
personally identifying information about non-public figures, but it may contain typos and factual errors.
The dataset may also reflect biases found in Wikipedia. Finally, the dataset likely reflects
the interests and semantic choices of Databricks employees, a demographic which is not representative of the global population at large.

Databricks is committed to ongoing research and development efforts to develop helpful, honest and harmless AI technologies that
maximize the potential of all individuals and organizations.

## Getting Started with Response Generation

If you'd like to simply test the model without training, the model is available on Hugging Face as [databricks/dolly-v2-12b](https://huggingface.co/databricks/dolly-v2-12b).

To use the model with the `transformers` library on a machine with GPUs:

```
from transformers import pipeline

instruct_pipeline = pipeline(model="databricks/dolly-v2-12b", trust_remote_code=True, device_map="auto")
```

You can then use the pipeline to answer instructions:

```
instruct_pipeline("Explain to me the difference between nuclear fission and fusion.")
```

To reduce memory usage you can load the model with `bfloat16`:

```
import torch
from transformers import pipeline

instruct_pipeline = pipeline(model="databricks/dolly-v2-12b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
```

## Getting Started with Training

The following instructions refer to Dolly v1 and still need to be updated for v2 training.

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
