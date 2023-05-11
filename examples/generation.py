# Databricks notebook source
# MAGIC %md
# MAGIC ## Generation Example
# MAGIC
# MAGIC This takes a pretrained Dolly model, either from Hugging face or from a local path, and runs generation with it
# MAGIC using the code from this repo.
# MAGIC
# MAGIC The model to load for generation is controlled by `input_model`.  The default options are the pretrained
# MAGIC Dolly models shared on Hugging Face.  Alternatively, the path to a local model that has been trained using the
# MAGIC `train_dolly` notebook can also be used.

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC
# MAGIC default_model = "databricks/dolly-v2-12b"
# MAGIC
# MAGIC suggested_models = [
# MAGIC     "databricks/dolly-v1-6b",
# MAGIC     "databricks/dolly-v2-3b",
# MAGIC     "databricks/dolly-v2-7b",
# MAGIC     "databricks/dolly-v2-12b",
# MAGIC ]
# MAGIC
# MAGIC dbutils.widgets.combobox("input_model", default_model, suggested_models, "input_model")

# COMMAND ----------

import logging
import re
from typing import List, Tuple
import torch

import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Pipeline,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from transformers.utils import is_tf_available

if is_tf_available():
    import tensorflow as tf



    

def load_model_tokenizer_for_generate_denson(
    pretrained_model_name_or_path: str,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Loads the model and tokenizer so that it can be used for generating responses.
    Args:
        pretrained_model_name_or_path (str): name or path for model
    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: model and tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True,
        load_in_8bit=True
    )
    return model, tokenizer

# COMMAND ----------

from training.generate import InstructionTextGenerationPipeline, load_model_tokenizer_for_generate

# input_model = dbutils.widgets.get("input_model")

input_model = "databricks/dolly-v2-12b"

model, tokenizer = load_model_tokenizer_for_generate_denson(input_model)

# COMMAND ----------

# Examples from https://www.databricks.com/blog/2023/03/24/hello-dolly-democratizing-magic-chatgpt-open-models.html
instructions = [
    "Explain to me the difference between nuclear fission and fusion.",
    "Give me a list of 5 science fiction books I should read next.",
]

# Use the model to generate responses for each of the instructions above.
for instruction in instructions:
    response = generate_response(instruction, model=model, tokenizer=tokenizer)
    if response:
        print(f"Instruction: {instruction}\n\n{response}\n\n-----------\n")
