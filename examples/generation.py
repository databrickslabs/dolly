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

default_model = "databricks/dolly-v2-3b"

suggested_models = [
    "databricks/dolly-v1-6b",
    "databricks/dolly-v2-3b",
    "databricks/dolly-v2-7b",
    "databricks/dolly-v2-12b",
]

dbutils.widgets.combobox("input_model", default_model, suggested_models, "input_model")

# COMMAND ----------

from training.generate import generate_response, load_model_tokenizer_for_generate

input_model = dbutils.widgets.get("input_model")

model, tokenizer = load_model_tokenizer_for_generate(input_model)

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
