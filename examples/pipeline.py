# Databricks notebook source
# MAGIC %md
# MAGIC ## Pipeline Example
# MAGIC
# MAGIC This takes a pretrained Dolly model, either from Hugging face or from a local path, and uses the pipeline from
# MAGIC this repo to perform generation.
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

from training.generate import InstructionTextGenerationPipeline, load_model_tokenizer_for_generate

input_model = dbutils.widgets.get("input_model")

model, tokenizer = load_model_tokenizer_for_generate(input_model)

# COMMAND ----------

generation_pipeline = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)

# Examples from https://www.databricks.com/blog/2023/03/24/hello-dolly-democratizing-magic-chatgpt-open-models.html
instructions = [
    "Explain to me the difference between nuclear fission and fusion.",
    "Give me a list of 5 science fiction books I should read next.",
]

# Use the model to generate responses for each of the instructions above.
for instruction in instructions:
    results = generation_pipeline(instruction, num_return_sequences=2)

    print(f"Instruction: {instruction}\n")
    for i, res in enumerate(results, 1):
        text = res["generated_text"]
        print(f"Sample #{i}:\n{text}\n")
    print("-----------\n")
