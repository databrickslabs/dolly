# Databricks notebook source
# MAGIC %md
# MAGIC ## Train Dolly
# MAGIC
# MAGIC This fine-tunes the [GPT-J 6B](https://huggingface.co/EleutherAI/gpt-j-6B) model on
# MAGIC the [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) dataset.
# MAGIC
# MAGIC ```
# MAGIC   Licensed under the Apache License, Version 2.0 (the "License");
# MAGIC   you may not use this file except in compliance with the License.
# MAGIC   You may obtain a copy of the License at
# MAGIC
# MAGIC       http://www.apache.org/licenses/LICENSE-2.0
# MAGIC
# MAGIC   Unless required by applicable law or agreed to in writing, software
# MAGIC   distributed under the License is distributed on an "AS IS" BASIS,
# MAGIC   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# MAGIC   See the License for the specific language governing permissions and
# MAGIC   limitations under the License.
# MAGIC ```
# MAGIC
# MAGIC Please note that while GPT-J 6B is [Apache 2.0 licensed](https://huggingface.co/EleutherAI/gpt-j-6B),
# MAGIC the Alpaca dataset is licensed under [Creative Commons NonCommercial (CC BY-NC 4.0)](https://huggingface.co/datasets/tatsu-lab/alpaca).

# COMMAND ----------

# MAGIC !wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcusparse-dev-11-3_11.5.0.58-1_amd64.deb -O /tmp/libcusparse-dev-11-3_11.5.0.58-1_amd64.deb && \
# MAGIC   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcublas-dev-11-3_11.5.1.109-1_amd64.deb -O /tmp/libcublas-dev-11-3_11.5.1.109-1_amd64.deb && \
# MAGIC   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcusolver-dev-11-3_11.1.2.109-1_amd64.deb -O /tmp/libcusolver-dev-11-3_11.1.2.109-1_amd64.deb && \
# MAGIC   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcurand-dev-11-3_10.2.4.109-1_amd64.deb -O /tmp/libcurand-dev-11-3_10.2.4.109-1_amd64.deb && \
# MAGIC   dpkg -i /tmp/libcusparse-dev-11-3_11.5.0.58-1_amd64.deb && \
# MAGIC   dpkg -i /tmp/libcublas-dev-11-3_11.5.1.109-1_amd64.deb && \
# MAGIC   dpkg -i /tmp/libcusolver-dev-11-3_11.1.2.109-1_amd64.deb && \
# MAGIC   dpkg -i /tmp/libcurand-dev-11-3_10.2.4.109-1_amd64.deb

# COMMAND ----------

# MAGIC %pip install -r requirements.txt

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)
logging.getLogger("py4j").setLevel(logging.WARNING)
logging.getLogger("sh.command").setLevel(logging.ERROR)

# COMMAND ----------

import os
from datetime import datetime
from dolly.trainer import load_training_dataset, load_tokenizer

dbutils.widgets.text("num_gpus", "", "num_gpus")

# COMMAND ----------

# Cache data and tokenizer locally before creating a bunch of deepspeed processes and make sure they succeeds.
load_training_dataset()
load_tokenizer()

# COMMAND ----------

timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
model_name = "dolly"
checkpoint_dir_name = f"{model_name}__{timestamp}"

root_path = os.getcwd()
deepspeed_config = os.path.join(root_path, "config/ds_z3_bf16_config.json")

local_training_root = os.path.join(os.path.expanduser('~'), "dolly_training")

os.makedirs(local_training_root, exist_ok=True)

local_output_dir = os.path.join(local_training_root, checkpoint_dir_name)
dbfs_output_dir = os.path.join("/dbfs/dolly_training", checkpoint_dir_name)

num_gpus_flag = ""
num_gpus = dbutils.widgets.get("num_gpus")
if num_gpus:
    num_gpus = int(num_gpus)
    num_gpus_flag = f"--num_gpus={num_gpus}"

tensorboard_display_dir = f"{local_output_dir}/runs"

print(f"Local Output Dir: {local_output_dir}")
print(f"DBFS Output Dir: {dbfs_output_dir}")
print(f"Tensorboard Display Dir: {tensorboard_display_dir}")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# COMMAND ----------

# MAGIC %load_ext tensorboard
# MAGIC %tensorboard --logdir '{tensorboard_display_dir}'

# COMMAND ----------

# MAGIC !deepspeed {num_gpus_flag} \
# MAGIC     --module dolly.trainer \
# MAGIC     --deepspeed {deepspeed_config} \
# MAGIC     --epochs 1 \
# MAGIC     --local-output-dir {local_output_dir} \
# MAGIC     --dbfs-output-dir {dbfs_output_dir} \
# MAGIC     --per-device-train-batch-size 8 \
# MAGIC     --per-device-eval-batch-size 8 \
# MAGIC     --lr 1e-5

# COMMAND ----------
