# Dolly

This fine-tunes the [GPT-J 6B](https://huggingface.co/EleutherAI/gpt-j-6B) model on the [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) dataset using a Databricks notebook.  Please note that while GPT-J 6B is [Apache 2.0 licensed](https://huggingface.co/EleutherAI/gpt-j-6B), the Alpaca dataset is licensed under [Creative Commons NonCommercial (CC BY-NC 4.0)](https://huggingface.co/datasets/tatsu-lab/alpaca).

## Get Started Training

* Add the `dolly` repo to Databricks (under Repos click Add Repo, enter `https://github.com/databrickslabs/dolly.git`, then click Create Repo).
* Start a `12.2 LTS ML (includes Apache Spark 3.3.2, GPU, Scala 2.12)` single-node cluster with node type having 8 A100 GPUs (e.g. `Standard_ND96asr_v4` or `p4d.24xlarge`).
* Open the `train_dolly` notebook in the `dolly` repo, attach to your GPU cluster, and run all cells.  When training finishes, the notebook will save the model under `/dbfs/dolly_training`.

## Running Unit Tests Locally

```
pyenv local 3.8.13
python -m venv .venv
. .venv/bin/activate
pip install -r requirements_dev.txt
./run_pytest.sh
```

### Train the model (shell command only)

* Start a single-node cluster with node type having 8 A100 (40GB memory) GPUs (e.g. `Standard_ND96asr_v4` or `p4d.24xlarge`).

```bash
export timestamp=`date +%Y-%m-%d_%H-%M-%S`
export model_name='dolly'
export checkpoint_dir_name="${model_name}__${timestamp}"
export deepspeed_config=`pwd`/config/ds_z3_bf16_config.json
export local_training_root='./'
export local_output_dir="${local_training_root}/${checkpoint_dir_name}"
export dbfs_output_dir=''
export tensorboard_display_dir="${local_output_dir}/runs"
export DATASET_FILE_PATH=`pwd`/parquet-train.arrow
export MODEL_PATH=`pwd`/model/
deepspeed --num_gpus=8 \
    --module training.trainer \
    --deepspeed $deepspeed_config \
    --epochs 1 \
    --local-output-dir $local_output_dir \
    --dbfs-output-dir "" \
    --per-device-train-batch-size 8 \
    --per-device-eval-batch-size 8 \
    --lr 1e-5
```

Note: You may want to set CUDA_HOME and PATH so that deepspeed can find `nvcc` to compile operators in a Just-In-Time manner.

## Generate some sentences

```
python generate.py
```

(It is recommended to use `ipython` to interactively generate sentences to avoid loading models from disk again and again.)
