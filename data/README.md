# Moved to Hugging Face!

The `databricks-dolly-15k` dataset is now 
[hosted on Hugging Face](https://huggingface.co/datasets/databricks/databricks-dolly-15k).

Please simply use `datasets` to load `databricks/databricks-dolly-15k`.

The data file is directly accessible at 
https://huggingface.co/datasets/databricks/databricks-dolly-15k/blob/main/databricks-dolly-15k.jsonl


# Train the model using local files?

Upload your training data to this folder and select "local_files" as the training_dataset in the "train_dolly" notebook.

Currently, we support JSON, CSV, and Parquet file formats. See more details in https://huggingface.co/docs/datasets/loading#local-and-remote-files