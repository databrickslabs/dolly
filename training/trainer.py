# Copyright 2023 Databricks, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import json
from functools import partial
from typing import Any, Dict, List, Tuple, Union

import click
import numpy as np
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .consts import (
    DEFAULT_INPUT_MODEL,
    DEFAULT_SEED,
    DEFAULT_TRAINING_DATASET,
    END_KEY,
    INSTRUCTION_KEY,
    RESPONSE_KEY_NL,
)

logger = logging.getLogger(__name__)

PROMPT_FORMAT = """{instruction}

### Questions:
{input_text}

### Response:{output_text}
"""

def create_data_set_from_json_list(json_list_file="/home/bo_ling/dolly/ma_data/so3_long.jsonl", 
                                   max_question_length:int=500, max_answer_length:int=500):
    """
    Tokens are important to understand because GPT-J, like other language models, have a maximum context length of 2048 tokens, or roughly 1500 words.
    """
    def gen():
        with open(json_list_file) as file:
            while True:
                line = file.readline()

                # if line is empty
                # end of file is reached
                if not line:
                    break 
                try:
                    data = json.loads(line)

                except:
                    # print(f"BAD DATA: {line}")
                    data = json.loads(line.replace("\\", ""))

                instruction = "Answer the following MA helpdesk questions:"
                input_text = data["prompt"][:max_question_length]
                output_text = "\n" + data["completion"][:max_answer_length]
                text = PROMPT_FORMAT.format(instruction=instruction,input_text=input_text,output_text=output_text)

                yield{
                    "instruction": instruction,
                    "input": input_text, 
                    "output": output_text,
                    "text": text
                }
    dataset = Dataset.from_generator(gen)
    return dataset



class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        # The prompt ends with the response key plus a newline.  We encode this and then try to find it in the
        # sequence of tokens.  This should just be a single token.
        response_token_ids = self.tokenizer.encode(RESPONSE_KEY_NL)

        labels = batch["labels"].clone()

        for i in range(len(examples)):

            response_token_ids_start_idx = None
            for idx in np.where(batch["labels"][i] == response_token_ids[0])[0]:
                response_token_ids_start_idx = idx
                break

            if response_token_ids_start_idx is None:
                print(f"========== {examples[i]}; {labels[i]}; {response_token_ids[0]}==========")
                raise RuntimeError(
                    f'Could not find response key {response_token_ids} in token IDs {batch["labels"][i]}'
                )

            response_token_ids_end_idx = response_token_ids_start_idx + 1

            # Make pytorch loss function ignore all tokens up through the end of the response key
            labels[i, :response_token_ids_end_idx] = -100

        batch["labels"] = labels

        return batch


def preprocess_batch(batch: Dict[str, List], tokenizer: AutoTokenizer, max_length: int) -> dict:
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )


def load_training_dataset(training_data_id: str = DEFAULT_TRAINING_DATASET, split: str = "train", 
                         local_data_file_path: str="") -> Dataset:
    logger.info(f"Loading {training_data_id} dataset")
    if local_data_file_path: 
        print(f"===============loading local dataset from file: {local_data_file_path}=====================")
        dataset: Dataset = create_data_set_from_json_list(local_data_file_path)
    else:
        print(f"===============loading public dataset: {training_data_id}=====================")
        dataset: Dataset = load_dataset(training_data_id)[split]
    logger.info("Found %d rows", dataset.num_rows)

    # Remove empty responses
    response_key_stripped = RESPONSE_KEY_NL.strip()
    dataset = dataset.filter(lambda rec: not rec["text"].strip().endswith(response_key_stripped))

    def _func(rec):
        rec["text"] += f"\n\n{END_KEY}"
        return rec

    dataset = dataset.map(_func)

    return dataset


def load_tokenizer(pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL) -> PreTrainedTokenizer:
    logger.info(f"Loading tokenizer for {pretrained_model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [END_KEY, INSTRUCTION_KEY, RESPONSE_KEY_NL]})
    return tokenizer


def load_model(
    pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL, *, gradient_checkpointing: bool = False
) -> AutoModelForCausalLM:
    logger.info(f"Loading model for {pretrained_model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path, trust_remote_code=True, use_cache=False if gradient_checkpointing else True
    )
    return model


def get_model_tokenizer(
    pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL, *, gradient_checkpointing: bool = False
) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    tokenizer = load_tokenizer(pretrained_model_name_or_path)
    model = load_model(pretrained_model_name_or_path, gradient_checkpointing=gradient_checkpointing)
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed=DEFAULT_SEED, 
                      local_data_file_path: str = "") -> Dataset:
    """Loads the training dataset and tokenizes it so it is ready for training.

    Args:
        tokenizer (AutoTokenizer): Tokenizer tied to the model.
        max_length (int): Maximum number of tokens to emit from tokenizer.

    Returns:
        Dataset: HuggingFace dataset
    """

    dataset = load_training_dataset(local_data_file_path=local_data_file_path)

    logger.info("Preprocessing dataset")
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["instruction", "input", "output", "text"],
    )

    logger.info("Shuffling dataset")
    dataset = dataset.shuffle(seed=seed)

    logger.info("Done preprocessing")

    return dataset


def train(
    local_output_dir,
    dbfs_output_dir,
    epochs,
    per_device_train_batch_size,
    per_device_eval_batch_size,
    lr,
    seed,
    deepspeed,
    gradient_checkpointing,
    local_rank,
    bf16,
    local_data_file_path="",
    test_size=1000,
):
    set_seed(seed)

    model, tokenizer = get_model_tokenizer(gradient_checkpointing=gradient_checkpointing)

    # Use the same max length that the model supports.  Try a couple different keys in case a different
    # model is used.  The default model uses n_positions.  If no config settings can be found just default
    # to 1024 as this is probably supported by most models.
    conf = model.config
    default_length = 2048
    max_length: int = getattr(conf, "n_positions", getattr(conf, "seq_lenth", default_length))

    processed_dataset = preprocess_dataset(tokenizer=tokenizer, max_length=max_length, seed=seed,
                                           local_data_file_path=local_data_file_path)

    split_dataset = processed_dataset.train_test_split(test_size=test_size, seed=seed)

    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer, mlm=False, return_tensors="pt", pad_to_multiple_of=8
    )

    if not dbfs_output_dir:
        logger.warn("Will NOT save to DBFS")

    training_args = TrainingArguments(
        output_dir=local_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        fp16=False,
        bf16=bf16,
        learning_rate=lr,
        num_train_epochs=epochs,
        deepspeed=deepspeed,
        gradient_checkpointing=gradient_checkpointing,
        logging_dir=f"{local_output_dir}/runs",
        logging_strategy="steps",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to="tensorboard",
        disable_tqdm=True,
        remove_unused_columns=False,
        local_rank=local_rank,
    )

    logger.info("Instantiating Trainer")

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        data_collator=data_collator,
    )

    logger.info("Training")
    trainer.train()

    logger.info(f"Saving Model to {local_output_dir}")
    trainer.save_model(output_dir=local_output_dir)

    if dbfs_output_dir:
        logger.info(f"Saving Model to {dbfs_output_dir}")
        trainer.save_model(output_dir=dbfs_output_dir)

    logger.info("Done.")


@click.command()
@click.option("--local-output-dir", type=str, help="Write directly to this local path", required=True)
@click.option("--dbfs-output-dir", type=str, help="Sync data to this path on DBFS")
@click.option("--epochs", type=int, default=3, help="Number of epochs to train for.")
@click.option("--per-device-train-batch-size", type=int, default=8, help="Batch size to use for training.")
@click.option("--per-device-eval-batch-size", type=int, default=8, help="Batch size to use for evaluation.")
@click.option("--lr", type=float, default=1e-5, help="Learning rate to use for training.")
@click.option("--seed", type=int, default=DEFAULT_SEED, help="Seed to use for training.")
@click.option("--deepspeed", type=str, default=None, help="Path to deepspeed config file.")
@click.option("--local-data-file-path", type=str, default="", help="""The local training data with list of json with `prompt` and `completion` as the key""")
@click.option("--test-size", type=int, default=1000, help="Path to deepspeed config file.")
@click.option(
    "--gradient-checkpointing/--no-gradient-checkpointing",
    is_flag=True,
    default=True,
    help="Use gradient checkpointing?",
)
@click.option(
    "--local_rank",
    type=str,
    default=True,
    help="Provided by deepspeed to identify which instance this process is when performing multi-GPU training.",
)
@click.option("--bf16", type=bool, default=True, help="Whether to use bf16 (preferred on A100's).")
def main(**kwargs):
    train(**kwargs)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )
    try:
        main()
    except Exception:
        logger.exception("main failed")
        raise
