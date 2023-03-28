import logging
import re
from typing import Tuple

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)

# The format of the instruction the model has been trained on.
INTRO = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
INSTRUCTION_FORMAT = """{intro}

### Instruction:
{instruction}

### Response:
"""


def load_model_tokenizer_for_generate(
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
        pretrained_model_name_or_path, device_map="auto", trust_remote_code=True
    )
    return model, tokenizer


def generate_response(
    instruction: str,
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    do_sample: bool = True,
    max_new_tokens: int = 128,
    top_p: float = 0.92,
    top_k: int = 0,
    **kwargs,
) -> str:
    """Given an instruction, uses the model and tokenizer to generate a response.  This formats the instruction in
    the instruction format that the model was fine-tuned on.

    Args:
        instruction (str): instruction to generate response for
        model (PreTrainedModel): model to use
        tokenizer (PreTrainedTokenizer): tokenizer to use
        do_sample (bool, optional): Whether or not to use sampling. Defaults to True.
        max_new_tokens (int, optional): Max new tokens after the prompt to generate. Defaults to 128.
        top_p (float, optional): If set to float < 1, only the smallest set of most probable tokens with probabilities
            that add up to top_p or higher are kept for generation. Defaults to 0.92.
        top_k (int, optional): The number of highest probability vocabulary tokens to keep for top-k-filtering.
            Defaults to 0.

    Returns:
        str: the generated response
    """
    input_ids = tokenizer(
        INSTRUCTION_FORMAT.format(intro=INTRO, instruction=instruction), return_tensors="pt"
    ).input_ids.to("cuda")

    gen_tokens = model.generate(
        input_ids,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        top_k=top_k,
        **kwargs,
    )
    decoded = tokenizer.batch_decode(gen_tokens)[0]

    # The response appears after "### Response:".  The model has been trained to append "### End" at the end.
    m = re.search(r"#+\s*Response:\s*(.+?)#+\s*End", decoded, flags=re.DOTALL)

    response = None
    if m:
        response = m.group(1).strip()
    else:
        # The model might not generate the "### End" sequence before reaching the max tokens.  In this case, return
        # everything after "### Response:".
        m = re.search(r"#+\s*Response:\s*(.+)", decoded, flags=re.DOTALL)
        if m:
            response = m.group(1).strip()
        else:
            logger.warn(f"Failed to find response in:\n{decoded}")

    return response
