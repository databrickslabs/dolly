import logging
from typing import Tuple

import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from .consts import END_KEY, PROMPT_FORMAT, RESPONSE_KEY_NL

logger = logging.getLogger(__name__)


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


def get_special_token_id(tokenizer: PreTrainedTokenizer, key: str) -> int:
    """Gets the token ID for a given string that has been added to the tokenizer as a special token.

    When training, we configure the tokenizer so that the sequences like "### Instruction:" and "### End" are
    treated specially and converted to a single, new token.  This retrieves the token ID each of these keys map to.

    Args:
        tokenizer (PreTrainedTokenizer): the tokenizer
        key (str): the key to convert to a single token

    Raises:
        RuntimeError: if more than one ID was generated

    Returns:
        int: the token ID for the given key
    """
    token_ids = tokenizer.encode(key)
    if len(token_ids) > 1:
        raise RuntimeError(f"Expected only a single token for '{key}' but found {token_ids}")
    return token_ids[0]


def generate_response(
    instruction: str,
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    do_sample: bool = True,
    max_new_tokens: int = 256,
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
    input_ids = tokenizer(PROMPT_FORMAT.format(instruction=instruction), return_tensors="pt").input_ids.to("cuda")

    response_key_token_id = get_special_token_id(tokenizer, RESPONSE_KEY_NL)
    end_key_token_id = get_special_token_id(tokenizer, END_KEY)

    gen_tokens = model.generate(
        input_ids,
        pad_token_id=tokenizer.pad_token_id,
        # Ensure generation stops once it generates "### End"
        eos_token_id=end_key_token_id,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        top_k=top_k,
        **kwargs,
    )[0].cpu()

    # The response will be set to this variable if we can identify it.
    decoded = None

    # Find where "### Response:" is first found in the generated tokens.  Considering this is part of the prompt,
    # we should definitely find it.  We will return the tokens found after this token.
    response_pos = None
    response_positions = np.where(gen_tokens == response_key_token_id)[0]
    if len(response_positions) == 0:
        logger.warn(f"Could not find response key {response_key_token_id} in: {gen_tokens}")
    else:
        response_pos = response_positions[0]

    if response_pos:
        # Next find where "### End" is located.  The model has been trained to end its responses with this sequence
        # (or actually, the token ID it maps to, since it is a special token).  We may not find this token, as the
        # response could be truncated.  If we don't find it then just return everything to the end.  Note that
        # even though we set eos_token_id, we still see the this token at the end.
        end_pos = None
        end_positions = np.where(gen_tokens == end_key_token_id)[0]
        if len(end_positions) > 0:
            end_pos = end_positions[0]

        decoded = tokenizer.decode(gen_tokens[response_pos + 1 : end_pos]).strip()

    return decoded
