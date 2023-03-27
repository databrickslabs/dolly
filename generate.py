from transformers import AutoModelForCausalLM, AutoTokenizer
path = '/path/to/model/checkpoint/directory'
model_name = path
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

import torch
device = torch.device('cuda:0')
model.to(device)

def generate_text(model, tokenizer, input_text, device):
    with torch.no_grad():
        input_tokens = tokenizer.encode(input_text, return_tensors="pt").to(device)
        output_tokens = model.generate(
            input_tokens,
            max_length=300,  # Controls the maximum length of the generated text
            num_return_sequences=1,  # Controls the number of sentences to generate
            no_repeat_ngram_size=2,  # Controls the repetition of n-grams in the generated text
            temperature=1.0,  # Controls the randomness of the generated text
            top_k=50,  # Controls the diversity of the generated text
            top_p=0.95,  # Controls the nucleus sampling parameter
        )
        output_tokens = output_tokens.cpu()
        generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        return generated_text

input_text = '''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
'''.format(
    instruction='Please write a draft tweet to announce the birth of a chatbot named Dolly. Please note for its special name "Dolly".',
    input='Dolly is fine-tuned from GPT-J 6B, using Alpaca dataset and 8 A100 GPU cards. The name "Dolly" is chosen to honor the first cloned sheep.')
print(generate_text(model, tokenizer, input_text, device))
