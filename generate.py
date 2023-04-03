local_output_dir = "/llm/dolly/dolly_training/checkpoint-2200"

from training.generate import generate_response, load_model_tokenizer_for_generate

model, tokenizer = load_model_tokenizer_for_generate(local_output_dir)

# COMMAND ----------

# Examples from https://www.databricks.com/blog/2023/03/24/hello-dolly-democratizing-magic-chatgpt-open-models.html
instructions = [
#    "Write a love letter to Edgar Allan Poe.",
#    "Write a tweet announcing Dolly, a large language model from Databricks.",
#    "I'm selling my Nikon D-750, write a short blurb for my ad.",
#    "Explain to me the difference between nuclear fission and fusion.",
#    "Give me a list of 5 science fiction books I should read next.",
#     "Write a blog announcement for Michelangelo, Uber's ML platform.",
     'Describe India food item "pani puri"',
]

# Use the model to generate responses for each of the instructions above.
for instruction in instructions:
    response = generate_response(instruction, model=model, tokenizer=tokenizer)
    if response:
        print(f"Instruction: {instruction}\n\n{response}\n\n-----------\n")
