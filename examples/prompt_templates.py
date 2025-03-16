# Databricks notebook source
# MAGIC %md
# MAGIC ## Prompt Template Example
# MAGIC
# MAGIC This example demonstrates how to use Dolly with predefined prompt templates for consistent responses.
# MAGIC It shows a straightforward way to create reusable prompt structures for common tasks.

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

from training.generate import load_model_tokenizer_for_generate, generate_response

input_model = dbutils.widgets.get("input_model")

model, tokenizer = load_model_tokenizer_for_generate(input_model)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Prompt Templates
# MAGIC
# MAGIC Creating reusable prompt templates helps maintain consistency in how you interact with the model.

# COMMAND ----------

class PromptTemplate:
    """A simple class to manage prompt templates for Dolly."""
    
    def __init__(self, template):
        """Initialize with a template string that has {placeholders}."""
        self.template = template
    
    def format(self, **kwargs):
        """Fill in the template with provided values."""
        return self.template.format(**kwargs)

# Define some useful templates
SUMMARIZE_TEMPLATE = PromptTemplate(
    "Summarize the following text in 3-5 sentences:\n\n{text}"
)

EXPLAIN_TEMPLATE = PromptTemplate(
    "Explain {concept} in simple terms that a high school student would understand."
)

COMPARE_TEMPLATE = PromptTemplate(
    "Compare and contrast {thing1} and {thing2}. Include at least 3 similarities and 3 differences."
)

LIST_TEMPLATE = PromptTemplate(
    "List {number} {category} that {criteria}."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Examples of Using Templates

# COMMAND ----------

# Example 1: Text Summarization
sample_text = """
The James Webb Space Telescope (JWST) is a space telescope designed primarily to conduct infrared astronomy. 
As the largest optical telescope in space, its high infrared resolution and sensitivity allow it to view objects 
too old, distant, or faint for the Hubble Space Telescope. This will enable investigations across many fields 
of astronomy and cosmology, such as observation of the first stars and the formation of the first galaxies, 
and detailed atmospheric characterization of potentially habitable exoplanets.
"""

summary_prompt = SUMMARIZE_TEMPLATE.format(text=sample_text)
print("Summary Prompt:", summary_prompt)
print("\nResponse:")
summary_response = generate_response(summary_prompt, model=model, tokenizer=tokenizer)
print(summary_response)
print("-" * 80)

# COMMAND ----------

# Example 2: Concept Explanation
concept_prompt = EXPLAIN_TEMPLATE.format(concept="quantum computing")
print("Explanation Prompt:", concept_prompt)
print("\nResponse:")
explanation_response = generate_response(concept_prompt, model=model, tokenizer=tokenizer)
print(explanation_response)
print("-" * 80)

# COMMAND ----------

# Example 3: Comparison
compare_prompt = COMPARE_TEMPLATE.format(thing1="renewable energy", thing2="fossil fuels")
print("Comparison Prompt:", compare_prompt)
print("\nResponse:")
comparison_response = generate_response(compare_prompt, model=model, tokenizer=tokenizer)
print(comparison_response)
print("-" * 80)

# COMMAND ----------

# Example 4: List Generation
list_prompt = LIST_TEMPLATE.format(number=5, category="books", criteria="are science fiction classics")
print("List Prompt:", list_prompt)
print("\nResponse:")
list_response = generate_response(list_prompt, model=model, tokenizer=tokenizer)
print(list_response)
print("-" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creating a Custom Template for Your Domain
# MAGIC
# MAGIC You can easily create templates for specific domains or use cases.

# COMMAND ----------

# Example: Custom template for a product description generator
PRODUCT_TEMPLATE = PromptTemplate(
    """Create a compelling product description for an e-commerce website with the following details:
    
    Product Name: {name}
    Category: {category}
    Key Features: {features}
    Target Audience: {audience}
    
    The description should be enthusiastic but factual, and approximately 100-150 words.
    """
)

product_prompt = PRODUCT_TEMPLATE.format(
    name="SolarBrew 3000",
    category="Kitchen Appliances",
    features="solar-powered coffee maker, temperature control, programmable timer, insulated carafe",
    audience="eco-conscious coffee enthusiasts"
)

print("Product Prompt:", product_prompt)
print("\nResponse:")
product_response = generate_response(product_prompt, model=model, tokenizer=tokenizer)
print(product_response)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC This example demonstrates how to use prompt templates with Dolly to create consistent, 
# MAGIC structured interactions with the model. This pattern can be extended to various applications 
# MAGIC and integrated into larger workflows.
