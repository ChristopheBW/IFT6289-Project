# IFT6289-Project

## Model GGUF

We uploaded the fine-tuned model and generated dialogues dataset to huggingface in gguf file, you can deploy them and evaluate directly without wasting time, electricity and money for API requests.

https://huggingface.co/ChristopheBW/IFT6289

## Environment

- **python version**: 3.12.3
- **venv requirements**: requirements.txt

## Instruction to run the reproduce

The code is well structured, the file name indicate it's function clearly so you can choose the part you want to reproduce. For example: if you want to reproduce WVS seed data, you just need to run: `1_data_preprocessing/generate_dialogues.py` **(but you need to setup your own api key! store it at `apikey.txt`)**

If you want to reproduce the responses csv file with ollama (download model from hugging face, import it using Modefile), we prepared scripts at `3_evaluation/generate_response_*`, just remember to change the url of the ollama server to your machine.

Here is a Modefile example for llama3.2-culture:

```modefile
# The path of the gguf
FROM ./llama32-culture.gguf

# Set the prompt template format for Llama 3 Instruct
# Reference: https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_2/
TEMPLATE """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|><|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

# Set a default system prompt (optional, can be overridden)
SYSTEM """You are a helpful AI assistant fine-tuned to understand and reflect nuances relevant to cultural contexts. Respond naturally and considerately within this context."""

# Set model parameters (optional examples)
PARAMETER temperature 0.7
#ARAMETER context_length 4096 # Adjust based on model and your needs (Llama 3.2 supports up to 128k, but check GGUF limitations/performance)
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|end_header_id|>"

# License (Optional but good practice - copy from base model if needed)
# LICENSE """<License details here>"""
```
