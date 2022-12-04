from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

modelname = "facebook/opt-30b"
text = "Hello my name is"
max_new_tokens = 20

def generate_from_model(model, tokenizer):
  encoded_input = tokenizer(text, return_tensors='pt')
  output_sequences = model.generate(input_ids=encoded_input['input_ids'].cuda())
  return tokenizer.decode(output_sequences[0], skip_special_tokens=True)


model_8bit = AutoModelForCausalLM.from_pretrained(modelname, device_map="auto", load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained(modelname)

generate_from_model(model_8bit, tokenizer)

print(model_8bit.hf_device_map)