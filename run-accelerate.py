from accelerate import init_empty_weights, infer_auto_device_map
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch

modelname = "facebook/opt-30b"
text = '''Premise === Britain said, Friday, that it has barred cleric, Omar Bakri, from returning to the country from Lebanon, where he was released by police after being detained for 24 hours
Hypothesis === Bakri was briefly detained, but was released
Answer === Entailment
The current format presents a 'Premise', 'Hypothesis', and an 'Answer'.  How should I present this to OPT so that it is easy for OPT to answer correctly? '''

# text = "How do you define entailment and non-entailement task? What is Premise and Hypothesis?"

max_new_tokens = 300

def generate_from_model(model, tokenizer, max_new_tokens):
  encoded_input = tokenizer(text, return_tensors='pt')
  output_sequences = model.generate(input_ids=encoded_input['input_ids'].cuda(), max_new_tokens=max_new_tokens)
  return tokenizer.decode(output_sequences[0], skip_special_tokens=True)

# config = AutoConfig.from_pretrained(modelname)
# with init_empty_weights():
#     model = AutoModelForCausalLM.from_config(config)

device_map = {
 'model.decoder.embed_tokens': 0,
 'model.decoder.embed_positions': 0,
 'model.decoder.final_layer_norm': 0,
 'model.decoder.layers.0': 0,
 'model.decoder.layers.1': 0,
 'model.decoder.layers.2': 0,
 'model.decoder.layers.3': 0,
 'model.decoder.layers.4': 0,
 'model.decoder.layers.5': 0,
 'model.decoder.layers.6': 0,
 'model.decoder.layers.7': 0,
 'model.decoder.layers.8': 0,
 'model.decoder.layers.9': 0,
 'model.decoder.layers.10': 0,
 'model.decoder.layers.11': 0,
 'model.decoder.layers.12': 0,
 'model.decoder.layers.13': 0,
 'model.decoder.layers.14': 0,
 'model.decoder.layers.15': 0,
 'model.decoder.layers.16': 0,
 'model.decoder.layers.17': 0,
 'model.decoder.layers.18': 0,
 'model.decoder.layers.19': 0,
 'model.decoder.layers.20': 0,
 'model.decoder.layers.21': 0,
 'model.decoder.layers.22': 0,
 'model.decoder.layers.23': 0,
 'model.decoder.layers.24': 1,
 'model.decoder.layers.25': 1,
 'model.decoder.layers.26': 1,
 'model.decoder.layers.27': 1,
 'model.decoder.layers.28': 1,
 'model.decoder.layers.29': 1,
 'model.decoder.layers.30': 1,
 'model.decoder.layers.31': 1,
 'model.decoder.layers.32': 1,
 'model.decoder.layers.33': 1,
 'model.decoder.layers.34': 1,
 'model.decoder.layers.35': 1,
 'model.decoder.layers.36': 1,
 'model.decoder.layers.37': 1,
 'model.decoder.layers.38': 1,
 'model.decoder.layers.39': 1,
 'model.decoder.layers.40': 1,
 'model.decoder.layers.41': 1,
 'model.decoder.layers.42': 1,
 'model.decoder.layers.43': 1,
 'model.decoder.layers.44': 1,
 'model.decoder.layers.45': 1,
 'model.decoder.layers.46': 1,
 'model.decoder.layers.47': 1,
 'lm_head': 1,
}


model_8bit = AutoModelForCausalLM.from_pretrained(modelname, device_map=device_map, load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained(modelname)

print(generate_from_model(model_8bit, tokenizer, max_new_tokens))

#print(model_8bit.hf_device_map)