from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda"

name = "facebook/opt-13b"
model_8bit = AutoModelForCausalLM.from_pretrained(name, device_map="auto", load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained(name)

text = "Premise === Britain said, Friday, that it has barred cleric, Omar Bakri, from returning to the country from Lebanon, where he was released by police after being detained for 24 hours \n Hypothesis === Bakri was briefly detained, but was released \n  Answer === Entailmen \n The current format presents a 'Premise', 'Hypothesis', and an 'Answer'.  How should I present this to OPT so that it is easy for OPT to answer correctly?"
text2 = "Premise: Britain said, Friday, that it has barred cleric, Omar Bakri, from returning to the country from Lebanon, where he was released by police after being detained for 24 hours \n Hypothesis: Bakri was briefly detained, but was released \n Answer: Entailment. What is meant by entailment and non-entailment task for OPT?"
text3 = "Premise: Britain said, Friday, that it has barred cleric, Omar Bakri, from returning to the country from Lebanon, where he was released by police after being detained for 24 hours \n Hypothesis: Bakri was briefly detained, but was released \nAnswer: Entailment \nPremise: Mangla was summoned after Madhumita's sister Nidhi Shukla, who was the first witness in the case \nHypothesis: Shukla is related to Mangla \nAnswer: Non-Entailment \n Generate description of the given task"
def generate_from_model(model, tokenizer, max_new_tokens):
  encoded_input = tokenizer(text2, return_tensors='pt')
  model = model.to(device)
  output_sequences = model.generate(input_ids=encoded_input['input_ids'].to(device), max_new_tokens=max_new_tokens)
  return tokenizer.decode(output_sequences[0], skip_special_tokens=True)

print(generate_from_model(model_8bit, tokenizer, max_new_tokens = 100))