import  datasets as datasets
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import Accelerator,infer_auto_device_map,init_empty_weights,load_checkpoint_and_dispatch

accelerator = Accelerator()


def main():
    modelname= 'facebook/opt-125m'
    model = AutoModelForCausalLM.from_pretrained(modelname,  device_map="auto", load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(modelname, return_tensors="pt")

    text = "HI there"
    model_input = tokenizer(text,  return_tensors="pt").to("cuda")
    print(model_input)
    output = model(**model_input, output_norms=False)
    print(output)

if __name__=='__main__':
        main()