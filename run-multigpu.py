'''“node” is a system in your distributed architecture. In lay man’s terms, a single system that has multiple GPUs can be called as a node.

“global rank” is a unique identification number for each node in our architecture.

“local rank” is a unique identification number for processes in each node.

“world” is a union of all of the above which can have multiple nodes where each node spawns multiple processes. (Ideally, one for each GPU)

“world_size” is equal to number of nodes * number of gpus
'''


import  datasets as datasets
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import Accelerator,infer_auto_device_map,init_empty_weights,load_checkpoint_and_dispatch

accelerator = Accelerator()


def main():
    modelname= 'facebook/opt-13b'
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = accelerator.device
    config = AutoConfig.from_pretrained(modelname)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    
    device_map = infer_auto_device_map(model, no_split_module_classes=["OPTDecoderLayer"],dtype="float16")

    model = AutoModelForCausalLM.from_pretrained(modelname, device_map=device_map,dtype="float16")
    #model = load_checkpoint_and_dispatch(model, modelname, device_map=device_map)
    
    tokenizer = AutoTokenizer.from_pretrained(modelname, return_tensors="pt")
    
    # data
    dev_set = datasets.load_dataset('super_glue', 'rte', split='validation') # to get few shot in-context examples
    dev_dataloader = DataLoader(dev_set, batch_size=4)

    model, dev_dataloader = accelerator.prepare(model, dev_dataloader)

    with torch.no_grad():
        for i, batch in enumerate(dev_dataloader):
            for j in range(len(batch['premise'])):
                tok_input = tokenizer(batch['premise'][j], padding=True, return_tensors="pt")
                inputs = tok_input['input_ids']
                # output = model(inputs, output_norms=False)
                output = model(inputs)
                print(output.logits.shape)

if __name__=='__main__':
        main()
