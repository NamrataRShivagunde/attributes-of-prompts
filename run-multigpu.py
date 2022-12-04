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

    model = AutoModelForCausalLM.from_pretrained(modelname, device_map=device_map, load_in_8bit=True)
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
