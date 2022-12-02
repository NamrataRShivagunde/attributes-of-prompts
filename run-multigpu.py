'''“node” is a system in your distributed architecture. In lay man’s terms, a single system that has multiple GPUs can be called as a node.

“global rank” is a unique identification number for each node in our architecture.

“local rank” is a unique identification number for processes in each node.

“world” is a union of all of the above which can have multiple nodes where each node spawns multiple processes. (Ideally, one for each GPU)

“world_size” is equal to number of nodes * number of gpus
'''


import  datasets as datasets
import torch
from torch.utils import data
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


def main():
    modelname= 'facebook/opt-125m'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(modelname).to(device)
    tokenizer = AutoTokenizer.from_pretrained(modelname, return_tensors="pt")
    
    # data
    dev_set = datasets.load_dataset('super_glue', 'rte', split='validation') # to get few shot in-context examples

    with torch.no_grad():
        for i in range(len(dev_set)):
            if i >= 2:
                break
            tok_input = tokenizer(dev_set[i]['premise'], padding=True, return_tensors="pt")
            inputs = tok_input['input_ids'].to(device)
            # output = model(inputs, output_norms=False)
            output = model(inputs)
            print(output.logits.shape)

if __name__=='__main__':
        main()
