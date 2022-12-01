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

BATCH_SIZE_PER_DEVICE = 1
NUM_GPUS = 2
DATASETNAME = "rte"
MODELNAME = "facebook/opt-125m"
MAX_BATCH = 3

def main():
    print("STEP1")
    dist.init_process_group("nccl", rank=0, world_size=2) # nccl backend for GPU, master node = 0 , world size = total number of gpus (2), rank = range from 0 to k-1
    dev0 = (rank * 2) % world_size
    dev1 = (rank * 2 + 1) % world_size

    print("STEP2")
    # load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(MODELNAME,  device_map="auto", load_in_8bit=True).to(dev0, dev1)
    ddp_model = DDP(model, device_ids=[device_id])

    print("STEP3")
    tokenizer = AutoTokenizer.from_pretrained(MODELNAME, return_tensors="pt")

    print("STEP4")
    # get dataset
    train_set = datasets.load_dataset('super_glue', DATASETNAME, split='train') # to get few shot in-context examples

    print("STEP5")
    # get dataloader
    ddp_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE_PER_DEVICE, sampler=ddp_sampler)

    print("STEP6")
    # evaluation loop
    for i, batch in enumerate(dataloader):
        print(batch)

if __name__=='__main__':
        main()
