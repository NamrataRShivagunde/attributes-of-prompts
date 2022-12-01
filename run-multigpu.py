import  datasets as datasets
import torch
from torch.utils import data
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP

BATCH_SIZE_PER_DEVICE = 1
NUM_GPUS = 2
DATASETNAME = "rte"
MODELNAME = "facebook/opt-125m"
MAX_BATCH = 3

def main():

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")

    # load model and tokenizer
    device_id = rank % torch.cuda.device_count()
    model = AutoModelForCausalLM.from_pretrained(MODELNAME,  device_map="auto", load_in_8bit=True).to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])

    tokenizer = AutoTokenizer.from_pretrained(MODELNAME, return_tensors="pt")

    # get dataset
    train_set = datasets.load_dataset('super_glue', DATASETNAME, split='train') # to get few shot in-context examples

    # get dataloader
    ddp_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE_PER_DEVICE, sampler=ddp_sampler)

    # evaluation loop

if __name__=='__main__':
        main()
