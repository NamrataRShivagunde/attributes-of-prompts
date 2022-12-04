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
    print(device_map)

if __name__=='__main__':
        main()