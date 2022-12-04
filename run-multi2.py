'''“node” is a system in your distributed architecture. In lay man’s terms, a single system that has multiple GPUs can be called as a node.

“global rank” is a unique identification number for each node in our architecture.

“local rank” is a unique identification number for processes in each node.

“world” is a union of all of the above which can have multiple nodes where each node spawns multiple processes. (Ideally, one for each GPU)

“world_size” is equal to number of nodes * number of gpus
'''


from logging import logger
import  datasets as datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.distributed as dist
from transformers import (Trainer, 
    HfArgumentParser, 
    TrainingArguments,
    EvalPrediction
)



def main():
    parser = HfArgumentParser((TrainingArguments))
    training_args = parser.parse_args_into_dataclasses()

    modelname= 'facebook/opt-125m'
    model = AutoModelForCausalLM.from_pretrained(modelname)
    tokenizer = AutoTokenizer.from_pretrained(modelname, return_tensors="pt")

    eval_dataset = datasets.load_dataset('super_glue', 'rte', split='validation') # to get few shot in-context examples


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=eval_dataset,
        compute_metrics=None,
        tokenizer=tokenizer,
        data_collator=None,
    )

    trainer.predict(eval_dataset=eval_dataset)
    
    # with torch.no_grad():
    #     for i, batch in enumerate(dev_dataloader):
    #         if i >= 2:
    #             break
    #         for j in range(len(batch['premise'])):
    #             tok_input = tokenizer(batch['premise'][j], padding=True, return_tensors="pt")
    #             inputs = tok_input['input_ids'].to(device)
    #             # output = model(inputs, output_norms=False)
    #             output = model(inputs)
    #             print(output.logits.shape)

if __name__=='__main__':
        main()
