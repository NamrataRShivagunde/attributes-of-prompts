from dataclasses import dataclass
import  datasets as datasets
import argparse
import torch
from torch.utils import data
from torch.utils.data import DataLoader
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import csv
import pandas as pd
from transformers import DataCollatorWithPadding
import copy
import os

def get_arguments():
    """Set the arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "templatename", 
        default="rte_base",
        help="template name e.g. rte_base"
        )

    parser.add_argument(
        "dataset", 
        help="dataset name e.g. rte"
        )

    parser.add_argument(
        "modelname",
        help="huggingface modelname e.g. facebook/opt-125m"
        )

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu"
        )

    parser.add_argument(
        "--num_shots",
        help="number of shots",
        default=4,
        type=int,
        )

    parser.add_argument(
        "--batch_size",
        help="batch_size",
        default=4,
        type=int,
        )
    # parser.add_argument(
    #     "--random",
    #     action='store_true', # flag if examples must be chosen randomy or not
    #     help="Boolean value suggesting if the in-context exampl should be chosen randomlly or not, True if random",
    #     )

    args = parser.parse_args()
    return args

@dataclass
class NLI():
    def __init__(self, data_temp, data):
        # dataset
        self.data = data

        # template details
        self.task = data_temp['task']
        self.templatename = data_temp['templatename']
        self.instruction = data_temp['instruction']
        self.demo_template = data_temp['demo_template']
        self.query_template = data_temp['query_template']
        self.targets = data_temp['targets']

        # tokenizer
        self.tokenizer = data_temp['tokenizer']

        # prompt
        self.prompt = data_temp['prompt']

        self.class_id_to_label = {}
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ''' for every query, add the prompt (instruction and few shots) and tokenize it
        '''
        final_prompt = self.apply_template(self.query_template, idx) # this is being used for dev set, prompt + filled query = final prompt
        label = self.data[idx]['label']

        # change label id to its word equivalent
        class_id_to_label = self.label_mapping()
        label_word = class_id_to_label[int(label)]

        tok_final_prompt = self.tokenizer(final_prompt)
        # assert len(tok_final_prompt.input_ids) <= 2048 # TODO apply len condition on input

        tok_label_word = self.tokenizer(str(label))


        return {'input_ids':tok_final_prompt.input_ids, # final_prompt input ids
                'attention_mask':tok_final_prompt['attention_mask'],
                'target_input_ids':tok_label_word.input_ids, # label id input ids e.bg. 0 , 1
                'target_attention_mask':tok_label_word['attention_mask']
                }
    
    def apply_template(self, template, idx):
        ''' get the item of index idx and apply template on it. add instruction if it exists
        '''
        
        premise = self.data[int(idx)]['premise']
        hypothesis = self.data[idx]['hypothesis']
        label = self.data[idx]['label']

        # change label id to its word equivalent
        class_id_to_label = self.label_mapping()
        label_word = class_id_to_label[int(label)]
        
        dp = template.replace('{premise}', premise) # dp is datapoint
        dp = dp.replace('{hypothesis}', hypothesis) # filled template

        if '{label}' in dp: # for demostrations only
            dp = dp.replace('{label}', label_word)

        if self.prompt != '':
          dp = self.prompt + "\n" + dp

        return dp

    def label_mapping(self):
        if self.task == 'rte':
            LM_targets = self.targets.split(';')
            self.class_id_to_label = {
                    0: LM_targets[0],  # entailment
                    1: LM_targets[1]}  # non-entailment
        
        return self.class_id_to_label
    


def main():
    # get arguments
    args = get_arguments()
    dataset_name = args.dataset
    num_shots = int(args.num_shots)
    templatename = args.templatename

    # initialize variables
    data_temp = {} # a dictionary which gathers train and dev data, template details, tokenizer
    data_temp['prompt'] = ''
    correct = 0
    

    # load tokenizer and model
    modelname = args.modelname
    # model = AutoModelForCausalLM.from_pretrained(modelname, device_map="auto", load_in_8bit=True).to(device)
    model = AutoModelForCausalLM.from_pretrained(modelname).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(modelname, return_tensors="pt",  device_map="auto", load_in_8bit=True)
    data_temp['tokenizer'] = tokenizer

    file = open('result/rte_base_yes_no/seed1/predictions.txt', 'w')
            
    # # load template TODO read through csv
    # templates = pd.read_csv("templates.xlsx")
    # print(templates)

    with open('templatescsv.csv') as p_file:
        reader = csv.DictReader(p_file)
        for row in reader:
            if row['template_name'] == args.templatename:
                data_temp['task'] = row['task']
                data_temp['templatename'] = row['template_name']
                data_temp['instruction'] = row['instruction']
                data_temp['demo_template'] = row['template-demo']
                data_temp['query_template'] = row['template-query'] # this is same as demo_tempalte without the label placeholder
                data_temp['targets'] = row['target'] # label names

    # get dataset
    train_set = datasets.load_dataset('super_glue', dataset_name, split='train') # to get few shot in-context examples
    dev_set = datasets.load_dataset('super_glue', dataset_name, split='validation') # to evaluate 
    dev_set_len = len(dev_set)
    print(dev_set_len)
    # prepare data 
    nli_train_set = NLI(data_temp, train_set) # self.data in NLI class train set 

    # choose random n integers
    seed=1
    random.seed(seed)
    random_ints =  random.sample(range(0, len(nli_train_set)), num_shots) # from train_set choose n demos randomly
    few_shots = []
    # apply template to demostrations and add it to few_shots list
    for num in random_ints:
        few_shots.append(nli_train_set.apply_template(data_temp['demo_template'], num))  
    few_shot_examples = "\n".join(few_shots)
    # add intruction to the few_shot demos to create a prompt
   
    if data_temp['instruction'] != None:
        prompt = data_temp['instruction'] + "\n" + few_shot_examples 
    else:
        prompt = few_shot_examples


    # this prompt will be added to each query
    data_temp['prompt'] = prompt
    nli_dev_set = NLI(data_temp, dev_set)
   
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dev_dataloader = DataLoader(nli_dev_set, batch_size=args.batch_size, shuffle=False, collate_fn =data_collator)

    # Evaluation loop
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(dev_dataloader):
            # argmax P(y/x)
            # multitoken labels, token prob are multiplied
            target_words = data_temp['targets']
            target_words = target_words.split(';') # list of target words e.g. ['entailment', 'non-entailment'] e.g.[yes, no]

            #output = model(batch["input_ids"], output_norms=True)
            #norm_attentions = output.norm_attentions
            inputs = batch["input_ids"].to(args.device)
            output = model(inputs, output_norms=True)
            torch.save(output.norm_attentions, "norm_attentions")
            # next_pred_word_ids = output.logits[:,-1,:].argmax(dim=-1)

            prob = {}
            # get log prob for first token of all target words e.g. 'ent' and 'non'
            i=0
            for target_word in target_words:
                target_work_tok = tokenizer(target_word)

                first_id = target_work_tok.input_ids[1]  # pick first input if after </s> and get prob of that
                prob[target_words[i]] = output.logits[:,-1,first_id] # logp of 'ent' and 'non'
                i+=1

            # create files and folders
            # if not os.path.exists('result/{}/seed{}/predictions.txt'.format(args.templatename, seed)):
            #     os.mkdir('result/{}/seed{}/predictions.txt'.format(args.templatename, seed))


            # write the P(y/x) prediction - supports single token
           
            for i in range(len(batch.input_ids)):
                label = tokenizer.decode(batch['target_input_ids'][0][1])
                label_mapping = nli_dev_set.label_mapping()
                label = label_mapping[int(label)]

                if prob[target_words[0]][i].item() >= prob[target_words[1]][i].item(): 
                    pred = target_words[0]
                    file.writelines([target_words[0],"\n"])
                else:
                    pred = target_words[1]
                    file.writelines([target_words[1],"\n"])
                if label == pred:
                    correct+=1
            
        print("accuarcy = ", correct/dev_set_len)
        print(correct)
        print(dev_set_len)
        
        torch.save(output.norm_attentions, "norm_attentions")
            # write eaxct match prediction

            # saved norm attentions

            # ori_batch = copy.deepcopy(batch) # original batch

            # for target_word in target_words:
            #     target_work_tok = tokenizer(target_word) # 'ent' 'ail','ment'   
            #     batch = copy.deepcopy(ori_batch)
            #     for i in range(1, len(target_work_tok.input_ids)): # <\s> not incluided and first token not included     
            #         id = target_work_tok.input_ids[i] # it keeps track of tokens to be appended to the exisitng prompt
            #         mask = target_work_tok["attention_mask"][i] # keep starck of attention mask to be appended to exisitng prompt attenton mask
            #         for j in range(len(batch.input_ids)): # loop over the batch and a token is added
            #             print(batch["input_ids"][j] )
            #             print([id])
            #             batch["input_ids"][j] =  batch["input_ids"][j] + [id]
            #             batch["attention_mask"][j] = batch["attention_mask"][j] + [mask]

            #             output = model(batch["input_ids"], output_norms=False)
            #             prob = torch.log(output.logits[:,-1,id])
            #             print(prob)
            #             prob[target_word] = torch.sum(prob[target_words[i]], prob)
                        







            # # for multi-token labels we get logits one by one
            # j=0 # keeps track of item in a batch
            # for target_word in target_words:
            #     target_work_tok = tokenizer(target_word)
            #     for i in range(1, len(target_work_tok.input_ids)): # <\s> not incluided
            #         id = target_work_tok.input_ids[i]  # it keeps track of tokens to be appended to the exisitng prompt
            #         mask = target_work_tok["attention_mask"][i] # keep starck of attention mask to be appended to exisitng prompt attenton mask
            #         batch["input_ids"][j] =  batch["input_ids"][j] + [id]
            #         batch["attention_mask"][j] = batch["attention_mask"][j] + [mask]
                    
            #     output = model(batch["input_ids"], output_norms=False)
            #     withnextid_prob = torch.log(output.logits[:,-1,id])

            #  # probs = torch.sum(torch.log(probs), dim=1)   
            # # generated words accuracy


if __name__=='__main__':
        main()







