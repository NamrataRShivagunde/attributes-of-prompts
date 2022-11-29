from dataclasses import dataclass
import  datasets as datasets
import argparse
import torch
from torch.utils import data
from torch.utils.data import DataLoader
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import csv
from transformers import DataCollatorWithPadding
from tqdm import tqdm
import numpy as np


def get_arguments():
    """Set the arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "datasetname", 
        help="dataset name e.g. rte",
        type=str,
        )

    parser.add_argument(
        "templatename", 
        default="rte_base",
        help="template name e.g. rte_base",
        type=str,
        )

    parser.add_argument(
        "modelname",
        help="huggingface modelname e.g. facebook/opt-125m",
        type=str,
        )

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu"
        )

    parser.add_argument(
        "--num_shots",
        default=4,
        help="number of shots",
        type=int,
        )

    parser.add_argument(
        "--batch_size",
        default=4,
        help="batch_size",
        type=int,
        )
    
    parser.add_argument(
        "--seed",
        default=1,
        help="seed",
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
    def __init__(self, temp):
        ''' temp (dict) : it contains all column from the template csv file
        '''
        self.targets = temp['targets'] # "yes;no"
        self.task = temp['task'] # "rte"
        self.query_template = temp['query_template']

        if self.task == 'rte':
            LM_targets = self.targets.split(';')
            self.class_id_to_label = {
                    0: LM_targets[0],  # yes
                    1: LM_targets[1]}  # no

        elif self.task == 'snli':
            LM_targets = self.targets.split(';')
            self.class_id_to_label = {
                    0: LM_targets[0],  # yes
                    1: LM_targets[1],  # neutral
                    2: LM_targets[2],  # no
                    -1: LM_targets[3]}  # none
        
        else: # mnli anli
            LM_targets = self.targets.split(';')
            self.class_id_to_label = {
                    0: LM_targets[0],  # yes
                    1: LM_targets[1],  # neutral
                    2: LM_targets[2],  # no
            }
    
    def apply_template(self, example, template):
        ''' get the example and apply template on it. 
            example = {premise:..., hypothesis:..., label: 0}
            template =  "Premise:{premise} Hypothesis:{hypothesis} label:{label}"
        '''
        premise = example['premise'] 
        hypothesis = example['hypothesis']
        label = example['label']

        # change label id to its word equivalent
        label_word = self.class_id_to_label[int(label)]
     
        example_filled = template.replace('{premise}', premise) # dp is datapoint
        example_filled = example_filled.replace('{hypothesis}', hypothesis) # filled template

        if '{label}' in example_filled: # for demostrations only
            example_filled = example_filled.replace('{label}', label_word)

        return example_filled

    def label_mapping(self):
        ''' maps the label word to label id
        '''
        if self.task == 'rte':
            LM_targets = self.targets.split(';')
            self.class_id_to_label = {
                    0: LM_targets[0],  # yes
                    1: LM_targets[1]}  # no

        elif self.task == 'snli':
            LM_targets = self.targets.split(';')
            self.class_id_to_label = {
                    0: LM_targets[0],  # yes
                    1: LM_targets[1],  # neutral
                    2: LM_targets[2],  # no
                    -1: LM_targets[3]}  # none
        
        else: # mnli anli
            LM_targets = self.targets.split(';')
            self.class_id_to_label = {
                    0: LM_targets[0],  # yes
                    1: LM_targets[1],  # neutral
                    2: LM_targets[2],  # no
            }
        
        return self.class_id_to_label
    
    def process_example(self, batch, idx):
        ''' take a query and apply template to it
        '''
       
        premise = batch['premise'][idx]
        hypothesis = batch['hypothesis'][idx]
        label_id = batch['label'][idx]
        label_word = (self.label_mapping())[int(label_id)]
        print(label_word)
        
        example = {'premise':premise,   # query
                    'hypothesis': hypothesis,
                    'label': label_id}

        filled_example =   self.apply_template(example, self.query_template)             # single filled query
        
        return filled_example, label_word

def main():
    
    # get arguments
    args = get_arguments()

    # load tokenizer and model
    modelname = args.modelname
    model = AutoModelForCausalLM.from_pretrained(modelname,  device_map="auto", load_in_8bit=True).to(args.device)
    # model = AutoModelForCausalLM.from_pretrained(modelname).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(modelname, return_tensors="pt")

    # get dataset
    if args.datasetname == "rte":
        train_set = datasets.load_dataset('super_glue', args.datasetname, split='train') # to get few shot in-context examples
        dev_set = datasets.load_dataset('super_glue', args.datasetname, split='validation') # to evaluate 
    elif args.datasetname == "snli":
        train_set = datasets.load_dataset(args.datasetname, split='train') # to get few shot in-context examples
        dev_set = datasets.load_dataset(args.datasetname, split='validation') # to evaluate 
    elif args.datasetname == "mnli":
        train_set = datasets.load_dataset('glue', args.datasetname, split='train') # to get few shot in-context examples
        dev_set = datasets.load_dataset('glue', args.datasetname, split='validation_matched') # to evaluate 
    elif args.datasetname == "anli":
        train_set = datasets.load_dataset(args.datasetname, split='train_r1') # to get few shot in-context examples
        dev_set = datasets.load_dataset(args.datasetname, split='dev_r1') # to evaluate

    # get template
    temp = {}
    with open('templatescsv.csv') as p_file:
        reader = csv.DictReader(p_file)
        for row in reader:
            if row['template_name'] == args.templatename:
                temp['task'] = row['task']
                temp['templatename'] = row['template_name']
                temp['instruction'] = row['instruction']
                temp['demo_template'] = row['template-demo']
                temp['query_template'] = row['template-query'] # this is same as demo_tempalte without the label placeholder
                temp['targets'] = row['targets'] # label names
    
    # initialize class
    if args.datasetname == "rte" or "snli" or "mnli" or "anli":
        data_cat = NLI(temp)

    # create prompt (instrcutions + in-context exmaples with templates)
    # choose random n integers
    seed=args.seed
    random.seed(seed)
    random_ints =  random.sample(range(0, len(train_set)), args.num_shots) # from train_set choose n demos randomly
    
    few_shots = []
    print(temp)
    # apply template to demostrations and add it to few_shots list
    for num in random_ints:
        filled_example = data_cat.apply_template(train_set[num],  temp['demo_template'])
        few_shots.append(filled_example)  
    
    if temp['instruction'] != '':
        prompt = temp['instruction'] + "\n" + "\n".join(few_shots)
    else:
        prompt = "\n".join(few_shots)

    # iterate over val set and apply tempalte and add prompt to each query
    dev_dataloader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False)

    target_words = temp['targets'].split(';')
    target_ids = []
    true_labels = []
    all_predictions = []
    all_next_word_predictions = []
    target_encoded = tokenizer(target_words) # {'input_ids': [[2, 10932], [2, 12516], [2, 2362], [2, 39763]], 'attention_mask': [[1, 1], [1, 1], [1, 1], [1, 1]]} 4 for snli
    for i in range(len(target_words)):
        target_ids.append(target_encoded['input_ids'][i][1])  # [10932,  12516, 2362, 39763] for snli

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dev_dataloader)):
            proc_batch = []
            batch_predictions = []
            batch_next_word_predictions = []
            for i in range(len(batch['premise'])):
                
                filled_example, label_word = data_cat.process_example(batch, i) # takes the ith query in a batch and add in-context examples and filles the template
                if prompt != '':
                    filled_example = prompt + "\n" + filled_example # add prompt too if it exists
            
                proc_batch.append(filled_example)
                
                true_labels.append(label_word) # will be used to compute accuracy

            tok_input = tokenizer(proc_batch, return_tensors="pt", padding=True)
            inputs = tok_input['input_ids'].to(args.device)
            # output = model(inputs, output_norms=False)
            output = model(inputs)

            # logits gather using torch.gather()
            logits = ((output.logits)[:,-1,:]).unsqueeze(1).to("cpu") # [b, 1, vocab] taking last set of logits

            # next word prediction
            for j in range(len(batch['premise'])):
                batch_next_word_predictions.append(tokenizer.decode(logits[j,-1,:].argmax(dim=0)))
    

            # P(y/x) where y are labels
            indices = torch.ones(logits.shape[0], 1, len(target_words)) # [b, 1, len(targetwords)]
            indices = indices.type(torch.int64)
            indices[:,-1,:] = torch.tensor(target_ids) 
            choice_id = torch.gather(logits, 2, indices)
            choice_id = choice_id.argmax(dim=2)[:,-1] # [b, 1]
            for id in choice_id:
                batch_predictions.append(target_words[id])   
          
            all_predictions.extend(batch_predictions)      
            all_next_word_predictions.extend(batch_next_word_predictions)

        accuracy =  (np.array(all_predictions) == np.array(true_labels)).mean()
        print("Accuracy for ", args.templatename, accuracy)

        accuracy_nextword =  (np.array(all_next_word_predictions) == np.array(true_labels)).mean()
        print("Accuracy for correct next word prediction ", args.templatename, accuracy_nextword)

if __name__=='__main__':
        main()

