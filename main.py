#from  datasets import dataset
import argparse
import torch


def get_arguments():
    """Set the arguments"""
    parser = argparse.ArgumentParser()

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

    args = parser.parse_args()
    return args


def main():
    # get arguments
    args = get_arguments()
    dataset_name = args.dataset

    # get dataset


    # create prompts
        # create a data class to load the data
        # create a dataloader to create batches

    # load model


    # evaluate model on the validation set of the given data

    
    # compute accuracy for each data for each model


    # save the norm attention weights of each example for each model


    # Need a jupyter notebook which can view the norm attention weight for any validtion datapoint


    print(dataset_name)
    


if __name__=='__main__':
    main()







