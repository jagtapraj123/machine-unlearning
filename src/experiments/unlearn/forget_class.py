import sys
sys.path.append("../../")
import numpy as np
import argparse
import random
import pickle
import os

from tqdm import tqdm

from utils.data_mappers import CustomCIFAR100, transform_train

from torch.utils.data import DataLoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="forget class", description="Generates a bitvector to simulate forget data")
    parser.add_argument("class_id", type=float, help='Class ID')
    parser.add_argument("bitvector_path", type=str, help='Path to save retension bitvector')

    args = parser.parse_args()

    class_id = args.class_id
    bitvector_path = args.bitvector_path

    print("Forgetting: {} class".format(class_id))

    train_set = CustomCIFAR100(root=".", train=True, download=True, transform=transform_train, coarse_labels=True)
    
    retension_bitvector = np.ones((len(train_set)), dtype=np.int8)

    for i in tqdm(range(len(train_set))):
        x, y, idx = train_set[i]
        if y == class_id:
            retension_bitvector[idx] = 0
        
    with open("{}.pkl".format(bitvector_path), "wb") as f:
            pickle.dump(retension_bitvector, f)
