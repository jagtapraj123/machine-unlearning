import numpy as np
import argparse
import random
import pickle
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="forget random", description="Generates a bitvector to simulate forget data")
    parser.add_argument("forget_size", type=float, help='Fraction of data to forget')
    parser.add_argument("bitvector_path", type=str, help='Path to save retension bitvector')

    args = parser.parse_args()

    forget_size = args.forget_size
    bitvector_path = args.bitvector_path

    print("Forgetting: {}% data".format(forget_size*100))

    train_size = 50000
    retension_bitvector = np.ones((train_size), dtype=np.int8)
    forget_idx = random.sample(range(train_size), k=int(train_size*forget_size))
    retension_bitvector[forget_idx] = 0

    with open("{}.pkl".format(bitvector_path), "wb") as f:
            pickle.dump(retension_bitvector, f)
