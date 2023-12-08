import sys
sys.path.append("../../")

import os
import torch

from torch.utils.data import DataLoader

from utils.data_mappers import CustomCIFAR100, transform_train
from models.model import ResNet18
from utils.constants import SCORE_FUNCTIONS_CLASSIFICATION, SCORE_FUNCTIONS_REGRESSION

from unlearner_methods.unsir import UNSIR, IMPAIR, REPAIR

import argparse
import pickle
from functools import partial
from torchsummary import summary
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="train_resnet18_scratch", description="Training ResNet18 from scratch")
    parser.add_argument("model_name", type=str, help='Model Name (without spaces)')
    parser.add_argument("test_name", type=str, help='Test Name (without spaces)')
    parser.add_argument("--subset_csv_path", type=str, default=None, help='Path to subset csv')
    parser.add_argument("trained_model_path", type=str, help='Path to saved trained teacher model')
    parser.add_argument("--cuda-num", dest="cuda_num", type=int, help="Device number for cuda")
    parser.add_argument("--num-workers", dest="num_workers", type=int, help="Number of workers for dataloader")
    parser.add_argument("--coarse_labels", dest="coarse_labels", action="store_true", help="Use coarse class labels (20) instead of 100")
    parser.add_argument("bitvector_path", type=str, help="Path to retention data samples bitvector")

    args = parser.parse_args()

    model_name = args.model_name
    test_name = args.test_name
    subset_file = args.subset_csv_path
    trained_model_path = args.trained_model_path
    cuda_num = args.cuda_num
    num_workers = args.num_workers
    coarse_labels = args.coarse_labels
    bitvector_path = args.bitvector_path
    
    print("Name: {}".format(model_name))
    print("Subset File: {}".format(subset_file))
    print("Trained Teacher Path: {}".format(trained_model_path))
    print("Cuda Num: {}".format(cuda_num), flush=True)
    print("Coarse Labels: {}".format(coarse_labels), flush=True)
    print("Samples Bitvector Path: {}".format(bitvector_path), flush=True)
    
    device = torch.device("cuda:{}".format(cuda_num) if torch.cuda.is_available() else "cpu")

    # Model
    model = ResNet18(20 if coarse_labels else 100, pretrained=True)
    # print(model)
    # summary(model, (3, 32, 32), device='cpu')

    model.load_state_dict(torch.load(trained_model_path, map_location='cpu')["model_state_dict"])

    model.to(device)
    model.eval()

    train_set = CustomCIFAR100(root=".", train=True, download=True, transform=transform_train, coarse_labels=coarse_labels)
    test_set = CustomCIFAR100(root=".", train=False, download=True, transform=transform_train, coarse_labels=coarse_labels)

    score_sets = [
        {"name": "full_train_set", "dataset": train_set}, 
        {"name": "test_set", "dataset": test_set},
    ]

    with open(bitvector_path, "rb") as f:
        samples_bitvector = pickle.load(f)
        print("Bitvector loaded from {}".format(bitvector_path))

    for score_set in score_sets:
        test_loader = DataLoader(
                score_set["dataset"], batch_size=512, num_workers=num_workers
            )
        
        ys = []
        y_preds = []
        with torch.no_grad():
            print(score_set["name"] == "full_train_set")
            used = 0
            for x_test, y_test, idx in tqdm(test_loader):
                x = x_test.type(torch.FloatTensor).to(device)
                y_truth = y_test.type(torch.LongTensor).to(device)

                if score_set["name"] == "full_train_set":
                    retension_selector = samples_bitvector[idx] == 1
                    forget_selector = samples_bitvector[idx] == 0
                    x = x[forget_selector]
                    y_truth = y_truth[forget_selector]
                    used += x.shape[0]
                else:
                    retension_selector = y_truth != 0
                    forget_selector = y_truth == 0
                    x = x[forget_selector]
                    y_truth = y_truth[forget_selector]
                    used += x.shape[0]

                y_pred = model(x)

                # loss = loss_func(y_pred, y_truth)

                ys += list(y_truth.cpu().detach().numpy())

                y_preds += list(torch.argmax(y_pred, dim=1).cpu().detach().numpy())

        print("Used: {}".format(used))
        score_functions=SCORE_FUNCTIONS_CLASSIFICATION
        validation_scores = []
        if isinstance(score_functions, list) and len(score_functions) > 0:
            for score_func in score_functions:
                score = score_func["func"](ys, y_preds)
                validation_scores.append({score_func["name"]: score})

            print(
                "Testing {}, Set:{}, Validation Scores:{}".format(
                    test_name, score_set["name"], validation_scores
                ),
                flush=True,
            )
