import sys
sys.path.append("../../")

import os
import torch

from utils.data_mappers import CustomCIFAR100, transform_train
from models.model import ResNet18
# from unlearner_methods.
from pipelines.classification_pipeline import Pipeline
from utils.constants import SCORE_FUNCTIONS_CLASSIFICATION, SCORE_FUNCTIONS_REGRESSION

import argparse
import pickle
from functools import partial
from torchsummary import summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="train_resnet18_scratch", description="Training ResNet18 from scratch")
    parser.add_argument("model_name", type=str, help='Model Name (without spaces)')
    parser.add_argument("--subset_csv_path", type=str, default=None, help='Path to subset csv')
    parser.add_argument("save_models", type=str, help='Path to directory to save models')
    parser.add_argument("--cuda-num", dest="cuda_num", type=int, help="Device number for cuda")
    parser.add_argument("--num-workers", dest="num_workers", type=int, help="Number of workers for dataloader")
    parser.add_argument("--coarse_labels", dest="coarse_labels", action="store_true", help="Use coarse class labels (20) instead of 100")
    parser.add_argument("--bitvector_path", dest="bitvector_path", default=None, type=str, help="Path to saved S1 model's samples bitvector")

    args = parser.parse_args()

    model_name = args.model_name
    subset_file = args.subset_csv_path
    save_dir = args.save_models
    cuda_num = args.cuda_num
    num_workers = args.num_workers
    coarse_labels = args.coarse_labels
    bitvector_path = args.bitvector_path
    
    print("Name: {}".format(model_name))
    print("Subset File: {}".format(subset_file))
    print("Save Dir: {}".format(save_dir))
    print("Cuda Num: {}".format(cuda_num), flush=True)
    print("Coarse Labels: {}".format(coarse_labels), flush=True)
    print("Samples Bitvector Path: {}".format(bitvector_path), flush=True)
    
    
    model = ResNet18(20 if coarse_labels else 100, pretrained=True)
    print(model)
    summary(model, (3, 32, 32), device='cpu')

    train_set = CustomCIFAR100(root=".", train=True, download=True, transform=transform_train, coarse_labels=coarse_labels)
    test_set = CustomCIFAR100(root=".", train=False, download=True, transform=transform_train, coarse_labels=coarse_labels)

    score_sets = [
        {"name": "full_train_set", "dataset": train_set}, 
        # *score_sets_low_mem,
        # *score_sets_high_mem,
        {"name": "test_set", "dataset": test_set},
    ]

    print(score_sets)

    if bitvector_path is not None:
        with open(bitvector_path, "rb") as f:
            samples_bitvector = pickle.load(f)
            print("Bitvector loaded from {}".format(bitvector_path))


    trainer = Pipeline(
        name=model_name,
        model=model,
        batch_size=128,
        workers=num_workers,
        train_set=train_set,
        test_sets=score_sets,
        log_files_path="logs/fit/",
        cuda_num=cuda_num,
        # pin_memory=True
    )

    num_epochs = 5
    lr1 = 0.001
    lr2 = 0.01

    loss_func_with_grad = torch.nn.CrossEntropyLoss() # torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
    loss_func = torch.nn.functional.cross_entropy # partial(torch.nn.functional.kl_div, reduction='batchmean', log_target=True)
    
    training_log, validation_log, _, _ = trainer.train(
        num_epochs=num_epochs,
        lr1=lr1,
        lr2=lr2,
        lr_scheduler=partial(torch.optim.lr_scheduler.ReduceLROnPlateau, mode='min', factor=0.5, patience=3, verbose=True),
        loss_func_with_grad=loss_func_with_grad,
        loss_func=loss_func,
        score_functions=SCORE_FUNCTIONS_CLASSIFICATION,
        samples_bitvector=samples_bitvector if bitvector_path is not None else None,
        validation_score_epoch=1,
        # save_checkpoints_epoch=25,
        # save_checkpoints_path="../../saved_models/",
    )

    trainer.save(save_dir, num_epochs)

    os.makedirs(os.path.join("logs/logfiles", trainer.name))
    with open(
        os.path.join("logs/logfiles", trainer.name, "training_log.pkl"), "wb"
    ) as f:
        pickle.dump(training_log, f)

    with open(
        os.path.join("logs/logfiles", trainer.name, "validation_log.pkl"), "wb"
    ) as f:
        pickle.dump(validation_log, f)
