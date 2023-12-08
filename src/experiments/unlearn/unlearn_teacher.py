import sys
sys.path.append("../../")

import os
import torch

from utils.data_mappers import CustomCIFAR100, transform_train
from models.model import ResNet18
from utils.constants import SCORE_FUNCTIONS_CLASSIFICATION, SCORE_FUNCTIONS_REGRESSION

from unlearner_methods.random_teacher import UnlearnerTeacher

import argparse
import pickle
from functools import partial
from torchsummary import summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="train_resnet18_scratch", description="Training ResNet18 from scratch")
    parser.add_argument("model_name", type=str, help='Model Name (without spaces)')
    parser.add_argument("--subset_csv_path", type=str, default=None, help='Path to subset csv')
    parser.add_argument("save_models", type=str, help='Path to directory to save models')
    parser.add_argument("trained_teacher_path", type=str, help='Path to saved trained teacher model')
    parser.add_argument("--cuda-num", dest="cuda_num", type=int, help="Device number for cuda")
    parser.add_argument("--num-workers", dest="num_workers", type=int, help="Number of workers for dataloader")
    parser.add_argument("--coarse_labels", dest="coarse_labels", action="store_true", help="Use coarse class labels (20) instead of 100")
    parser.add_argument("bitvector_path", type=str, help="Path to retention data samples bitvector")

    args = parser.parse_args()

    model_name = args.model_name
    subset_file = args.subset_csv_path
    save_dir = args.save_models
    trained_teacher_path = args.trained_teacher_path
    cuda_num = args.cuda_num
    num_workers = args.num_workers
    coarse_labels = args.coarse_labels
    bitvector_path = args.bitvector_path
    
    print("Name: {}".format(model_name))
    print("Subset File: {}".format(subset_file))
    print("Save Dir: {}".format(save_dir))
    print("Trained Teacher Path: {}".format(trained_teacher_path))
    print("Cuda Num: {}".format(cuda_num), flush=True)
    print("Coarse Labels: {}".format(coarse_labels), flush=True)
    print("Samples Bitvector Path: {}".format(bitvector_path), flush=True)
    
    # Model
    model = ResNet18(20 if coarse_labels else 100, pretrained=True)
    print(model)
    summary(model, (3, 32, 32), device='cpu')

    model.load_state_dict(torch.load(trained_teacher_path)["model_state_dict"])

    # Trained Teacher
    trained_teacher_model = ResNet18(20 if coarse_labels else 100, pretrained=False)
    print(trained_teacher_model)
    summary(trained_teacher_model, (3, 32, 32), device="cpu")

    trained_teacher_model.load_state_dict(torch.load(trained_teacher_path)["model_state_dict"])

    for params in trained_teacher_model.parameters():
        params.requires_grad = False

    trained_teacher_model.eval()

    # Random Teacher
    random_teacher_model = ResNet18(20 if coarse_labels else 100, pretrained=False)
    print(random_teacher_model)
    summary(random_teacher_model, (3, 32, 32), device="cpu")

    for params in random_teacher_model.parameters():
        params.requires_grad = False
    
    random_teacher_model.eval()


    train_set = CustomCIFAR100(root=".", train=True, download=True, transform=transform_train, coarse_labels=coarse_labels)
    test_set = CustomCIFAR100(root=".", train=False, download=True, transform=transform_train, coarse_labels=coarse_labels)

    score_sets = [
        {"name": "full_train_set", "dataset": train_set}, 
        # *score_sets_low_mem,
        # *score_sets_high_mem,
        {"name": "test_set", "dataset": test_set},
    ]

    print(score_sets)

    
    with open(bitvector_path, "rb") as f:
        samples_bitvector = pickle.load(f)
        print("Bitvector loaded from {}".format(bitvector_path))

    trainer = UnlearnerTeacher(
        name=model_name,
        model=model,
        trained_teacher_model=trained_teacher_model,
        random_teacher_model=random_teacher_model,
        batch_size=256,
        workers=num_workers,
        train_set=train_set,
        test_sets=score_sets,
        log_files_path="logs/fit/",
        cuda_num=cuda_num,
        KL_temperature=1,
        pin_memory=True
    )

    num_epochs = 5
    lr = 0.0001

    loss_func = torch.nn.functional.cross_entropy

    training_log, validation_log = trainer.train(
        num_epochs=num_epochs,
        lr=lr,
        lr_scheduler=partial(torch.optim.lr_scheduler.ReduceLROnPlateau, mode='min', factor=0.5, patience=3, verbose=True),
        loss_func=loss_func,
        validation_score_epoch=1,
        samples_bitvector = samples_bitvector, 
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
