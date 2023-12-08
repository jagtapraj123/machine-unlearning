import torch
import os
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pipelines.classification_pipeline import Pipeline

from utils.constants import SCORE_FUNCTIONS_CLASSIFICATION

from typing import List, Dict


class UnlearnerTeacher(Pipeline):
    def __init__(
        self,
        name: str,
        model: torch.nn.Module,
        trained_teacher_model: torch.nn.Module,
        random_teacher_model: torch.nn.Module,
        batch_size: int,
        workers: int,
        train_set: Dataset,
        test_sets: List[Dict],
        log_files_path: str,
        cuda_num: int = 0,
        KL_temperature: float = 1,
        **kwargs
    ):
        super().__init__(
            name,
            model,
            batch_size,
            workers,
            train_set,
            test_sets,
            log_files_path,
            cuda_num,
            **kwargs
        )

        self.trained_teacher_model = trained_teacher_model
        self.trained_teacher_model.to(self.device)
        self.trained_teacher_model.eval()

        self.random_teacher_model = random_teacher_model
        self.random_teacher_model.to(self.device)
        self.random_teacher_model.eval()

        self.KL_temperature = KL_temperature


    def unlearning_loss(self, y_pred: torch.Tensor, retension_selector: torch.Tensor, forget_selector: torch.Tensor, y_trained: torch.Tensor, y_random: torch.Tensor):
        trained_teacher_out = torch.nn.functional.softmax(y_trained/self.KL_temperature, dim=1)
        random_teacher_out = torch.nn.functional.softmax(y_random/self.KL_temperature, dim=1)
        student_out = torch.nn.functional.softmax(y_pred, dim=1)

        # print(retension_selector.shape, trained_teacher_out.shape, forget_selector.shape, random_teacher_out.shape, flush=True)
        teacher_out = (retension_selector.unsqueeze(1))*trained_teacher_out + (forget_selector.unsqueeze(1))*random_teacher_out

        return torch.nn.functional.kl_div(student_out, teacher_out)


    def train(
        self,
        num_epochs: int = 100,
        lr: float = 0.001,
        # lr2: float = 0.01,
        score_functions: list = SCORE_FUNCTIONS_CLASSIFICATION,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        loss_func=torch.nn.functional.cross_entropy,
        validation_score_epoch: int = 1,
        save_checkpoints_epoch: int = -1,
        save_checkpoints_path: str = "",
        samples_bitvector: np.ndarray = None,
    ):
        self.epochs = num_epochs

        # Setting optimzer
        # param_groups = [
        #     {'params': self.model.base.parameters(),'lr': lr1},
        #     {'params': self.model.final.parameters(),'lr': lr2}
        # ]
        optimizer = optimizer(self.model.parameters(), lr=lr)
        lr_scheduler = lr_scheduler(optimizer)

        training_log = {"errors": [], "scores": []}
        # validation_log = {"errors": [], "scores": []}
        
        # new
        validation_log = {}
        for k in self.test_loaders.keys():
            validation_log[k] = {"errors": [], "scores": []}

        # Training
        # pbar = tqdm(range(self.epochs), desc="Training epoch")
        
        train_size = 50000
        if samples_bitvector is None:
            samples_bitvector = np.ones((train_size), dtype=np.int8)
        else:
            print("Using given samples_bitvector")
        
        samples_bitvector = torch.from_numpy(samples_bitvector).to(self.device)

        for epoch in range(1, self.epochs + 1):
            # print("lr: {}".format(lr_scheduler.get_last_lr()))
            # print("lr1: {}, lr2: {}".format(optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']))
            print("lr: {}".format(optimizer.param_groups[0]['lr']))

            # Putting model in training mode to calculate back gradients
            self.model.train()

            ys = []
            y_preds = []
            batch_losses = []

            # Batch-wise optimization
            pbar = tqdm(self.train_loader, desc="Training epoch {}".format(epoch))
            for x_train, y_train, idx in pbar:
                x = x_train.type(torch.FloatTensor).to(self.device)
                y_truth = y_train.type(torch.LongTensor).to(self.device)

                retension_selector = samples_bitvector[idx] == 1
                forget_selector = samples_bitvector[idx] == 0
                if epoch == 1:
                    print("Using {} samples".format(sum(retension_selector)))
                # x = x[retension_selector]
                # y_truth = y_truth[retension_selector]
                # idx = idx[retension_selector]
                with torch.no_grad():
                    y_trained = self.trained_teacher_model(x)
                    y_random = self.random_teacher_model(x)
                y_pred = self.model(x)

                optimizer.zero_grad()

                loss = self.unlearning_loss(y_pred, retension_selector, forget_selector, y_trained, y_random)
                batch_losses.append(loss.detach())

                loss.backward()
                optimizer.step()

                # Save/show loss per step of training batches
                pbar.set_postfix({"training error": loss.item()})
                training_log["errors"].append({"epoch": epoch, "loss": loss.item()})

                self.train_writer.add_scalar("loss", loss.item(), epoch)
                self.train_writer.flush()

            try:
                lr_scheduler.step()
            except:
                epoch_loss = torch.stack(batch_losses).mean()
                lr_scheduler.step(epoch_loss)

            if epoch == 1 or epoch % validation_score_epoch == 0:

                for test_name, test_loader in self.test_loaders.items():

                    ys = []
                    y_preds = []

                    # Putting model in evaluation mode to stop calculating back gradients
                    self.model.eval()
                    with torch.no_grad():
                        for x_test, y_test, idx in tqdm(
                            test_loader, desc="Validation '{}' epoch {}".format(test_name, epoch)
                        ):
                            x = x_test.type(torch.FloatTensor).to(self.device)
                            y_truth = y_test.type(torch.LongTensor).to(self.device)

                            # Predicting
                            y_pred = self.model(x)

                            # Calculating loss
                            loss = loss_func(y_pred, y_truth)
                            
                            # Save/show loss per batch of validation data
                            # pbar.set_postfix({"test error": loss})
                            validation_log[test_name]["errors"].append(
                                {"epoch": epoch, "loss": loss.item()}
                            )
                            self.valid_writers[test_name].add_scalar("loss", loss.item(), epoch)

                            # Save y_true and y_pred in lists for calculating epoch-wise scores
                            ys += list(y_truth.cpu().detach().numpy())
                            y_preds += list(
                                torch.argmax(y_pred, dim=1).cpu().detach().numpy()
                            )

                    # Save/show validation scores per epoch
                    validation_scores = []
                    if isinstance(score_functions, list) and len(score_functions) > 0:
                        for score_func in score_functions:
                            score = score_func["func"](ys, y_preds)
                            validation_scores.append({score_func["name"]: score})
                            self.valid_writers[test_name].add_scalar(score_func["name"], score, epoch)

                        self.valid_writers[test_name].flush()
                        print(
                            "epoch:{}, Validation '{}' Scores:{}".format(
                                epoch, test_name, validation_scores
                            ),
                            flush=True,
                        )
                        validation_log[test_name]["scores"].append(
                            {"epoch": epoch, "scores": validation_scores}
                        )

            # Saving model at specified checkpoints
            if save_checkpoints_epoch > 0:
                if epoch % save_checkpoints_epoch == 0:
                    chkp_path = os.path.join(
                        save_checkpoints_path,
                        self.name,
                        "checkpoints",
                        "{}".format(epoch),
                    )
                    os.makedirs(chkp_path)
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": loss,
                        },
                        chkp_path + "/model.pth",
                    )
        
        return training_log, validation_log
