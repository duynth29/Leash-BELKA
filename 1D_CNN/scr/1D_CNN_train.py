import gc
import os
import pickle
import random
import joblib
import pandas as pd

# import polars as pd
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor

import mlflow.pytorch  # tracking
from datetime import datetime


from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)

current_date = datetime.now()
date_string = current_date.strftime("%Y%m%d_%H%M")

# Create a directory to save the results
dir = f"../result/{date_string}/"
os.makedirs(dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Specify tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")


class CFG:

    PREPROCESS = False
    EPOCHS = 51  # 20
    BATCH_SIZE = 2048
    LR = 1e-3
    WD = 1e-6
    NBR_FOLDS = 5
    SELECTED_FOLDS = [0, 1, 2, 3, 4]

    SEED = 2024


if CFG.PREPROCESS:
    enc = {
        "l": 1,
        "y": 2,
        "@": 3,
        "3": 4,
        "H": 5,
        "S": 6,
        "F": 7,
        "C": 8,
        "r": 9,
        "s": 10,
        "/": 11,
        "c": 12,
        "o": 13,
        "+": 14,
        "I": 15,
        "5": 16,
        "(": 17,
        "2": 18,
        ")": 19,
        "9": 20,
        "i": 21,
        "#": 22,
        "6": 23,
        "8": 24,
        "4": 25,
        "=": 26,
        "1": 27,
        "O": 28,
        "[": 29,
        "D": 30,
        "B": 31,
        "]": 32,
        "N": 33,
        "7": 34,
        "n": 35,
        "-": 36,
    }
    train_raw = pd.read_parquet("../../data/train.parquet")
    smiles = train_raw[train_raw["protein_name"] == "BRD4"]["molecule_smiles"].values
    assert (
        smiles
        != train_raw[train_raw["protein_name"] == "HSA"]["molecule_smiles"].values
    ).sum() == 0
    assert (
        smiles
        != train_raw[train_raw["protein_name"] == "sEH"]["molecule_smiles"].values
    ).sum() == 0

    def encode_smile(smile):
        tmp = [enc[i] for i in smile]
        tmp = tmp + [0] * (142 - len(tmp))
        return np.array(tmp).astype(np.uint8)

    smiles_enc = joblib.Parallel(n_jobs=96)(
        joblib.delayed(encode_smile)(smile) for smile in tqdm(smiles)
    )
    smiles_enc = np.stack(smiles_enc)
    train = pd.DataFrame(smiles_enc, columns=[f"enc{i}" for i in range(142)])
    train["bind1"] = train_raw[train_raw["protein_name"] == "BRD4"]["binds"].values
    train["bind2"] = train_raw[train_raw["protein_name"] == "HSA"]["binds"].values
    train["bind3"] = train_raw[train_raw["protein_name"] == "sEH"]["binds"].values
    train.to_parquet("../../data/train_enc.parquet")

    # test_raw = pd.read_parquet("../../data/test.parquet")
    # smiles = test_raw["molecule_smiles"].values

    # smiles_enc = joblib.Parallel(n_jobs=96)(
    #     joblib.delayed(encode_smile)(smile) for smile in tqdm(smiles)
    # )
    # smiles_enc = np.stack(smiles_enc)
    # test = pd.DataFrame(smiles_enc, columns=[f"enc{i}" for i in range(142)])
    # test.to_parquet("test_enc.parquet")

else:
    train = pd.read_parquet("../../data/train_enc.parquet")

    # test = pd.read_parquet("../../data/test_enc.parquet")
print("Data loaded")


class CNN_1D(torch.nn.Module):
    def __init__(
        self,
        input_dim=142,
        input_dim_embedding=37,
        hidden_dim=256,
        num_filters=64,
        output_dim=3,
        lr=1e-3,
        weight_decay=1e-6,
    ):
        super(CNN_1D, self).__init__()

        # Initialize class variables
        self.input_dim_embedding = input_dim_embedding
        self.hidden_dim = hidden_dim
        self.num_filters = num_filters
        self.output_dim = output_dim
        self.lr = lr
        self.weight_decay = weight_decay

        self.embedding = nn.Embedding(
            num_embeddings=self.input_dim_embedding,
            embedding_dim=self.hidden_dim,
            padding_idx=0,
        )
        self.conv1 = nn.Conv1d(
            in_channels=self.hidden_dim,
            out_channels=self.num_filters,
            kernel_size=4,
            stride=1,
            padding=0,
            dilation=2,
        )
        self.conv2 = nn.Conv1d(
            in_channels=self.num_filters,
            out_channels=self.num_filters * 2,
            kernel_size=4,
            stride=1,
            padding=0,
            dilation=2,
        )
        self.conv3 = nn.Conv1d(
            in_channels=self.num_filters * 2,
            out_channels=self.num_filters * 3,
            kernel_size=4,
            stride=1,
            padding=0,
            dilation=2,
        )
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(self.num_filters * 3, 1024)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.output = nn.Linear(512, self.output_dim)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)  # Transpose for Conv1d (N, C, L) format
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.global_max_pool(x).squeeze(
            2
        )  # Remove the last dimension after pooling
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.output(x)
        return x


def calculate_metrics(y_pred, y_true, epoch: int, type: str, fold: int):
    """
    Calculate metrics for the three proteins BRD4, HSA, sEH for bind1, bind2, bind3 respectively
    y_pred: list
    y_true: list
    epoch: int - current epoch for naming the file
    type: str - train or test
    fold: int - fold number
    """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    protein_names = ["BRD4", "HSA", "sEH"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for i, protein_name in enumerate(protein_names):
        print(f"\n {protein_name} - {type}")
        y_pred_ = y_pred[:, i]
        y_true_ = y_true[:, i]
        aps = average_precision_score(y_true_, y_pred_)
        print(f"Average Precision Score: {aps}")
        mlflow.log_metric(
            key=f"APS-{type}-{protein_name}", value=float(aps), step=epoch
        )

        # Define the threshold
        threshold = 0.5

        # Convert to 0 or 1 based on the threshold
        y_pred_thres = (y_pred_ > threshold).astype(int)

        cm = confusion_matrix(y_pred_thres, y_true_)
        classes = ["0", "1"]
        df_cfm = pd.DataFrame(cm, index=classes, columns=classes)
        cfm_plot = sns.heatmap(df_cfm, annot=True, cmap="Blues", fmt="g", ax=ax[i])
        ax[i].set_title(f"{protein_name} - APS: {aps:.4f} - {type}")
        ax[i].set_ylabel("Prediction")
        ax[i].set_xlabel("Ground Truth")
    cfm_plot.figure.savefig(dir + f"cm_fold_{fold}_{epoch}_{type}.png")
    mlflow.log_artifact(dir + f"cm_fold_{fold}_{epoch}_{type}.png")


def train_model(epoch, model, train_loader, optimizer, loss_fn, fold):
    """
    epoch: int
    model: torch.nn.Module
    train_loader: torch.utils.data.DataLoader
    optimizer: torch.optim.Optimizer
    loss_fn: torch.nn.Module
    fold: int
    """
    # Enumerate over the data
    all_preds = []
    all_labels = []
    running_loss = 0.0
    step = 0
    for _, batch in enumerate(tqdm(train_loader)):
        # Use GPU
        model = model.to(device)
        # Reset gradients
        optimizer.zero_grad()
        # Passing the node features and the connection info
        pred = model(batch[0].to(device))
        # Calculating the loss and gradients
        loss = loss_fn(pred, batch[1].to(device))
        loss.backward()
        optimizer.step()
        # Update tracking
        running_loss += loss.item()
        step += 1
        all_preds.append(torch.sigmoid(pred).cpu().detach().numpy())
        all_labels.append(batch[1].cpu().detach().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    # calculate_metrics(all_preds, all_labels, epoch, "train", fold)
    return running_loss / step


def test_model(epoch, model, test_loader, loss_fn, fold):
    """
    epoch: int
    model: torch.nn.Module
    test_loader: torch.utils.data.DataLoader
    loss_fn: torch.nn.Module
    fold: int
    """
    all_preds = []
    all_preds_raw = []
    all_labels = []
    running_loss = 0.0
    step = 0
    for batch in test_loader:
        x = batch[0]
        y = batch[1]
        pred = model(x.to(device))
        loss = loss_fn(pred, y.to(device))
        # Update tracking
        running_loss += loss.item()
        step += 1
        all_preds.append(torch.sigmoid(pred).cpu().detach().numpy())
        all_labels.append(y.cpu().detach().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Calculate metrics
    calculate_metrics(all_preds, all_labels, epoch, "test", fold)
    # log_conf_matrix(all_preds, all_labels, epoch)
    return running_loss / step


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


FEATURES = [f"enc{i}" for i in range(142)]
TARGETS = ["bind1", "bind2", "bind3"]
skf = StratifiedKFold(n_splits=CFG.NBR_FOLDS, shuffle=True, random_state=42)
all_preds = []

# sample the data
# train = train.sample(frac=0.01, random_state=42).reset_index(drop=True)

# Load data
with mlflow.start_run() as run:  # log the parameters with mlflow
    for fold, (train_idx, valid_idx) in enumerate(
        skf.split(train, train[TARGETS].sum(1))
    ):
        if fold not in CFG.SELECTED_FOLDS:
            continue

        # Convert pandas dataframes to PyTorch tensors
        X_train = torch.tensor(train.loc[train_idx, FEATURES].values, dtype=torch.int)
        y_train = torch.tensor(
            train.loc[train_idx, TARGETS].values, dtype=torch.float16
        )
        X_val = torch.tensor(train.loc[valid_idx, FEATURES].values, dtype=torch.int)
        y_val = torch.tensor(train.loc[valid_idx, TARGETS].values, dtype=torch.float16)

        # Create TensorDatasets
        train_dataset = TensorDataset(X_train, y_train)
        valid_dataset = TensorDataset(X_val, y_val)

        # Prepare dataloader
        train_loader = DataLoader(
            train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True
        )

        # Loading the model
        print("Loading model...")
        model = CNN_1D(lr=CFG.LR, weight_decay=CFG.WD)
        model = model.to(device)
        print(f"Number of parameters: {count_parameters(model)}")

        # < 1 increases precision, > 1 recall
        weight = torch.tensor(50, dtype=torch.float32).to(device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=CFG.LR,
            momentum=0.8,
            weight_decay=CFG.WD,
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        mlflow.log_params(
            {
                "weight_decay": CFG.WD,
                "learning_rate": CFG.LR,
                "kernel_size": 4,
            }
        )
        # Start training
        num_epoch = CFG.EPOCHS
        best_loss = 1000
        early_stopping_counter: int = 0
        for epoch in range(num_epoch):
            if early_stopping_counter <= 3:  # Early stopping with patience 5
                # Training
                model.train()
                loss = train_model(epoch, model, train_loader, optimizer, loss_fn, fold)
                mlflow.log_metric(key="Train loss", value=float(loss), step=epoch)
                print(f"Epoch {epoch} | Train Loss {loss}")
                # Testing
                model.eval()
                if (
                    epoch % 2 == 0
                ):  # Test every 2 epochs with validation data and calculate loss and average precision score
                    loss = test_model(epoch, model, valid_loader, loss_fn, fold)
                    print(f"Epoch {epoch} | Test Loss {loss}")
                    mlflow.log_metric(key="Test loss", value=float(loss), step=epoch)
                    # Update best loss
                    if float(loss) < best_loss:
                        early_stopping_counter = 0
                        best_loss = loss
                        # Save the currently best model
                        mlflow.pytorch.log_model(model, "model")
                        torch.save(
                            model.state_dict(),
                            dir + f"fold_{fold}_epoch_{epoch}_model.pth",
                        )
                    else:
                        early_stopping_counter += 1
                scheduler.step()
            else:
                print("Stop early due to no improvement")
                print(f"Finishing training with best test loss: {best_loss}")
                torch.save(
                    model.state_dict(),
                    dir + f"fold_{fold}_final_model.pth",
                )

                # return best_loss
