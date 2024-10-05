import gc
import os
import pickle
import random
import joblib
import pandas as pd
import duckdb

# import polars as pd
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set PJRT_DEVICE to CPU
#
#
# os.environ["PJRT_DEVICE"] = "CPU"
# # Check if CUDA is available and set the device to GPU or fall back to CPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print(f"Using device: {device}")


test = pd.read_parquet("../../data/test_enc.parquet")

print("Loading data completed")


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
        x = torch.sigmoid(x)
        return x


model_path = "../result/20240811_0557/fold_2_epoch_50_model.pth"
date_time = model_path.split("/")[-2]
results_path = "../result/" + date_time + "/"
model_name = model_path.split("/")[-1].split(".")[0]
model = CNN_1D()
model.load_state_dict(torch.load(model_path))
model = model.to(device)

FEATURES = [f"enc{i}" for i in range(142)]
# Convert pandas dataframes
# to PyTorch tensors
X_test = torch.tensor(test.loc[:, FEATURES].values, dtype=torch.int)
test_dataset = TensorDataset(X_test)
test_loader = DataLoader(test_dataset, batch_size=256, num_workers=40)
print("Model loaded. Starting inference...")
# Disable gradient computation for inference
model.eval()
preds = []
with torch.no_grad():
    for _, batch in enumerate(tqdm(test_loader)):
        pred = model(batch[0].to(device))
        pred = pred.cpu().detach().numpy()
        pred = pred.tolist()
        for _ in pred:
            preds.append(_)

preds = np.array(preds)
print("Inference completed. Saving submission file...")
tst = pd.read_parquet("../../data/test.parquet")
tst["binds"] = 0
tst["binds"] = tst["binds"].astype(float)

tst.loc[tst["protein_name"] == "BRD4", "binds"] = preds[
    tst["protein_name"] == "BRD4", 0
]
tst.loc[tst["protein_name"] == "HSA", "binds"] = preds[tst["protein_name"] == "HSA", 1]

tst.loc[tst["protein_name"] == "sEH", "binds"] = preds[tst["protein_name"] == "sEH", 2]

# saving file
tst[["id", "binds"]].to_parquet(
    results_path + f"{date_time}_{model_name}_submission.parquet", index=False
)
print("Submission file saved.")
