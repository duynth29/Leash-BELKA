# Standard library imports
from datetime import datetime
import os
import re
import unicodedata
import warnings

# Third party imports
import duckdb
import matplotlib as mpl
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import pysmiles
import seaborn as sns
from matplotlib import pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import IPythonConsole
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from tqdm import tqdm
from xgboost import XGBClassifier, plot_importance as plot_importance_xgb
from sklearn.utils import shuffle


# Local application/library specific imports
import watermark

# Configuration and settings
IPythonConsole.ipython_useSVG = True
sns.set_theme(style="whitegrid")
palette = "viridis"
warnings.filterwarnings("ignore")
import json

# import packages for hyperparameters tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe


def modl(molecule_data, radius=4, bits=1024):
    if molecule_data is None:
        return None
    return list(
        AllChem.GetMorganFingerprintAsBitVect(molecule_data, radius, nBits=bits)
    )


def load_data_rep_pos(
    protein_name: str, i: int = 0, n_limit: int = 1e6, n_rep: float = 0.5, offset=0
):
    """Load data and replicate the positive class to balance the dataset (default: 1: 0.5 ratio)."""
    df_path = f"../../data/targets/{protein_name}/chunks/{protein_name}_{i}.parquet"
    con = duckdb.connect()
    df = con.query(
        f"""(SELECT fp, binds
            FROM parquet_scan('{df_path}')
            ORDER BY random()
            LIMIT {n_limit}
            OFFSET {offset}
            )"""
    ).df()
    con.close()
    df["fp"] = df["fp"].apply(lambda x: np.array(x))
    # Check how many additional samples we need
    neg_class = df["binds"].value_counts()[0]
    pos_class = df["binds"].value_counts()[1]
    multiplier = int(n_rep * neg_class / pos_class) - 1

    # Replicate the dataset for the positive class
    replicated_pos = [df[df["binds"] == 1]] * multiplier

    # Append replicated data
    df = df._append(replicated_pos, ignore_index=True)

    # Shuffle dataset
    df = df.sample(frac=1).reset_index(drop=True)

    X = np.array(df["fp"].tolist())
    y = np.array(df["binds"].tolist())
    return X, y


def load_data(protein_name: str, i: int = 0, n_limit: int = 1e3):
    df_path = f"../../data/targets/{protein_name}/chunks/{protein_name}_{i}.parquet"
    con = duckdb.connect()
    df = con.query(
        f"""(SELECT fp, binds
            FROM parquet_scan('{df_path}')
            ORDER BY random()
            LIMIT {n_limit}
            )"""
    ).df()
    con.close()
    df["fp"] = df["fp"].apply(lambda x: np.array(x))

    # Shuffle dataset
    df = df.sample(frac=1).reset_index(drop=True)

    X = np.array(df["fp"].tolist())
    y = np.array(df["binds"].tolist())

    return X, y


def load_data_by_chunk(
    protein_name: str, i: int = 0, n_limit: int = 1e6, offset: int = 0
):
    df_path = f"../../data/targets/{protein_name}/chunks/{protein_name}_{i}.parquet"
    con = duckdb.connect()
    df = con.query(
        f"""(SELECT fp, binds
            FROM parquet_scan('{df_path}')
            ORDER BY random()
            LIMIT {n_limit}
            OFFSET {offset}
            )"""
    ).df()
    con.close()
    df["fp"] = df["fp"].apply(lambda x: np.array(x))

    # Shuffle dataset
    df = df.sample(frac=1).reset_index(drop=True)

    X = np.array(df["fp"].tolist())
    y = np.array(df["binds"].tolist())

    return X, y


def load_pos_data(protein_name: str, i: int = 0, n_limit: int = 1e6):
    df_path = (
        f"../../data/targets/{protein_name}/bind_compound/{protein_name}_{i}.parquet"
    )
    con = duckdb.connect()
    df = con.query(
        f"""(SELECT fp, binds
            FROM parquet_scan('{df_path}')
            ORDER BY random()
            LIMIT {n_limit}
            )"""
    ).df()
    con.close()
    df["fp"] = df["fp"].apply(lambda x: np.array(x))

    # Shuffle dataset
    df = df.sample(frac=1).reset_index(drop=True)

    X = np.array(df["fp"].tolist())
    y = np.array(df["binds"].tolist())

    return X, y


def plot_result(protein_name, y_test, X_test, model, date_string, i: int):

    y_pred = model.predict(X_test)

    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    average_precision = average_precision_score(y_test, y_pred)

    print("Accuracy:", acc, "Average Precision:", average_precision)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # Plotting the ROC curve
    ax[0].plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    ax[0].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax[0].set_xlim([0.0, 1.0])
    ax[0].set_ylim([0.0, 1.05])
    ax[0].set_xlabel("False Positive Rate")
    ax[0].set_ylabel("True Positive Rate")
    ax[0].set_title("Receiver Operating Characteristic")
    ax[0].legend(loc="lower right")
    ax[0].grid(False)
    ax[0].text(
        0.6,
        0.6,
        f"Classification Report = {classification_report(y_test, y_pred)}",  # Classification Report
        fontsize=8,
    )
    ax[0].text(
        0.6,
        0.5,
        f"Average precision = {average_precision:.5f}",  # Average precision
        fontsize=8,
    )

    # Plotting the confusion matrix with Seaborn
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax[1])
    ax[1].set_title("Confusion Matrix - XGBoost")
    ax[1].set_xlabel("Predicted")
    ax[1].set_ylabel("Actual")
    ax[1].set_xticks(ticks=[0.5, 1.5], labels=["Non-bind", "Bind"])
    ax[1].set_yticks(ticks=[0.5, 1.5], labels=["Non-bind", "Bind"], rotation=0)
    plt.tight_layout()
    plt.savefig(
        f"../results/{protein_name}/{date_string}/{date_string}_{i}_eval_metrix_xgboost.png"
    )


def load_test_data(protein_name, dir_in):
    filename = dir_in
    if os.path.isfile(filename):
        print("File already exists. Loading data...")
        df = pd.read_csv(filename)
        # Convert the string representation of lists to actual arrays
        X = df["H1_ecfp"].apply(lambda x: np.array(json.loads(x))).tolist()
        idx = df["Id"].tolist()
        print("Data loaded.")
    else:
        data_test = "../../data/test.parquet"
        print("Loading evaluation data...")
        # Bank connection
        con = duckdb.connect()
        test = con.query(
            f"""SELECT * FROM parquet_scan('{data_test}') 
            WHERE protein_name = '{protein_name}'
            ORDER BY random()
            LIMIT 1e6"""
        ).df()
        con.close()
        print("Loading evaluation data completed...")
        X = []
        idx = []
        for smiles, id in tqdm(
            zip(test["molecule_smiles"], test["id"]), total=len(test["id"])
        ):
            mol = Chem.MolFromSmiles(smiles)
            H1_ecfp = modl(mol)
            X.append(H1_ecfp)
            idx.append(id)

        df_export = pd.DataFrame({"Id": idx, "H1_ecfp": X})
        df_export.to_csv(filename, index=False)
    return X, idx


def evaluation(model, X, idx, date_string, protein_name):
    print("Evalutation started...")
    # Use the model to make probability predictions
    prob_predictions = model.predict_proba(X)
    # If the positive class is the second class, you would use prob_predictions[:, 1]
    positive_probabilities = prob_predictions[:, 1]
    # Create a pandas DataFrame with the predicted probabilities
    df = pd.DataFrame({"id": idx, "prediction": positive_probabilities})
    df.to_csv(
        f"../results/{protein_name}/{date_string}/{date_string}_prediction_XGboost.csv",
        index=False,
    )
    print("Evalutation completed.")


if __name__ == "__main__":

    # training parameters
    protein_name = "HSA"

    os.makedirs(f"../results/{protein_name}", exist_ok=True)

    # Train the model on the remaining chunks
    n_chunks = 6

    # number of molecules per batch to load
    n_limit = 300e3

    print(f"Optimize paramaters for {protein_name}...")
    date_string = datetime.now().strftime("%Y%m%d_%H%M")

    space = {
        "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)),
        "max_depth": hp.quniform("max_depth", 3, 18, 1),
        # "gamma": hp.uniform("gamma", 1, 9),
        "reg_alpha": hp.quniform("reg_alpha", 40, 180, 1),
        # "reg_lambda": hp.uniform("reg_lambda", 0, 1),
        "subsample": hp.uniform("subsample", 0.1, 0.5),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
        # "min_child_weight": hp.quniform("min_child_weight", 0, 10, 1),
        "n_estimators": hp.quniform("n_estimators", 100, 1000, 1),
        "seed": 42,
    }

    # Load evaluation data
    print("Loading the first chunk of data as evaluation...")
    X_test, y_test = load_data(protein_name, i=0, n_limit=300e3)
    eval_set = [(X_test, y_test)]

    def objective(space):

        param = {
            "learning_rate": space["learning_rate"],
            "subsample": space["subsample"],
            "sampling_method": "gradient_based",
            "colsample_bytree": space["colsample_bytree"],
            "objective": "binary:logistic",
            "random_state": 42,
            "tree_method": "gpu_hist",
            "predictor": "gpu_predictor",
            "eval_metric": "logloss",
            "max_depth": int(space["max_depth"]),
            "n_estimators": int(space["n_estimators"]),
            "reg_alpha": int(space["reg_alpha"]),
        }

        model = XGBClassifier(**param)
        for i in range(1, n_chunks):
            if i == 1:
                X_pos, y_pos = load_pos_data(protein_name, i=0, n_limit=n_limit)
                X, y = load_data_by_chunk(protein_name, i, n_limit=n_limit, offset=0)
                X = np.concatenate([X, X_pos])
                y = np.concatenate([y, y_pos])
                # shuffle X and y
                X, y = shuffle(X, y, random_state=42)
                model.fit(
                    X, y, eval_set=eval_set, early_stopping_rounds=10, verbose=False
                )

                prob_predictions = model.predict_proba(X_test)[:, 1]

                average_precision_prob = average_precision_score(
                    y_test, prob_predictions
                )
                print(
                    f"Round {i}: average precision using probability:  {average_precision_prob}"
                )
                threshold = 0.22
                if average_precision_prob < threshold:
                    print(f"Stop the loop, average precision is less than {threshold}")
                    break

            else:
                X_pos, y_pos = load_pos_data(protein_name, i=0, n_limit=n_limit)
                X, y = load_data_by_chunk(protein_name, i, n_limit=n_limit, offset=0)
                X = np.concatenate([X, X_pos])
                y = np.concatenate([y, y_pos])
                # shuffle X and y
                X, y = shuffle(X, y, random_state=42)
                model.fit(
                    X,
                    y,
                    eval_set=eval_set,
                    early_stopping_rounds=10,
                    xgb_model=model.get_booster(),
                    verbose=False,
                )

                prob_predictions = model.predict_proba(X_test)[:, 1]

                average_precision_prob = average_precision_score(
                    y_test, prob_predictions
                )
                print(
                    f"Round {i}: average precision using probability:  {average_precision_prob}"
                )
        return {"loss": -average_precision_prob, "status": STATUS_OK}

    trials = Trials()

    best_hyperparams = fmin(
        fn=objective, space=space, algo=tpe.suggest, max_evals=75, trials=trials
    )
    best_hyperparams_df = pd.DataFrame([best_hyperparams])
    best_hyperparams_df["average_precision"] = -trials.best_trial["result"]["loss"]
    # Save the DataFrame to a CSV file
    best_hyperparams_df.to_csv(
        f"../results/{protein_name}/{date_string}_best_hyperparams.csv", index=False
    )
