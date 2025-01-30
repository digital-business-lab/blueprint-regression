"""
This files brings all the components together:
    1. Loads / Updates Config
    2. Lodas Dataset into dataframes
    3. Executes model
File is written in pylint standard.

@author Lukas Graf
"""
from functools import partial

import optuna
import torch
import pandas as pd

from src.Dataset import Dataset
from src.Model import Model, objective
from src.Config import ConfigPaths, ConfigYAML


def load_dataset() -> dict:
    """ 
    This function loads and preprocesses the given dataset
        1. Train-Val-Test Split
        2. Preprocessing
        3. Creating dataloaders

    Returns
    -------
        dict
            -> DataLoaders for train, val and test
    """
    dataset = Dataset()
    data = dataset.read_dataset(
        f"{ConfigPaths().folder_data()}/{dataset.config_data['Dataset']['name']}"
        )
    X_train, X_val, X_test, y_train, y_val, y_test = dataset.split_dataset(
        data,
        dataset.config_data['Dataset']['target_column'], 
        dataset.config_data['Dataset']['train_size']
        )

    # Preprocess features
    X_train = pd.DataFrame(dataset.preprocess_dataset(X_train, mode="train"))
    X_val = pd.DataFrame(dataset.preprocess_dataset(X_val, mode="test"))
    X_test = pd.DataFrame(dataset.preprocess_dataset(X_test, mode="test"))

    # Get final number of features
    dataset.config_data["modelParams"]["input_size"] = len(X_train.columns)
    ConfigYAML.write_yaml(dataset.config_data)

    dataloaders = {
        "data_train" : dataset.prepare_dataloaders(X_train, y_train, shuffle=True),
        "data_val" : dataset.prepare_dataloaders(X_val, y_val, shuffle=False),
        "data_test" : dataset.prepare_dataloaders(X_test, y_test, shuffle=False)
    }

    return dataloaders

def model_mode_output(model_name: str, mode: str, X: pd.DataFrame =None):
    """ 
    Loads, trains / evaluates / makes prediction model and performs
    hyperparameter optimization if train and hypTuning
        1. Load a given model or create a new one
        2. Hyperparameteroptimization
        3. Make training / evaluation / prediction

    Parameters:
    -----------
        model_name : str
            -> Name of the model which should be loaded
            -> If None, a new one will be created
        mode : str
            -> Mode which the model should execute
            -> "train", "evaluate", "predict", "train_predict"
        X : pd.DataFrame
            -> Value for predictions
    Returns:
    --------
        Any
    """
    dataloaders = load_dataset()
    dataset = Dataset()

    # Hyperparam tuning and updating params for final model
    hyp_tuning: bool = dataset.config_data["HyperParamTuning"]["hypTuning"]
    if hyp_tuning and mode=="train":
        study = optuna.create_study(direction="minimize")
        study.optimize(
            partial(
                objective, 
                train_loader=dataloaders["data_train"], 
                val_loader=dataloaders["data_val"]
                ),
            n_trials=dataset.config_data["HyperParamTuning"]["hypNumTrials"]
        )

        best_params = study.best_params
        dataset.config_data["modelParams"]["hidden_size"] = best_params["hidden_size"]
        dataset.config_data["modelParams"]["dropout_rate"] = best_params["dropout_rate"]
        dataset.config_data["modelParams"]["lr"] = best_params["learning_rate"]
        ConfigYAML.write_yaml(dataset.config_data)

    elif hyp_tuning and mode!="train":
        raise ValueError(
            "If you want to do hyperparameter optimization you have to set modelMode to train!"
            )

    # Choose model
    if model_name is None:
        model = Model(
            dataset.config_data["modelParams"]["input_size"],
            dataset.config_data["modelParams"]["hidden_size"],
            dataset.config_data["modelParams"]["output_size"]
        )
    else:
        model = Model(
            dataset.config_data["modelParams"]["input_size"],
            dataset.config_data["modelParams"]["hidden_size"],
            dataset.config_data["modelParams"]["output_size"]
        )

        print("Loading model dict state")
        model.load_state_dict(torch.load(
            f"{ConfigPaths().folder_model()}/{model_name}",
            weights_only=True
            ))

    # Choose model mode
    if mode == "train":
        loss, r2 = model.train_model(
                            train_loader=dataloaders["data_train"],
                            val_loader=dataloaders["data_val"]
                        )
        results = {"loss" : loss, "r2" : r2}

    elif mode == "evaluate":
        avg_loss, r2 = model.evaluate(val_loader=dataloaders["data_test"])
        results = {"avg_loss" : avg_loss, "r2" : r2}

    elif mode == "predict":
        X = torch.tensor(X.values, dtype=torch.float32)
        results = model.predict(X)

    elif mode == "train_predict":
        model.train_model(
            train_loader=dataloaders["data_train"],
            val_loader=dataloaders["data_val"]
        )

        X = torch.tensor(X.values, dtype=torch.float32)
        results = model.predict(X)

    else:
        results = ValueError(
            "Wrong input! Select 'train', 'evaluate', 'predict' or 'train_predict'!"
            )

    return results


if __name__ == "__main__":
    config = ConfigYAML()
    modelName = config.config_data["model"]["modelName"]
    MODE = config.config_data["model"]["modelMode"]
    result = model_mode_output(model_name=modelName, mode=MODE)
    print(result)
