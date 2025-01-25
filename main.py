"""
Not implemented yet
"""
import torch
import pandas as pd

from src.Model import Model
from src.Dataset import Dataset
from src.Config import ConfigPaths, ConfigYAML


def load_dataset() -> dict:
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
    dataloaders = load_dataset()
    input_size = Dataset().config_data["modelParams"]["input_size"]

    # Choose model
    if model_name == "None":
        model = Model(
            input_size,
            config.config_data["modelParams"]["hidden_size"],
            config.config_data["modelParams"]["output_size"]
        )
    else:
        model = Model(
            input_size,
            config.config_data["modelParams"]["hidden_size"],
            config.config_data["modelParams"]["output_size"]
        )

        print("Loading model dict state")
        model.load_state_dict(torch.load(
            f"{ConfigPaths().folder_model()}/{modelName}",
            weights_only=True
            ))


    # Choose model mode
    if mode == "train":
        model.train_model(
            train_loader=dataloaders["data_train"],
            val_loader=dataloaders["data_val"]
        )
        results = print("Model trained successfully!")

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
