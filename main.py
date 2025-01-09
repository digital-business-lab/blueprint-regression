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
    
    dataloaders = {
        "data_train" : dataset.prepare_dataloaders(X_train, y_train, shuffle=True),
        "data_val" : dataset.prepare_dataloaders(X_val, y_val, shuffle=False),
        "data_test" : dataset.prepare_dataloaders(X_test, y_test, shuffle=False)
    }

    return dataloaders

def model_mode_output(model: Model, mode: str, X: pd.DataFrame =None):
    dataloaders = load_dataset()

    if mode == "train":
        model.train_model(
            train_loader=dataloaders["data_train"],
            val_loader=dataloaders["data_val"]
        )
        results = print("Model trained successfully!")

    elif mode == "evaluate":
        # model.evaluate()
        ...

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
            f"Wrong input! Select 'train', 'predict' or 'train_predict'! Your input was: '{mode}'"
            )
        
    return results


if __name__ == "__main__":
    config = ConfigYAML()
    modelName = config.config_data["model"]["modelName"]

    if modelName == "None":
        MODEL = Model(
            config.config_data["modelParams"]["input_size"],
            config.config_data["modelParams"]["hidden_size"],
            config.config_data["modelParams"]["output_size"]
        )
    else:
        MODEL = Model(
            config.config_data["modelParams"]["input_size"],
            config.config_data["modelParams"]["hidden_size"],
            config.config_data["modelParams"]["output_size"]
        )

        print("Loading model dict state")
        MODEL.load_state_dict(torch.load(
            f"{ConfigPaths().folder_model()}/{modelName}"
            ))

    MODE = config.config_data["model"]["modelMode"]

    model_mode_output(model=MODEL, mode=MODE)
