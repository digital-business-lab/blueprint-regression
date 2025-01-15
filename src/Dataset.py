""" 
Not implemented yet
"""
import torch
import chardet
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, TensorDataset

from src.Config import ConfigPaths, ConfigYAML


class Dataset(ConfigYAML):
    """Not implemented yet"""
    def __init__(self):
        super().__init__()
        self.column_transformer = None

    def read_dataset(self, file_path: str):
        """Not implemented yet"""
        encoding = self.__get_encoding(file_path=file_path)
        data = pd.read_csv(file_path, encoding=encoding)
        return data

    def split_dataset(self, data: pd.DataFrame, target_variable: str, train_size: float):
        """Not implemented yet"""
        if train_size >= 1:
            return ValueError("Size of trainingdata can not be bigger than the whole dataset!")
        else:
            y = data[target_variable]
            X = data.drop(target_variable, axis=1)
            test_size = 1 - train_size
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

            return X_train, X_val, X_test, y_train, y_val, y_test
        
    def preprocess_dataset(self, X, mode: str):
        # Validate mode
        if mode not in ["train", "test"]:
            raise ValueError("Mode must be 'train' or 'test'!")
        
        # Variables for numerical columns
        cols_imp_num: list = self.config_data["Preprocessing"]["numColumns"]
        strat_imp_num: str = self.config_data["Preprocessing"]["numColsProcessor"]
        
        # Variables for categorical columns
        cols_imp_cat: list = self.config_data["Preprocessing"]["catColumns"]
        strat_imp_cat: str = self.config_data["Preprocessing"]["catColsProcessor"]

        scale_num: bool = self.config_data["Preprocessing"]["numColsScale"]

        pipe_preprocessing_num = Pipeline([
            ("imp_num", SimpleImputer(strategy=strat_imp_num)),
            ("scaler", StandardScaler()) if scale_num else ("passthrough", "passthrough")
        ])

        pipe_preprocessing_cat = Pipeline([
                ("imp_cat", SimpleImputer(strategy=strat_imp_cat)),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ])

        # Initialize ColumnTransformer (only on training)
        if mode == "train":
            self.column_transformer = ColumnTransformer([
                ("preprocess_num", pipe_preprocessing_num, cols_imp_num),
                ("preprocess_cat", pipe_preprocessing_cat, cols_imp_cat)
            ],
            remainder="passthrough")
            X = self.column_transformer.fit_transform(X)

        elif mode == "test":
            if not self.column_transformer:
                raise ValueError("Transformer has not been fitted. Please preprocess training data first.")
            X = self.column_transformer.transform(X)

        print("Preprocessing was successfully!")
        return X

    def prepare_dataloaders(self, X: pd.DataFrame, y: pd.DataFrame, shuffle: bool):
        """Not implemented yet"""
        X = torch.tensor(X.values, dtype=torch.float32)
        y = torch.tensor(y.values, dtype=torch.float32)
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, self.config_data["modelParams"]["batch_size"], shuffle, drop_last=True)

    #----------Private Methods----------#
    def __get_encoding(self, file_path: str) -> str:
        """Automatically gets right encoding"""
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
 
        return result["encoding"]


if __name__ == "__main__":
    dataset = Dataset()
    read_data = dataset.read_dataset(
        f"{ConfigPaths().folder_data()}/{dataset.config_data['Dataset']['name']}"
        )
    X_train, X_val, X_test, y_train, y_val, y_test = dataset.split_dataset(
        read_data, dataset.config_data['Dataset']['target_column'], 0.7
        )
    print(len(X_train))
    print(len(X_val))
    print(len(X_test))
    print(len(y_train))
    print(len(y_val))
    print(len(y_test))
