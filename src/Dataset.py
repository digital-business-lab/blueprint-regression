""" 
This files holds a class responsible for preprocessing
the dataset.
File is written in pylint standard.

@author Lukas Graf
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
        """
        Reads your dataset
        """
        encoding = self.__get_encoding(file_path=file_path)
        data = pd.read_csv(file_path, encoding=encoding)
        return data

    def split_dataset(self, data: pd.DataFrame, target_variable: str, train_size: float):
        """
        Splits the dataset into train-, val- and test-dataset

        Parameters:
        -----------
            data : pd.DataFrame
                -> The wholte data read from your file
            target_variable : str
                -> Target variable (y)
            train_size : float
                -> Train size (Test size = 1 - Train size)
        """
        if train_size >= 1:
            return ValueError("Size of trainingdata can not be bigger than the whole dataset!")

        y = data[target_variable]
        X = data.drop(target_variable, axis=1)
        test_size = 1 - train_size

        # The random_state parameter is important. It does not effect your dataset
        # split but rather ensures that the split for the same dataset is always
        # the same. -> This is needed so a trained model for the dataset has the
        # same input_size when you want to evaluate or pretrain the model again
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_size, random_state=42
            )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
            )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def preprocess_dataset(self, X, mode: str):
        """ 
        Preprocesses the dataset.
            1. Drop unwanted columns
            2. Apply ColumnTransformer for Categorical and numerical values

        Parameters:
        -----------
            X
                -> Desired Dataframe that should be preprocessed
            mode : str
                -> Given mode "train" or "test". If test the ColumnsTransformer
                -> will not be fitted

        Returns:
        --------
            DenseMatrix
        """
        # Validate mode
        if mode not in ["train", "test"]:
            raise ValueError("Mode must be 'train' or 'test'!")

        # Variables for dropping columns
        drop_columns: list = self.config_data["Preprocessing"]["dropColumns"]
        X = X.drop(drop_columns, axis=1)

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
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
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
                raise ValueError(
                    "Transformer has not been fitted. Please preprocess training data first."
                    )
            X = self.column_transformer.transform(X)

        print("Preprocessing was successfully!")
        return X

    def prepare_dataloaders(self, X: pd.DataFrame, y: pd.DataFrame, shuffle: bool) -> DataLoader:
        """
        PyTorch models work with DataLoaders. This method prepares them.

        Parameters:
        -----------
            X : pd.DataFrame
                -> Dataframe of features
            y : pd.DataFrame
                -> Target variable
            shuffle : bool
                -> Shuffles dataset if true (Used for training)

        Returns:
        --------
            DataLoader
        """
        X = torch.tensor(X.values, dtype=torch.float32)
        y = torch.tensor(y.values, dtype=torch.float32)
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, self.config_data["modelParams"]["batch_size"],
                          shuffle, drop_last=True)

    #----------Private Methods----------#
    def __get_encoding(self, file_path: str) -> str:
        """
        Automatically gets right encoding

        Parameters:
        -----------
            file_path : str
                -> File path to dataset

        Returns:
        --------
            str
                -> right encoding method
        """
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())

        return result["encoding"]


if __name__ == "__main__":
    DATASET = Dataset()
    read_data = DATASET.read_dataset(
        f"{ConfigPaths().folder_data()}/{DATASET.config_data['Dataset']['name']}"
        )
    X_TRAIN, X_VAL, X_TEST, y_TRAIN, y_VAL, y_TEST = DATASET.split_dataset(
        read_data, DATASET.config_data['Dataset']['target_column'], 0.7
        )
    print(len(X_TRAIN))
    print(len(X_VAL))
    print(len(X_TEST))
    print(len(y_TRAIN))
    print(len(y_VAL))
    print(len(y_TEST))
