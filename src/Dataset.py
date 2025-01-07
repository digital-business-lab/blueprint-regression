""" 
Not implemented yet
"""
import pandas as pd
import chardet
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from Config import ConfigPaths, ConfigYAML


class Dataset(ConfigYAML):
    """Not implemented yet"""
    def __init__(self):
        super().__init__()

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
        
    def prepare_dataloaders(self, X: pd.DataFrame, y: pd.DataFrame, shuffle: bool):
        """
        Not sure if we should implement
        """
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, self.config_data["Model"]["batch_size"], shuffle)

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
