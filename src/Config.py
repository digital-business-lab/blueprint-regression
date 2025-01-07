""" 
Holds all the classes and functions for config specific operations

File is written in pylint standard
"""

import os

import yaml


class ConfigPaths:
    """
    Class config paths
    ---------------------
    Methods:
        folder_config
            Path to the folder 'config'
        folder_data
            Path to the folder 'data'
        folder_model
            Path to the folder 'model'
        folder_src
            Path to the folder 'src'
        folder_mlruns
            Path to the folder 'mlruns'

    Private Methods:
        __path_checker
            Checks if the path exists
    """
    def __init__(self):
        self.defaultpaths = os.listdir()

    def folder_config(self) -> str:
        """
        Path to the folder 'config'

        return: str
        """
        if "config" in self.defaultpaths:
            config_folder = os.path.join(".", "config")
        else:
            config_folder = os.path.join("..", "config")
        return self.__path_checker(config_folder)

    def folder_data(self) -> str:
        """
        Path to the folder 'data'

        return: str
        """
        if "data" in self.defaultpaths:
            folder_data = os.path.join(".", "data")
        else:
            folder_data = os.path.join("..", "data")
        return self.__path_checker(folder_data)

    def folder_model(self) -> str:
        """
        Path to the folder 'model'

        return: str
        """
        if "model" in self.defaultpaths:
            folder_model = os.path.join(".", "model")
        else:
            folder_model = os.path.join("..", "model")
        return self.__path_checker(folder_model)

    def folder_src(self) -> str:
        """
        Path to the folder 'src'

        return: str
        """
        if "src" in self.defaultpaths:
            folder_src = os.path.join(".", "src")
        else:
            folder_src = os.path.join("..", "src")
        return self.__path_checker(folder_src)

    def folder_mlruns(self) -> str:
        """
        Path to the folder 'mlruns'

        return: str
        """
        if "mlruns" in self.defaultpaths:
            folder_mlruns = os.path.join(".", "mlruns")
        else:
            folder_mlruns = os.path.join("..", "mlruns")
        return self.__path_checker(folder_mlruns)

    #----------Private Methods----------#
    def __path_checker(self, path: str) -> str:
        """
        Checks if the path exists
        Parameters:
            path: str

        return: str
        """
        return path if os.path.exists(path) else print("Path does not exist.")


class ConfigYAML:
    """
    Class config paths
    ---------------------
    Methods:
        read_yaml
            Reads data out of yaml-file

    Static Methods:
        write_yaml
            Writes data into yaml file
    """
    def __init__(self):
        self.config_data = self.read_yaml()

    def read_yaml(self, file_path: str ="./config/config.yaml") -> dict:
        """
        Reads data out of yaml-file

        Parameters:
        -----------
            file_path: str -> './config/config.yaml'

        Returns:
        -------
            dict
        """
        with open(file_path, "r", encoding="utf-8") as file:
            data: dict = yaml.safe_load(file)
        return data

    @staticmethod
    def write_yaml(data: dict, file_path: str ="./config/config.yaml") -> None:
        """
        Writes data into yaml-file

        Parameters:
        -----------
            file_path: str -> './config/config.yaml'

        Returns:
        -------
            None
        """
        yaml_str = yaml.safe_dump(data, sort_keys=False)
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(yaml_str)


if __name__ == "__main__":
    print(ConfigPaths().folder_config())
