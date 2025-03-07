""" 
This file holds the model.
File is written in pylint standard.

@author Lukas Graf
"""
import logging

import numpy as np
import mlflow
from mlflow import pytorch
import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt

from src.Config import ConfigYAML, ConfigPaths


class Model(nn.Module, ConfigYAML):
    """
    This class defines the model for training / evaluating / predicting

    Attributes:
    -----------
        input_size : int
            -> Number of features the model should expect as input
        hidden_size : int
            -> Size of hidden layers
        output_size : int, optional
            -> Output of Model. For regression its usually 1
            -> Default to 1
        dropout_rate : float, optional
            -> Percentage of neurans that randomly deactvate during 
            training (helps against overfitting)
            -> Defaults to 0.3 (30%)

    Methods:
    --------
        forward
            -> Forward-pass of the model
        train_model
            -> Trains the model
        evaluate
            -> evaluates the model
        predict
            -> Makes predictions
    """
    def __init__(self, input_size: int, hidden_size: int,
                 output_size: int):
        ConfigYAML.__init__(self)
        nn.Module.__init__(self)
        # Initialize logger
        self.logger = logging.getLogger(__name__)

        self.lr = self.config_data["modelParams"]["lr"]
        self.dropout_rate = self.config_data["modelParams"]["dropout_rate"]

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)

        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)

        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)

        self.layer4 = nn.Linear(hidden_size, hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size)

        self.layer5 = nn.Linear(hidden_size, output_size)

        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.relu = nn.ReLU()

        self.logger.info("Initialized Model layers.")

    def forward(self, x):
        """
        Forward-pass of the model. Should not be used!

        Parameters:
        -----------
            x

        Returns:
        --------
            x
        """
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.dropout(x)

        x = self.relu(self.bn2(self.layer2(x)))
        x = self.dropout(x)

        x = self.relu(self.bn3(self.layer3(x)))
        x = self.dropout(x)

        x = self.relu(self.bn4(self.layer4(x)))
        x = self.dropout(x)

        x = self.layer5(x)
        return x

    def train_model(self, train_loader, val_loader, save: bool =True) -> None:
        """
        Trains and saves the model

        Parameters:
        -----------
            train_loader : DataLoader
                -> Dataloader for training set
            val_loader : DataLoader
                -> DataLoader for validation / test set

        Returns:
        --------
            None
        """
        self.logger.info("Started model training.")
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        epochs = self.config_data["modelParams"]["epochs"]
        criterion = nn.MSELoss()

        input_example = np.random.randn(1, self.layer1.in_features).astype(np.float32)
        # Start MLflow experiment
        with mlflow.start_run():
            mlflow.log_param("learning_rate", self.lr)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("dropout_rate", self.dropout.p)

            for epoch in range(epochs):
                self.train()
                running_loss = 0.0

                for batch in train_loader:
                    inputs, targets = batch
                    optimizer.zero_grad()
                    outputs = self(inputs)

                    #Remove extra dimension
                    outputs = outputs.squeeze()

                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                train_loss = running_loss / len(train_loader)
                self.logger.info(
                    "Epoch [%d / %d], Training loss: %.4f", epoch+1, epochs, train_loss
                    )
                mlflow.log_metric("training_loss", train_loss, step=epoch + 1)

                # Validation
                val_loss, r2 = self.evaluate(val_loader=val_loader, criterion=criterion)
                self.logger.info("Validation loss after epoch %d: %.4f", epoch + 1, val_loss)
                self.logger.info("Validation R2 after epoch %d: %.4f", epoch+1, r2)
                mlflow.log_metric("validation_loss", val_loss, step=epoch + 1)
                mlflow.log_metric("validation_r2", r2, step=epoch + 1)

            # Save model
            if save:
                model_name = str(self.config_data['Dataset']['name']).split('.', maxsplit=1)[0]
                model_path = f"{ConfigPaths().folder_model()}/{model_name}.pth"
                torch.save(self.state_dict(), model_path)
                mlflow.log_artifact(model_path, artifact_path="models")
                pytorch.log_model(self, "pytorch-model", input_example=input_example)
                self.logger.info("Saved model.")

            self.logger.info("Finished model training.")
            return val_loss, r2

    def evaluate(self, val_loader, criterion=nn.MSELoss()) -> list:
        """
        Evaluates the model's performance on the validation set and computes R² score

        Parameters:
        -----------
            val_loader : DataLoader
                -> Loader for validation / test set
            criterion : optional
                -> Loss Function

        Returns:
        --------
            list
        """
        self.eval()
        self.logger.info("Set model to evaluation mode.")
        val_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():  # No gradients needed for evaluation
            for batch in val_loader:
                inputs, targets = batch
                outputs = self(inputs)

                #Remove extra dimension
                outputs = outputs.squeeze()

                # Calculate loss
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                # Collect predictions and targets for further evaluation
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        # Convert all predictions and targets to numpy arrays
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        if self.config_data["Visualization"]["visBool"]:
            plt.figure(figsize=(12, 8))
            plt.title("Comparison between Prediction and GroundTruth")
            plt.grid(True)
            plt.xlabel("Datapoints")
            plt.ylabel(self.config_data["Dataset"]["target_column"])

            # Visualize lines
            max_steps = self.config_data["Visualization"]["visMaxSteps"]
            if max_steps is None:
                step = 1
            else:
                step: int = max(1, len(all_predictions) // int(max_steps))
            plt.plot(all_predictions[::step], color="blue", label="Prediction", alpha=0.7)
            plt.plot(all_targets[::step], color="orange", label="GroundTruth", alpha=0.7)

            plt.legend()
            format_vis = self.config_data["Visualization"]["visFormat"]
            plt.savefig(
                f"./model/plots/evaluate_{str(
                    self.config_data['Dataset']['name']
                    ).split('.', maxsplit=1)[0]}.{format_vis}",
                    format=format_vis)

            self.logger.info("Visualization of evaluation was saved.")

        # Calculate Mean Squared Error (MSE)
        avg_loss = val_loss / len(val_loader)

        # Calculate R² (Coefficient of Determination)
        r2 = self.__r2_score(all_targets, all_predictions)

        return avg_loss, r2

    def predict(self, X):
        """
        Makes predictions

        Parameters:
        -----------
            X
                -> Dataset on which predictions should be made
        """
        self.eval()
        self.logger.info("Set model to evaluation mode.")
        with torch.no_grad():
            predictions = self(X)

        return predictions

    #---------Private Methods---------#
    def __r2_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the R² (coefficient of determination) score
        Score of -1 worse than random model
        Score of 0 bad but not as bad as random model
        Score of 1 perfect

        Parameters:
        -----------
            y_true : np.ndarray
                -> Ground Truth
            y_pred : np.ndarray
                -> predicted y values

        Returns:
        --------
            float
                -> R2 score
        """
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        r2: float = 1 - (ss_residual / ss_total)
        return r2

# Objective function for optuna
def objective(trial, train_loader, val_loader) -> float:
    """ 
    Objective function for optunas hyperparamet optimization

    Parameters:
    -----------
        trial -> Needed for optuna (automatic)
        train_loader
            -> Training DataLoader
        val_loader
            -> Validation DataLoader

    Returns:
    --------
        float
            -> Loss of the model
    """
    config_params = ConfigYAML().config_data
    input_size = config_params["modelParams"]["input_size"]
    output_size = config_params["modelParams"]["output_size"]

    # Hyperparams to tune
    hidden_size = trial.suggest_int("hidden_size", 32, 256, step=32)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.8)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

    # Create model
    model = Model(input_size, hidden_size, output_size)
    model.lr = learning_rate
    model.dropout_rate = dropout_rate

    # Returns loss and r2
    loss, _ = model.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        save=False)
    return loss


if __name__ == "__main__":
    ...
