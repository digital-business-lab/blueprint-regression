""" 
This file holds the model.
File is written in pylint standard.

@author Lukas Graf
"""
import numpy as np
import torch
from torch import nn
from torch import optim

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
                 output_size: int =1, dropout_rate: float =0.3):
        ConfigYAML.__init__(self)
        nn.Module.__init__(self)

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)

        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)

        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)

        self.layer4 = nn.Linear(hidden_size, hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size)

        self.layer5 = nn.Linear(hidden_size, output_size)

        self.dropout = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()

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

    def train_model(self, train_loader, val_loader) -> None:
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
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=self.config_data["modelParams"]["lr"])
        epochs = self.config_data["modelParams"]["epochs"]

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

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

            # Validation
            val_loss, r2 = self.evaluate(val_loader=val_loader, criterion=criterion)
            print(f"Validation Loss after epoch {epoch+1}: {val_loss:.4f}")
            print(f"Validation R² after epoch {epoch+1}: {r2:.4f}")

        # Save model
        model_name = str(self.config_data['Dataset']['name']).split('.', maxsplit=1)[0]
        torch.save(
            self.state_dict(),
            f"{ConfigPaths().folder_model()}/{model_name}.pth"
        )

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
        self.eval()  # Set model to evaluation mode
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


if __name__ == "__main__":
    ...
