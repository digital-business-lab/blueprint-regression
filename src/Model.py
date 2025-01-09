""" 
Module Docstring not implemented yet
"""
import torch
import torch.nn as nn
import torch.optim as optim

from src.Config import ConfigYAML, ConfigPaths


class Model(nn.Module, ConfigYAML):
    """Not implemented yet"""
    def __init__(self, input_size: int, hidden_size: int, output_size: int =1, dropout_rate: float =0.3):
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
        """Not implemented yet"""
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
        """Not implemented yet"""
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
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

            # Validation
            val_loss = self.evaluate(val_loader=val_loader, criterion=criterion)
            print(f"Validation Loss after epoch {epoch+1}: {val_loss:.4f}")

        # Save model
        torch.save(
            self.state_dict(), 
            f"{ConfigPaths().folder_model()}/{str(self.config_data["Dataset"]["name"]).split(".")[0]}.pth")

    def evaluate(self, val_loader, criterion =nn.MSELoss()):
        """Not implemented yet"""
        self.eval()
        val_loss = 0.0

        with torch.no_grad(): # No gradients needed for evaluation
            for batch in val_loader:
                inputs, targets = batch
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        return val_loss / len(val_loader)
    
    def predict(self, X):
        """Not implemented yet"""
        self.eval()
        with torch.no_grad():
            predictions = self(X)

        return predictions
                

if __name__ == "__main__":
    ...
