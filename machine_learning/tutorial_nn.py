import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from data.parameters_1d_function import *


# Scale the features using StandardScaler
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)
X_plot_scaled  = scaler.transform(X_plot)

# Convert numpy arrays to PyTorch tensors (ensure targets are column vectors)
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor  = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor  = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
X_plot_tensor  = torch.tensor(X_plot_scaled, dtype=torch.float32)


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim=1, neurons1=10, neurons2=10, output_dim=1, activation='relu'):
        """
        Initializes a neural network with one input layer, two hidden layers, and one output neuron.
        
        Parameters:
            input_dim (int): Number of input features.
            neurons1 (int): Number of neurons in the first hidden layer.
            neurons2 (int): Number of neurons in the second hidden layer.
            activation (str): Activation function to use ('relu' or 'tanh').
        """
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, neurons1)
        self.fc2 = nn.Linear(neurons1, neurons2)
        self.out = nn.Linear(neurons2, output_dim)
        
        # Set activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()  # Default
        
    def forward(self, x):
        x = self.activation(self.fc1(x))  # First hidden layer
        x = self.activation(self.fc2(x))  # Second hidden layer
        x = self.out(x)                   # Output layer
        return x



def train_model(model, criterion, optimizer, X_train, y_train, X_val, y_val, epochs=130, patience=30):
    """
    Trains the model using the provided data, and implements early stopping.
    
    Parameters:
        model (nn.Module): The neural network model.
        criterion: Loss function.
        optimizer: Optimizer for training.
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        epochs (int): Maximum number of epochs.
        patience (int): Number of epochs with no improvement to wait before stopping.
    
    Returns:
        train_losses, val_losses: Lists containing the loss values for each epoch.
    """
    train_losses = []
    val_losses = []
    best_val_loss = np.inf
    best_epoch = 0
    best_model_state = None

    for epoch in range(epochs):
        #Training Phase
        model.train()
        optimizer.zero_grad()                # Clear gradients
        outputs = model(X_train)             # Forward pass
        loss = criterion(outputs, y_train)   # Compute training loss
        loss.backward()                      # Backward pass
        optimizer.step()                     # Update parameters
        train_losses.append(loss.item())
        # Validation Phase #TO DO: Complete the validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val) 
            val_loss = criterion(val_outputs, y_val)
            val_losses.append(val_loss.item())

        # Print progress
        print(f"Epoch {epoch+1}/{epochs}  -  Training Loss: {loss.item():.4f}  -  Validation Loss: {val_loss.item():.4f}")

        # Early Stopping Check
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_epoch = epoch
            best_model_state = model.state_dict()  # Save the best model state
        elif epoch - best_epoch >= patience:
            print("Early stopping triggered")
            break

    # Load the best model state before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return train_losses, val_losses

#TO DO: Define the input and output dimensions
input_dim = 1 # Number of input features
output_dim = 1# Number of output neurons

# Define network architecture parameters
neurons1 = 200
neurons2 = 10

# Create the model instance
model = NeuralNetwork(input_dim=input_dim, neurons1=neurons1, neurons2=neurons2,output_dim=output_dim, activation='relu')

# Define Mean Squared Error Loss and the Adam optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 500  # Maximum number of epochs
patience = 30 # Early stopping patience

train_losses, val_losses = train_model(model, criterion, optimizer,
                                         X_train_tensor, y_train_tensor,
                                         X_test_tensor, y_test_tensor,
                                         epochs=epochs, patience=patience)

plt.figure()
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


model.eval()
with torch.no_grad():
    y_pred_plot = model(X_plot_tensor).numpy()

# Plot training data, model predictions, and ground truth function
fig, axs = plt.subplots(1, 1, figsize=(8, 5))
axs.plot(x_train, y_train, 'ro', label='Training data')
axs.plot(x_plot, y_pred_plot, label='Prediction')
axs.plot(x_plot, f(x_plot), label=r'Ground truth: $f(x)=x\,\sin(x)$')
axs.set_xlabel("x")
axs.set_ylabel("y")
axs.legend(loc="lower right")
plt.show()
#Test Set Prediction
with torch.no_grad():
    y_test_pred = model(X_test_tensor).numpy()

# Plot predicted vs true test values
plt.figure(figsize=(6, 6))
plt.plot(y_test, y_test, '-', label='$y_{true}$')
plt.plot(y_test, y_test_pred, 'r.', label='$\hat{y}$')
plt.xlabel('$y_{true}$')
plt.ylabel('$\hat{y}$')
plt.legend(loc='upper left')
plt.show()

mse = mean_squared_error(y_test, y_test_pred)
r2  = r2_score(y_test, y_test_pred)
print("Mean squared error: %.2f" % mse)
print("Variance score (R^2): %.2f" % r2)

plt.show()