import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import time

# Define the enhanced neural network architecture
class EnhancedNN(nn.Module):
    def __init__(self, input_size, num_classes, hidden_layers=[256, 128, 64], dropout_rate=0.5):
        super(EnhancedNN, self).__init__()
        layers = []
        input_dim = input_size

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Function to log time
def log_time(start_time, step):
    elapsed_time = time.time() - start_time
    print(f'{step} took {elapsed_time // 60:.0f} minutes and {elapsed_time % 60:.0f} seconds')

# Start time
start_time = time.time()

print("Loading data...")
# Load and preprocess the data
train_data = np.loadtxt('traindata.txt', delimiter=',')
train_labels = np.loadtxt('trainlabels.txt')
log_time(start_time, 'Loading data')

print("Normalizing data...")
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
log_time(start_time, 'Data normalization')

print("Splitting data...")
X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=42, stratify=train_labels)
log_time(start_time, 'Data splitting')

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)

# Hyperparameter grid
param_dist = {
    'hidden_layers': [[256, 128, 64], [512, 256, 128], [128, 64, 32]],
    'dropout_rate': [0.3, 0.5, 0.7],
    'learning_rate': [0.01, 0.001, 0.0001],
    'batch_size': [16, 32, 64],
    'epochs': [100, 200, 300]
}

def create_model(hidden_layers, dropout_rate, learning_rate):
    model = EnhancedNN(input_size=X_train.shape[1], num_classes=len(np.unique(train_labels)), hidden_layers=hidden_layers, dropout_rate=dropout_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer

best_val_loss = float('inf')
best_model = None

# Random search
num_trials = 20
for trial in range(num_trials):
    hidden_layers = param_dist['hidden_layers'][np.random.choice(len(param_dist['hidden_layers']))]
    dropout_rate = np.random.choice(param_dist['dropout_rate'])
    learning_rate = np.random.choice(param_dist['learning_rate'])
    batch_size = np.random.choice(param_dist['batch_size'])
    epochs = np.random.choice(param_dist['epochs'])
    params = {
        'hidden_layers': hidden_layers,
        'dropout_rate': dropout_rate,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs': epochs
    }
    print(f"Trial {trial + 1}/{num_trials}: {params}")
    
    model, optimizer = create_model(params['hidden_layers'], params['dropout_rate'], params['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    # Train the model
    num_epochs = params['epochs']
    grad_clip = 1.0
    patience = 20
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        permutation = torch.randperm(X_train.size()[0])
        for i in range(0, X_train.size()[0], batch_size):
            indices = permutation[i:i + batch_size]
            batch_X, batch_y = X_train[indices], y_train[indices]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        # Validate the model
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve == patience:
            print(f'Early stopping after {epoch + 1} epochs')
            break
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

log_time(start_time, 'Randomized hyperparameter search')

# Load the best model
best_model.load_state_dict(torch.load('best_model.pth'))

# Validate the best model
print("Validating the model...")
best_model.eval()
with torch.no_grad():
    outputs = best_model(X_val)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = accuracy_score(y_val, predicted)
    print(f'Validation Accuracy: {accuracy}')

log_time(start_time, 'Model validation')

print("Saving scaler...")
joblib.dump(scaler, 'scaler.pkl')
log_time(start_time, 'Saving model')

print("Training complete.")
