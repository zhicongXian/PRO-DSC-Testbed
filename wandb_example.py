import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import wandb

# Dummy binary classification dataset
def make_data(n=1000):
    X = torch.randn(n, 20)
    y = (X[:, :5].sum(dim=1) > 0).long()
    return TensorDataset(X, y)

class Net(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        return self.net(x)

def train():
    with wandb.init() as run:
        config = wandb.config

        dataset = make_data()
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

        model = Net(hidden_dim=config.hidden_dim, dropout=config.dropout)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        for epoch in range(config.epochs):
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for xb, yb in loader:
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * xb.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

            avg_loss = total_loss / total
            acc = correct / total

            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "train_accuracy": acc
            })

# Sweep config
sweep_config = {
    "method": "random",
    "metric": {
        "name": "train_loss",
        "goal": "minimize"
    },
    "parameters": {
        "learning_rate": {"min": 1e-4, "max": 1e-2},
        "batch_size": {"values": [32, 64, 128]},
        "hidden_dim": {"values": [32, 64, 128]},
        "dropout": {"values": [0.1, 0.2, 0.4]},
        "epochs": {"value": 5}
    }
}

if __name__ == "__main__":
    wandb.login()

    sweep_id = wandb.sweep(sweep=sweep_config, project="pytorch-sweep-demo")
    wandb.agent(sweep_id, function=train, count=10)