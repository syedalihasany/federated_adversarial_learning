import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# -------------------------
# Config
# -------------------------
NUM_CLIENTS = 5
NUM_ROUNDS = 10
LOCAL_EPOCHS = 10
BATCH_SIZE = 64
LR = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# -------------------------
# Simple MLP model
# -------------------------
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


# -------------------------
# Data: MNIST & split into clients
# -------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# Split train dataset into 5 (almost) equal parts
total_len = len(train_dataset)
lengths = [total_len // NUM_CLIENTS] * NUM_CLIENTS
for i in range(total_len % NUM_CLIENTS):
    lengths[i] += 1

client_datasets = random_split(train_dataset, lengths)

client_loaders = [
    DataLoader(cd, batch_size=BATCH_SIZE, shuffle=True) for cd in client_datasets
]

test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


# -------------------------
# Local training on one client
# -------------------------
def local_train(model, dataloader, epochs=1, lr=0.01, device="cpu"):
    model = model.to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    return model.state_dict()


# -------------------------
# FedAvg aggregation
# -------------------------
def fed_avg(state_dicts, weights):
    """
    state_dicts: list of client state_dicts
    weights:     list of dataset sizes (one per client)
    """
    avg_state = copy.deepcopy(state_dicts[0])
    total_weight = float(sum(weights))

    for k in avg_state.keys():
        avg_state[k] = sum(
            (w / total_weight) * state_dicts[i][k]
            for i, w in enumerate(weights)
        )

    return avg_state


# -------------------------
# Evaluation on global test set
# -------------------------
def evaluate(model, dataloader, device="cpu"):
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total


# -------------------------
# Federated training loop
# -------------------------
global_model = SimpleMLP().to(DEVICE)
global_acc_history = []

for rnd in range(1, NUM_ROUNDS + 1):
    print(f"\n=== Round {rnd}/{NUM_ROUNDS} ===")

    # 1) Broadcast global model to each client and train locally
    client_states = []
    client_weights = []

    for cid, loader in enumerate(client_loaders):
        print(f"  Client {cid} local training...")
        client_model = SimpleMLP().to(DEVICE)
        client_model.load_state_dict(copy.deepcopy(global_model.state_dict()))

        state_dict = local_train(
            client_model,
            loader,
            epochs=LOCAL_EPOCHS,
            lr=LR,
            device=DEVICE
        )

        client_states.append(state_dict)
        client_weights.append(len(loader.dataset))

    # 2) Server aggregates via FedAvg
    new_global_state = fed_avg(client_states, client_weights)
    global_model.load_state_dict(new_global_state)

    # 3) Evaluate global model
    acc = evaluate(global_model, test_loader, DEVICE)
    global_acc_history.append(acc)
    print(f"  Global test accuracy after round {rnd}: {acc * 100:.2f}%")

# -------------------------
# Plot accuracy over rounds
# -------------------------
plt.figure()
plt.plot(range(1, NUM_ROUNDS + 1),
         [a * 100 for a in global_acc_history],
         marker="o")
plt.xlabel("Round")
plt.ylabel("Global Test Accuracy (%)")
plt.title("Federated Learning on MNIST (5 clients, FedAvg)")
plt.grid(True)
plt.tight_layout()
plt.show()
