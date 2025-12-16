import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# -------------------------
# Config
# -------------------------
NUM_CLIENTS = 5
NUM_ROUNDS = 10
LOCAL_EPOCHS = 5        
BATCH_SIZE = 64
LR = 0.01

# Malicious behavior
MALICIOUS_CLIENTS = [3, 4]   # indices of malicious clients
ATTACK_SCALE = 10.0          # scale factor for scaled-up attack

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

# Split train dataset into NUM_CLIENTS parts
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
# FedAvg aggregation (weighted)
# -------------------------
def fed_avg(state_dicts, weights):
    """
    state_dicts: list of client state_dicts
    weights:     list of dataset sizes (one per client)
    """
    total_weight = float(sum(weights))
    avg_state = {}

    for k in state_dicts[0].keys():
        avg_state[k] = torch.zeros_like(state_dicts[0][k])
        for s, w in zip(state_dicts, weights):
            avg_state[k] += (w / total_weight) * s[k]
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
# Apply scaled-up attack to a client's update
# -------------------------
def apply_scaled_attack(global_state, local_state, scale):
    """
    global_state: state_dict of the global model BEFORE local training
    local_state:  state_dict AFTER local training
    scale:        attack scale (e.g., 10.0)
    Returns a poisoned state_dict where the update is scaled up.
    """
    poisoned = {}
    for k in global_state.keys():
        update = local_state[k] - global_state[k]
        poisoned[k] = global_state[k] + scale * update
    return poisoned


# -------------------------
# Federated training loop with malicious clients
# -------------------------
global_model = SimpleMLP().to(DEVICE)
global_acc_history = []

for rnd in range(1, NUM_ROUNDS + 1):
    print(f"\n=== Round {rnd}/{NUM_ROUNDS} ===")

    client_states = []
    client_weights = []

    # Keep a copy of the global state to compute updates
    base_global_state = copy.deepcopy(global_model.state_dict())

    for cid, loader in enumerate(client_loaders):
        print(f"  Client {cid} local training...")

        # Initialize client model from current global
        client_model = SimpleMLP().to(DEVICE)
        client_model.load_state_dict(copy.deepcopy(base_global_state))

        # Local training
        local_state = local_train(
            client_model,
            loader,
            epochs=LOCAL_EPOCHS,
            lr=LR,
            device=DEVICE
        )

        # Benign or malicious behavior
        if cid in MALICIOUS_CLIENTS:
            print(f"    -> Client {cid} is MALICIOUS (scaled-up attack, scale={ATTACK_SCALE})")
            attacked_state = apply_scaled_attack(base_global_state, local_state, ATTACK_SCALE)
            client_states.append(attacked_state)
        else:
            client_states.append(local_state)

        client_weights.append(len(loader.dataset))

    # FedAvg aggregation on server
    new_global_state = fed_avg(client_states, client_weights)
    global_model.load_state_dict(new_global_state)

    # Evaluate global model
    acc = evaluate(global_model, test_loader, DEVICE)
    global_acc_history.append(acc)
    print(f"  Global test accuracy after round {rnd}: {acc * 100:.2f}%")

# -------------------------
# Plot accuracy over rounds
# -------------------------
rounds = range(1, NUM_ROUNDS + 1)

plt.figure()
plt.plot(rounds, [a * 100 for a in global_acc_history],
         marker="o", label="With 2 malicious clients (scaled attack)")
plt.xlabel("Round")
plt.ylabel("Global Test Accuracy (%)")
plt.title("Federated Learning on MNIST with Malicious Clients (Scaled-Up Attack)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
