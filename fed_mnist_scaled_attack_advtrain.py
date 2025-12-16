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
NUM_CLIENTS = 16
NUM_ROUNDS = 10
LOCAL_EPOCHS = 10
BATCH_SIZE = 64
LR = 0.01

# Malicious behavior
MALICIOUS_CLIENTS = [3, 4]
ATTACK_SCALE = 10.0  # scale factor for scaled-up attack

# Adversarial training settings (client-side, for benign clients)
USE_ADV_TRAINING = True
FGSM_EPS_RAW = 0.3  # epsilon in [0,1] pixel space (standard MNIST value)

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
mnist_mean = 0.1307
mnist_std = 0.3081

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mnist_mean,), (mnist_std,))
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
# Standard local training (no adversarial examples)
# -------------------------
def local_train_clean(model, dataloader, epochs=1, lr=0.01, device="cpu"):
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
# Local adversarial training (FGSM on benign clients)
# -------------------------
def local_train_adv(model, dataloader, epochs=1, lr=0.01, device="cpu"):
    """
    Simple FGSM adversarial training:
    - For each batch, generate FGSM adversarial examples in pixel space,
      map back to normalized space, and train on both clean and adversarial data.
    """
    model = model.to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    mean = torch.tensor(mnist_mean, device=device).view(1, 1, 1, 1)
    std = torch.tensor(mnist_std, device=device).view(1, 1, 1, 1)

    # Convert pixel-space epsilon to normalized-space epsilon
    eps_norm = FGSM_EPS_RAW / mnist_std

    # Normalized limits corresponding to pixel range [0,1]
    min_norm = (0.0 - mnist_mean) / mnist_std
    max_norm = (1.0 - mnist_mean) / mnist_std

    for _ in range(epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            # Ensure we can take gradient w.r.t. input
            x.requires_grad_(True)

            # Forward on clean
            logits_clean = model(x)
            loss_clean = criterion(logits_clean, y)
            loss_clean.backward(retain_graph=True)

            # FGSM adversarial example in normalized space
            grad = x.grad.data.sign()
            x_adv = x + eps_norm * grad
            x_adv = torch.clamp(x_adv, min=min_norm, max=max_norm)

            # Detach adversarial examples for second forward
            x_adv = x_adv.detach()

            # Forward on adversarial
            logits_adv = model(x_adv)
            loss_adv = criterion(logits_adv, y)

            # Combine losses (simple 50/50 mix)
            loss = 0.5 * (loss_clean + loss_adv)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model.state_dict()


# -------------------------
# FedAvg aggregation (weighted)
# -------------------------
def fed_avg(state_dicts, weights):
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
    global_state: state_dict of global model BEFORE local training
    local_state:  state_dict AFTER local training
    scale:        attack scale factor
    Returns poisoned state_dict where update is scaled up.
    """
    poisoned = {}
    for k in global_state.keys():
        update = local_state[k] - global_state[k]
        poisoned[k] = global_state[k] + scale * update
    return poisoned


# -------------------------
# Federated training loop with:
# - malicious scaled-up attackers
# - benign clients using adversarial training
# -------------------------
global_model = SimpleMLP().to(DEVICE)
global_acc_history = []

for rnd in range(1, NUM_ROUNDS + 1):
    print(f"\n=== Round {rnd}/{NUM_ROUNDS} ===")

    client_states = []
    client_weights = []

    # Global model snapshot before local training
    base_global_state = copy.deepcopy(global_model.state_dict())

    for cid, loader in enumerate(client_loaders):
        print(f"  Client {cid} local training...")

        # Start each client from the same global model
        client_model = SimpleMLP().to(DEVICE)
        client_model.load_state_dict(copy.deepcopy(base_global_state))

        if cid in MALICIOUS_CLIENTS:
            # Malicious clients: normal local training, then scaled update
            print(f"    -> Client {cid} is MALICIOUS (scaled-up attack, scale={ATTACK_SCALE})")
            local_state = local_train_clean(
                client_model,
                loader,
                epochs=LOCAL_EPOCHS,
                lr=LR,
                device=DEVICE
            )
            attacked_state = apply_scaled_attack(base_global_state, local_state, ATTACK_SCALE)
            client_states.append(attacked_state)
        else:
            # Benign clients: adversarial training (FGSM)
            if USE_ADV_TRAINING:
                print(f"    -> Client {cid} is BENIGN (FGSM adversarial training)")
                local_state = local_train_adv(
                    client_model,
                    loader,
                    epochs=LOCAL_EPOCHS,
                    lr=LR,
                    device=DEVICE
                )
            else:
                print(f"    -> Client {cid} is BENIGN (clean training)")
                local_state = local_train_clean(
                    client_model,
                    loader,
                    epochs=LOCAL_EPOCHS,
                    lr=LR,
                    device=DEVICE
                )
            client_states.append(local_state)

        client_weights.append(len(loader.dataset))

    # Server aggregates with FedAvg
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
plt.plot(
    rounds,
    [a * 100 for a in global_acc_history],
    marker="o",
    label="With 2 malicious clients + FGSM adversarial training (benign clients)"
)
plt.xlabel("Round")
plt.ylabel("Global Test Accuracy (%)")
plt.title("Federated Learning on MNIST with Scaled-Up Attack\nBenign Clients Use FGSM Adversarial Training")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
