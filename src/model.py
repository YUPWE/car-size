"""ResNet-18 transfer learning: build, train, predict."""
import copy
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from .data import CLASSES


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_resnet18(freeze_backbone, pretrained=True):
    """ResNet-18 pretrained on ImageNet, with a new head (one output per class: small, big).
    freeze_backbone=True  -> only the new head learns (fast, the baseline)
    freeze_backbone=False -> every layer is fine-tuned"""
    from torchvision.models import resnet18, ResNet18_Weights
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))  # new head, always trainable
    return model


@torch.no_grad()
def predict_probs(model, loader, device):
    """Softmax probabilities for every image in the loader, plus the true labels."""
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x.to(device)), dim=1).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def train(model, train_loader, val_loader, epochs, lr, device, class_weights=None, weight_decay=1e-4):
    """Standard training loop. Keeps the weights from the epoch with the best validation macro-F1."""
    model.to(device)
    weight = None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
    best_f1, best_state = -1.0, None
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)               # forward
            loss = loss_fn(outputs, y)       # how wrong
            optimizer.zero_grad()            # clear old gradients
            loss.backward()                  # compute gradients
            optimizer.step()                 # update weights
            loss_sum += loss.item() * len(y)
            correct += (outputs.argmax(1) == y).sum().item()
            total += len(y)
        scheduler.step()

        # validation
        model.eval()
        v_total, v_correct, v_loss, preds, trues = 0, 0, 0.0, [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                v_loss += loss_fn(out, y).item() * len(y)
                p = out.argmax(1)
                v_correct += (p == y).sum().item()
                v_total += len(y)
                preds.append(p.cpu())
                trues.append(y.cpu())
        val_f1 = f1_score(torch.cat(trues), torch.cat(preds), average="macro")

        history["train_loss"].append(loss_sum / total)
        history["train_acc"].append(correct / total)
        history["val_loss"].append(v_loss / v_total)
        history["val_acc"].append(v_correct / v_total)
        history["val_f1"].append(val_f1)
        print(f"epoch {epoch + 1:2d}/{epochs} | train loss {loss_sum / total:.3f} acc {correct / total:.3f} "
              f"| val loss {v_loss / v_total:.3f} acc {v_correct / v_total:.3f} F1 {val_f1:.3f} "
              f"| {time.time() - t0:.0f}s")
        if val_f1 > best_f1:
            best_f1, best_state = val_f1, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    print(f"Best validation macro-F1: {best_f1:.3f}")
    return model, history
