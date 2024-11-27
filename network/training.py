from .data_loader import get_dataloaders

import torch
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

def train_epoch(model, optimizer, scheduler, criterion, train_loader, epoch, device):
    loss_history = []
    accuracy_history = []
    lr_history = []

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        pred = output.argmax(dim=1, keepdim=True)
        accuracy_float = pred.eq(target.view_as(pred)).sum().item() / len(target)
        loss_float = loss.item()
        loss_history.append(loss_float)
        accuracy_history.append(accuracy_float)
        lr_history.append(scheduler.get_last_lr()[0])
        if batch_idx % (len(train_loader.dataset) // len(data) // 10) == 0: # len(train_loader.dataset) // len(data) = number of batches in train_loader
            print(
                f"Train Epoch: {epoch}-{batch_idx:03d} "  # formats batch_idx as a three-digit integer, adding leading zeros if necessary (e.g., 001, 010)
                f"batch_loss={loss_float:0.2e} " # Formats loss_float in scientific notation with 2 decimal places (e.g., 1.23e-03)
                f"batch_acc={accuracy_float:0.3f} " # Formats accuracy_float as a fixed-point decimal with 3 decimal places (e.g., 0.975)
                f"lr={scheduler.get_last_lr()[0]:0.3e} " # Formats the learning rate in scientific notation with 3 decimal places (e.g., 1.000e-03)
            )

    return loss_history, accuracy_history, lr_history


@torch.no_grad()
def validate(model, device, val_loader, criterion):
    model.eval()  # Important: eval mode (affects dropout, batch norm etc)
    test_loss = 0
    correct = 0
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += criterion(output, target).item() * len(data) # allows the case when batch sizes vary, for example for the last batch (truncated)
        pred = output.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_loader.dataset)

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            test_loss,
            correct,
            len(val_loader.dataset),
            100.0 * correct / len(val_loader.dataset),
        )
    )
    return test_loss, correct / len(val_loader.dataset)


@torch.no_grad()
def get_predictions(model, device, val_loader, criterion, num=None):
    model.eval()
    points = []
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        pred = output.argmax(dim=1, keepdim=True)

        data = np.split(data.cpu().numpy(), len(data))
        loss = np.split(loss.cpu().numpy(), len(data))
        pred = np.split(pred.cpu().numpy(), len(data))
        target = np.split(target.cpu().numpy(), len(data))
        points.extend(zip(data, loss, pred, target)) # --> <=> ((data[i], loss[i], pred[i], target[i]) for i in range(len(data))

        if num is not None and len(points) > num:
            break

    return points


def run_training(
    model_factory,
    num_epochs,
    optimizer_kwargs,
    data_kwargs,
    device="cuda",
):
    # ===== Data Loading =====
    train_loader, val_loader = get_dataloaders(**data_kwargs)

    # ===== Model, Optimizer and Criterion =====
    model = model_factory()
    model = model.to(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    criterion = torch.nn.functional.cross_entropy
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=(len(train_loader.dataset) * num_epochs) // train_loader.batch_size, # <=> number of batches x number of epochs
    )

    # ===== Train Model =====
    lr_history = []
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc, lrs = train_epoch(
            model, optimizer, scheduler, criterion, train_loader, epoch, device
        )
        train_loss_history.extend(train_loss)
        train_acc_history.extend(train_acc)
        lr_history.extend(lrs)

        val_loss, val_acc = validate(model, device, val_loader, criterion)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

    # ===== Plot training curves =====
    n_train = len(train_acc_history)
    t_train = num_epochs * np.arange(n_train) / n_train
    t_val = np.arange(1, num_epochs + 1)

    fig, axes = plt.subplots(1,3, figsize=(6.4 * 3, 4.8))
    axes[0].plot(t_train, train_acc_history, label="Train")
    axes[0].plot(t_val, val_acc_history, label="Val")
    axes[0].legend()
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")

    axes[1].plot(t_train, train_loss_history, label="Train")
    axes[1].plot(t_val, val_loss_history, label="Val")
    axes[1].legend()
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")

    axes[2].plot(t_train, lr_history)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")

    # ===== Plot low/high loss predictions on validation set =====
    points = get_predictions(
        model,
        device,
        val_loader,
        partial(torch.nn.functional.cross_entropy, reduction="none"),
    )
    points.sort(key=lambda x: x[1]) # sort by increasing loss
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    for k in range(5):
        axes[k].imshow(points[k][0][0, 0], cmap="gray") # low loss
        axes[k].set_title(f"true={int(points[k][3])} pred={int(points[k][2])}")
        axes[5+k].imshow(points[-k - 1][0][0, 0], cmap="gray") # high loss
        axes[5+k].set_title(f"true={int(points[-k-1][3])} pred={int(points[-k-1][2])}")

    return sum(train_acc) / len(train_acc), val_acc