# train.py

import torch
import torch.nn as nn
import torch.optim as optim

from model import NarutoModel
from dataset import naruto_loader
from config import epochs, learning_rate, device, train_dir, val_dir


# ----------------------------
# Training + Validation Logic
# ----------------------------
def train_model(model, num_epochs, train_loader, loss_fn, optimizer):
    """
    Trains the model using training data and evaluates on validation data.
    Saves the best model based on validation accuracy.
    """

    # Move model to CPU/GPU
    model = model.to(device)

    # Load validation data internally
    val_loader = naruto_loader(val_dir, train=False)

    best_acc = 0  # Track best validation accuracy

    for epoch in range(num_epochs):

        # -------------------
        # TRAINING PHASE
        # -------------------
        model.train()
        train_loss = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # -------------------
        # VALIDATION PHASE
        # -------------------
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)

                # Count correct predictions
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        # Compute accuracy
        acc = 100 * correct / total

        # -------------------
        # SAVE BEST MODEL
        # -------------------
        if acc > best_acc:
            best_acc = acc

            # Save model in required submission format
            torch.save(model.state_dict(), "checkpoints/final_weights.pth")

        # Print training progress
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {train_loss:.4f} | Val Acc: {acc:.2f}%")

    return model


# ----------------------------
# Main Entry Point
# ----------------------------
def main():
    """
    Entry point for training script.
    Initializes model, loads data, and starts training.
    """

    print("Starting training pipeline...")

    # Initialize model
    model = NarutoModel()

    # Load datasets
    train_loader = naruto_loader(train_dir, train=True)
    # val_loader is created internally inside train_model

    # Loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    train_model(model, epochs, train_loader, loss_fn, optimizer)

    print("Training complete.")
    print("Model saved at: checkpoints/final_weights.pth")


# ----------------------------
# Script Execution
# ----------------------------
if __name__ == "__main__":
    main()
