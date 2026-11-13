import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from model import get_model

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if __name__ == "__main__":
    data_dir = "sneakers-dataset"
    full_train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    full_val_dataset = datasets.ImageFolder(root=data_dir, transform=val_transform)

    with open("classes.json", "w", encoding="utf-8") as f:
        json.dump(full_train_dataset.classes, f, ensure_ascii=False, indent=2)

    targets = full_train_dataset.targets
    train_idx, val_idx = train_test_split(
        range(len(targets)), test_size=0.2, stratify=targets, random_state=42
    )

    train_dataset = Subset(full_train_dataset, train_idx)
    val_dataset = Subset(full_val_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    print(f"Всего классов: {len(full_train_dataset.classes)}")
    print(f"Картинок в train: {len(train_dataset)}, в val: {len(val_dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=len(full_train_dataset.classes)).to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 10
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for x_train, y_train in train_loader:
            x_train, y_train = x_train.to(device), y_train.to(device)

            optimizer.zero_grad()
            predict = model(x_train)
            loss = loss_func(predict, y_train)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                outputs = model(x_val)
                _, predicted = torch.max(outputs, 1)
                total += y_val.size(0)
                correct += (predicted == y_val).sum().item()

        val_acc = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{epochs}] | Loss: {train_loss / len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")