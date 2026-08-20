import os
import sys
import time
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm
import timm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# CONFIGURATIONS
# ==============================================================================
CSV_PATH = r""
DATA_DIR = r""
OUTPUT_DIR = r"./result"

BATCH_SIZE = 16
LEARNING_RATE = 5e-6
EPOCHS = 100
PATIENCE = 20
NUM_EXPERTS = 16
USE_CACHE = True
USE_PRETRAINED_WEIGHTS = True

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from soft_moe import soft_moe_vit_base


# ==============================================================================
# 1. DATASET & AUGMENTATION
# ==============================================================================
class PotatoDataset(Dataset):
    def __init__(self, data, transform=None, use_cache=True):
        self.paths = []
        self.labels = []
        self.transform = transform
        self.use_cache = use_cache
        self.cache = []

        resize_transform = transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BILINEAR
        )

        if self.use_cache:
            print(f"Loading {len(data)} images into RAM...")
            for path, label in tqdm(zip(data["Path"], data["Label"]), total=len(data)):
                try:
                    img = Image.open(path).convert("RGB")
                    img = resize_transform(img)
                    self.cache.append(img)
                    self.paths.append(path)
                    self.labels.append(label)
                except Exception as e:
                    print(f"Error reading image: {path} - {e}")
                    continue
            print(f"Successfully loaded {len(self.cache)} images into RAM.")
        else:
            self.paths = data["Path"].tolist()
            self.labels = data["Label"].tolist()

    def __getitem__(self, index):
        label = torch.tensor(self.labels[index]).long()

        if self.use_cache:
            img = self.cache[index]
        else:
            path = self.paths[index]
            try:
                img = Image.open(path).convert("RGB")
                img = transforms.Resize(
                    (224, 224),
                    interpolation=transforms.InterpolationMode.BILINEAR
                )(img)
            except Exception as e:
                print(f"Failed to read image: {path}")
                raise e

        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.labels)


# ==============================================================================
# 2. TRAINING LOOP
# ==============================================================================
def train_model(data_loader, model, criterion, optimizer, num_epochs, device, early_stop=True, patience=10):
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = nn.DataParallel(model)
    model = model.to(device)

    since = time.time()

    best_model_wts = copy.deepcopy(
        model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    )

    best_val_loss = float('inf')
    best_epoch = 0
    wait = 0

    history = {'Epoch': [], 'Train_Loss': [], 'Train_Acc': [], 'Validation_Loss': [], 'Validation_Acc': [], 'Time': []}
    result_current = {
        "Train_Loss": None,
        "Train_Acc": None,
        "Validation_Loss": None,
        "Validation_Acc": None
    }

    for epoch in range(num_epochs):
        print("-----------------------------------------------------------------------")
        print(f"Epoch {epoch+1}/{num_epochs}")
        epoch_start = time.time()

        for phase in ["Train", "Validation"]:
            if phase == "Train":
                model.train()
            else:
                model.eval()

            running_loss, running_correct = 0.0, 0
            total_samples = 0

            for images, labels in tqdm(data_loader[phase], desc=f"{phase} Phase"):
                images = images.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(phase == "Train"):
                    outputs = model(images)
                    _, predicts = torch.max(outputs, dim=1)
                    loss = criterion(outputs, labels)

                    if phase == "Train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * images.size(0)
                running_correct += torch.sum(predicts == labels).item()
                total_samples += images.size(0)

            epoch_loss = running_loss / total_samples
            epoch_acc = running_correct / total_samples

            result_current[f"{phase}_Loss"] = epoch_loss
            result_current[f"{phase}_Acc"] = epoch_acc

            if phase == "Validation":
                history['Epoch'].append(epoch + 1)
                history['Train_Loss'].append(result_current['Train_Loss'])
                history['Train_Acc'].append(result_current['Train_Acc'])
                history['Validation_Loss'].append(epoch_loss)
                history['Validation_Acc'].append(epoch_acc)

                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_model_wts = copy.deepcopy(
                        model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                    )
                    best_epoch = epoch
                    wait = 0
                else:
                    wait += 1

        epoch_duration = time.time() - epoch_start
        history["Time"].append(epoch_duration)
        print(f"Train Loss: {result_current['Train_Loss']:.4f}, Train Acc: {result_current['Train_Acc']:.4f}")
        print(f"Valid Loss: {result_current['Validation_Loss']:.4f}, Valid Acc: {result_current['Validation_Acc']:.4f}")
        print(f"Epoch {epoch+1} finished in {epoch_duration:.2f}s")

        if early_stop and wait >= patience:
            print(f"Early stopping at epoch {epoch+1} (no improvement in {patience} epochs).")
            break

    print("-----------------------------------------------------------------------")
    time_elapse = time.time() - since
    print(f"Training completed in {time_elapse:.2f}s | Best Val Loss: {best_val_loss:.4f} at Epoch {best_epoch+1}")

    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(best_model_wts)
        model = model.module
    else:
        model.load_state_dict(best_model_wts)

    return model, pd.DataFrame(history), time_elapse, best_val_loss, best_epoch


# ==============================================================================
# 3. MAIN EXECUTION
# ==============================================================================
def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"CSV Path: {CSV_PATH}")
    print(f"Data Folder: {DATA_DIR}")

    df = pd.read_csv(CSV_PATH)
    df["Path"] = df.apply(lambda row: os.path.join(DATA_DIR, str(row["Label"]), str(row["Path"])), axis=1)

    df_train = df.loc[df["Type"] == "Train"].copy().reset_index(drop=True)
    df_val = df.loc[df["Type"].isin(["Validation", "Val"])].copy().reset_index(drop=True)
    df_test = df.loc[df["Type"] == "Test"].copy().reset_index(drop=True)

    print(f"Dataset split counts: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

    label_encoder = LabelEncoder()
    df_train["Label"] = label_encoder.fit_transform(df_train["Label"])
    df_val["Label"] = label_encoder.transform(df_val["Label"])
    df_test["Label"] = label_encoder.transform(df_test["Label"])

    classes = list(label_encoder.classes_)
    num_classes = len(classes)
    print(f"Target classes ({num_classes}): {classes}")

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    transform = {
        "Train": transforms.Compose([
            transforms.ColorJitter(brightness=0.15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        "Validation": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        "Test": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }

    train_dataset = PotatoDataset(data=df_train, transform=transform["Train"], use_cache=USE_CACHE)
    valid_dataset = PotatoDataset(data=df_val, transform=transform["Validation"], use_cache=USE_CACHE)
    test_dataset = PotatoDataset(data=df_test, transform=transform["Test"], use_cache=USE_CACHE)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    data_loader = {"Train": train_dataloader, "Validation": valid_dataloader}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Initializing SoftMoEVisionTransformer-Base (num_classes={num_classes}, num_experts={NUM_EXPERTS}, embed_dim=768)...")
    model = soft_moe_vit_base(
        num_classes=num_classes,
        img_size=224,
        num_experts=NUM_EXPERTS,
        slots_per_expert=1,
        moe_layer_index=6
    )

    if USE_PRETRAINED_WEIGHTS:
        try:
            print("Loading pre-trained ImageNet-21k ViT-Base weights from timm...")
            pretrained_vit = timm.create_model('vit_base_patch16_224.augreg_in21k_ft_in1k', pretrained=True)
            pretrained_dict = pretrained_vit.state_dict()
            model_dict = model.state_dict()

            matched_dict = {
                k: v for k, v in pretrained_dict.items() 
                if k in model_dict and v.shape == model_dict[k].shape
            }

            model_dict.update(matched_dict)
            model.load_state_dict(model_dict)

            loaded_param_count = sum(v.numel() for v in matched_dict.values())
            total_param_count = sum(p.numel() for p in model.parameters())

            print("Successfully loaded pre-trained ImageNet weights for backbone and attention layers.")
            print(f"Inherited ImageNet parameters: {loaded_param_count:,} / {total_param_count:,} ({loaded_param_count/total_param_count*100:.1f}%)")
        except Exception as e:
            print(f"Failed to load pre-trained weights ({e}). Falling back to random initialization.")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    trained_model, history_df, time_elapse, best_val_loss, best_epoch = train_model(
        data_loader=data_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=EPOCHS,
        device=device,
        early_stop=True,
        patience=PATIENCE
    )

    model_save_path = os.path.join(OUTPUT_DIR, "best_softmoe_potato_model.pth")
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"Model weights saved to: {model_save_path}")

    history_save_path = os.path.join(OUTPUT_DIR, "history_softmoe.csv")
    history_df.to_csv(history_save_path, index=False)
    print(f"Training history saved to: {history_save_path}")

    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history_df['Epoch'], history_df['Train_Loss'], label='Train Loss', color='blue', linewidth=2)
    plt.plot(history_df['Epoch'], history_df['Validation_Loss'], label='Validation Loss', color='red', linewidth=2)
    plt.title('Loss Curve', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(history_df['Epoch'], history_df['Train_Acc'], label='Train Accuracy', color='blue', linewidth=2)
    plt.plot(history_df['Epoch'], history_df['Validation_Acc'], label='Validation Accuracy', color='red', linewidth=2)
    plt.title('Accuracy Curve', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    learning_curves_path = os.path.join(OUTPUT_DIR, "learning_curves.png")
    plt.tight_layout()
    plt.savefig(learning_curves_path, dpi=300)
    plt.close()
    print(f"Learning curves saved to: {learning_curves_path}")

    print("\n=======================================================================")
    print("EVALUATING MODEL ON TEST SET")
    print("=======================================================================")
    trained_model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_dataloader, desc="Testing Phase"):
            images = images.to(device)
            outputs = trained_model(images)
            _, predicts = torch.max(outputs, dim=1)
            all_preds.extend(predicts.cpu().numpy())
            all_labels.extend(labels.numpy())

    report_text = classification_report(all_labels, all_preds, target_names=classes, digits=4)
    print("\nClassification Report (4 decimal digits):")
    print(report_text)

    report_save_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(report_save_path, "w", encoding="utf-8") as f:
        f.write("=== CLASSIFICATION REPORT - SOFT-MOE POTATO MODEL ===\n\n")
        f.write(f"Best Epoch: {best_epoch + 1}\n")
        f.write(f"Best Validation Loss: {best_val_loss:.4f}\n\n")
        f.write(report_text)
    print(f"Classification report saved to: {report_save_path}")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix - Soft-MoE Potato Classification', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    cm_save_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to: {cm_save_path}")


if __name__ == "__main__":
    main()
