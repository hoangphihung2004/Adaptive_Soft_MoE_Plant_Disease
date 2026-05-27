from model.model import MoE_Classifier
import pandas as pd
from losses.losses import Criterion
from config import Configuration
from dataset.dataset import dataset
from torch import nn
import time
import copy
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score


def drawing(history, save_path):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    axs[0].plot(history['Train_Acc'], label='Train Accuracy')
    axs[0].plot(history['Validation_Acc'], label='Validation Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_title('Training and Validation Accuracy')
    axs[0].legend()

    # Plot Loss
    axs[1].plot(history['Train_Loss'], label='Train Loss')
    axs[1].plot(history['Validation_Loss'], label='Validation Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Training and Validation Loss')
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "plot_loss_acc.png"))




def training(data_loader, model, criterion, optimizer, num_epochs, device, early_stop=True, patience=20):
    model = nn.DataParallel(model)
    model = model.to(device)

    since = time.time()

    best_model_wts = copy.deepcopy(
        model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict())

    best_val_loss = float('inf')
    best_epoch = 0
    wait = 0

    history = {'Train_Loss': [], 'Train_Acc': [], 'Validation_Loss': [], 'Validation_Acc': [], 'Time': []}
    result_current = {"Train_Loss": None,
                      "Train_Acc": None,
                      "Validation_Loss": None,
                      "Validation_Acc": None}

    for epoch in range(num_epochs):

        print("-----------------------------------------------------------------------")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        epoch_start = time.time()

        for phase in ["Train", "Validation"]:
            if phase == "Train":
                model.train()
            else:
                model.eval()

            running_loss, running_correct = 0.0, 0
            total_samples = 0

            for images, labels in tqdm(data_loader[phase], desc="Training And Evaluation"):
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

            history[f"{phase}_Loss"].append(epoch_loss)
            history[f"{phase}_Acc"].append(epoch_acc)

            if phase == "Validation":
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_model_wts = copy.deepcopy(
                        model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict())
                    best_epoch = epoch
                    wait = 0
                else:
                    wait += 1

        epoch_duration = time.time() - epoch_start
        history["Time"].append(epoch_duration)
        print(f"Train Loss: {result_current['Train_Loss']:.4f}, Train Acc: {result_current['Train_Acc']:.4f}")
        print(f"Valid Loss: {result_current['Validation_Loss']:.4f}, Valid Acc: {result_current['Validation_Acc']:.4f}")
        print(f"Epoch {epoch + 1} finished in {epoch_duration:.2f}s")

        if early_stop and wait >= patience:
            print(f"Early stopping at epoch {epoch + 1} (no improvement in {patience} epochs).")
            break

    print("-----------------------------------------------------------------------")
    time_elapse = time.time() - since
    print(f"Training Complete In {time_elapse}s")

    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(best_model_wts)
        model = model.module
    else:
        model.load_state_dict(best_model_wts)

    return model, pd.DataFrame(history), time_elapse, best_val_loss, best_epoch

def evaluation(model, test_dataloader, config, label_encoder, average="macro"):
    model.eval()

    with torch.no_grad():
        all_correct = []
        all_predict = []

        for images, labels in test_dataloader:
            images, labels = images.to(config.device), labels.to(config.device)
            outputs = model(images)
            _, predicts = torch.max(outputs, dim=1)

            all_correct.extend(labels.cpu().numpy())
            all_predict.extend(predicts.cpu().numpy())

        acc = accuracy_score(all_correct, all_predict)
        precision = precision_score(all_correct, all_predict, average=average)
        recall = recall_score(all_correct, all_predict, average=average)
        f1 = f1_score(all_correct, all_predict, average=average)

        result = {
            "Learning_Rate": config.learning_rate,
            "Batch_Size": config.batch_size,
            "Num_Epoch": config.num_epochs,
            "Early_Stop": config.early_stop,
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        }

        report = classification_report(all_correct, all_predict, target_names=label_encoder.classes_, digits=4)

    return result, result

def main():
    config = Configuration()
    df = pd.read_csv(config.data_path)
    data_loader, test_dataloader, label_encoder = dataset(data=df, config=config)
    model = MoE_Classifier(num_classes=len(label_encoder.classes_)).to(config.device)
    criterion = Criterion()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    model, history, time_elapse, best_val_loss, best_epoch = training(data_loader=data_loader,
                                                                      model=model,
                                                                      criterion=criterion,
                                                                      optimizer=optimizer,
                                                                      num_epochs=config.num_epochs,
                                                                      device=config.device,
                                                                      early_stop=config.early_stop,
                                                                      patience=config.patience)

    drawing(history=history, save_path=config.save_path)

    result, report = evaluation(model=model, test_dataloader=test_dataloader, config=config, label_encoder=label_encoder)

    result["Best_Val_Loss"] = best_val_loss
    result["Best_Epoch"] = best_epoch

    result_df = pd.DataFrame([result])
    result_df.to_csv(os.path.join(config.save_path, "result.csv"), index=False)

    with open(os.path.join(config.save_path, "classification_report.txt"), "w") as f:
        f.write(report)

    torch.save(model.state_dict(), os.path.join(config.save_path, f'weights.pth'))
    history.to_csv(os.path.join(config.save_path, "history.csv"), index=False)

if __name__ == "__main__":
    main()



