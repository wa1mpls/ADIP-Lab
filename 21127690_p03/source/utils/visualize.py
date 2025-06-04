import matplotlib.pyplot as plt
import pandas as pd
import json

def plot_loss(csv_file, output_path="loss_plot.png"):
    data = pd.read_csv(csv_file)

    # Lọc các dòng mà giá trị trong cột "epoch" là số nguyên
    valid_data = data[data["epoch"].astype(str).str.isdigit()]
    
    epochs = valid_data["epoch"].astype(int)
    losses = valid_data["loss"].astype(float)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def plot_accuracy_comparison(json_files, output_path="accuracy_comparison.png"):
    accuracies = []
    labels = []

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            accuracies.append(data["accuracy"])
            labels.append(f"patch={data['params']['patch_size']}, layers={data['params']['layers']}")

    plt.figure(figsize=(10, 6))
    plt.bar(labels, accuracies)
    plt.xlabel("Experiment")
    plt.ylabel("Top-1 Accuracy (%)")
    plt.title("Accuracy Comparison Across Experiments")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
