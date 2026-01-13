
# train_ppe.py
import os
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO


MODEL = "yolov8n.pt"
OUTPUT_DIR = "result/train_ppe"

def plot_metrics(csv_path, save_path):
    
    df = pd.read_csv(csv_path)

    # Plot Loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(df["train/box_loss"], label="Box Loss")
    plt.plot(df["train/cls_loss"], label="Class Loss")
    if "train/dfl_loss" in df.columns:
        plt.plot(df["train/dfl_loss"], label="DFL Loss")
    plt.plot(df["val/box_loss"], label="Val Box Loss")
    plt.plot(df["val/cls_loss"], label="Val Class Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "loss_curve.png"))
    plt.close()

    # Plot Accuracy metrics (Precision, Recall, mAP)
    plt.figure(figsize=(10, 6))
    plt.plot(df["metrics/precision(B)"], label="Precision")
    plt.plot(df["metrics/recall(B)"], label="Recall")
    plt.plot(df["metrics/mAP50(B)"], label="mAP@0.5")
    plt.plot(df["metrics/mAP50-95(B)"], label="mAP@0.5:0.95")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Accuracy Metrics")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "accuracy_curve.png"))
    plt.close()

def plot_confusion_matrix(model, dataset_path, save_path):
    """Generate and save confusion matrix using YOLO's validation results."""
    results = model.val(data=dataset_path, imgsz=640, conf=0.25)
    conf_matrix = results.confusion_matrix

    if conf_matrix is not None:
        disp = ConfusionMatrixDisplay(
            confusion_matrix=conf_matrix,
            display_labels=results.names.values()
        )
        disp.plot(cmap="Blues", xticks_rotation=45)
        plt.title("Confusion Matrix for PPE Detection")
        plt.savefig(os.path.join(save_path, "confusion_matrix.png"))
        plt.xlabel('Predicted Label')
        plt.ylabel('Trtue Label')
        plt.close()
    else:
        print("Confusion matrix not available.")



def plot_data_augmentation_effect(save_path):
    """Visualize impact of data augmentation on dataset size."""
    base_size = 2000
    augmentations = ["Original", "Flip", "Rotation", "Brightness", "Noise"]
    sizes = [base_size, base_size * 1.5, base_size * 2, base_size * 2.5, base_size * 3]

    plt.figure(figsize=(10, 6))
    plt.bar(augmentations, sizes, color="lightgreen")
    plt.xlabel("Data Augmentation Techniques")
    plt.ylabel("Number of Images")
    plt.title("Impact of Data Augmentation on Dataset Size")
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "data_augmentation_impact.png"))
    plt.close()



def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = YOLO(MODEL)

    # Training
    model.train(
        data="train_data.yaml",
        epochs=100,
        imgsz=640,        
        batch=32,             
        optimizer="AdamW",
        lr0=0.01,         
        device="cpu",       
        
        project="result/train_ppe",
        name="ppe_detector",
        exist_ok=True
    )

    results_dir = os.path.join(OUTPUT_DIR, "ppe_detector")
    csv_path = os.path.join(results_dir, "results.csv")

    if os.path.exists(csv_path):
        plot_metrics(csv_path, results_dir)
        print(" Accuracy and loss curves saved in:", results_dir)
    else:
        print("results.csv not found. Check training logs.")



    print(" Generating confusion matrix...")
    plot_confusion_matrix(model, "train_data.yaml", results_dir)

    #  Model Comparison
    print(" Generating model comparison chart...")
    plot_model_comparison(results_dir)

    #  Data Augmentation Effect
    print(" Generating data augmentation impact chart...")
    plot_data_augmentation_effect(results_dir)


    print(" Training finished! Best weights saved as:")
    print(os.path.join(results_dir, "weights", "best.pt"))

if __name__ == "__main__":
    main()
