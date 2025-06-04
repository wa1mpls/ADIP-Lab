import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import torch
import unittest
from models.vision_transformer import VisionTransformer
from training.train import train
from evaluation.evaluate import evaluate
from utils.data_utils import get_dataloaders
from experiments.logger import ExperimentLogger
from configs.experiment_configs import EXPERIMENTS
from utils.visualize import plot_loss, plot_accuracy_comparison

def run_experiment(params, num_epochs, device, logger):
    """
    Chạy một thí nghiệm với các siêu tham số được chỉ định.
    
    Args:
        params (dict): Siêu tham số của mô hình
        num_epochs (int): Số epoch huấn luyện
        device (torch.device): Thiết bị chạy (CPU/GPU)
        logger (ExperimentLogger): Đối tượng ghi log
    """
    # Tải dữ liệu
    trainloader, testloader = get_dataloaders()
    
    # Lọc các tham số hợp lệ cho VisionTransformer (loại bỏ num_epochs)
    model_params = {k: v for k, v in params.items() if k != "num_epochs"}
    model = VisionTransformer(**model_params).to(device)
    
    # Huấn luyện (chỉ truyền num_epochs riêng, không qua **params)
    train_params = {k: v for k, v in params.items() if k != "num_epochs"}  # Loại num_epochs khỏi params
    train(model, trainloader, num_epochs, device, logger, **train_params)
    
    # Đánh giá
    accuracy = evaluate(model, testloader, device)
    print(f"Top-1 Accuracy: {accuracy:.2f}%")
    
    # Lưu kết quả
    logger.log_result(accuracy, **params)
    csv_file, json_file = logger.get_log_files()
    print(f"Results saved to: {csv_file} and {json_file}")
    
    return csv_file, json_file

def run_tests():
    """Chạy tất cả unit test trong thư mục tests/"""
    loader = unittest.TestLoader()
    suite = loader.discover('tests')
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()

if __name__ == "__main__":
    # Thiết lập device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Chạy unit test trước
    print("Running unit tests...")
    if not run_tests():
        print("Unit tests failed! Please check the errors above.")
        exit(1)
    print("All unit tests passed successfully!\n")
    
    # Danh sách các file JSON để so sánh accuracy
    json_files = []
    
    # Chạy từng thí nghiệm từ EXPERIMENTS trong configs
    for idx, exp_params in enumerate(EXPERIMENTS):
        print(f"Running Experiment {idx + 1} with params: {exp_params}")
        
        # Tạo logger cho mỗi thí nghiệm
        logger = ExperimentLogger(log_dir="logs")
        
        # Chạy thí nghiệm
        csv_file, json_file = run_experiment(
            params=exp_params,
            num_epochs=exp_params["num_epochs"],
            device=device,
            logger=logger
        )
        
        # Lưu file log và vẽ biểu đồ loss
        json_files.append(json_file)
        plot_loss(csv_file, f"logs/loss_exp_{idx}.png")
    
    # Vẽ biểu đồ so sánh accuracy giữa các thí nghiệm
    plot_accuracy_comparison(json_files, "logs/accuracy_comparison.png")
    print("\nAll experiments completed. Visualizations saved in 'logs/' directory.")