import os
import csv
import json
from datetime import datetime

class ExperimentLogger:
    def __init__(self, log_dir="logs"):
        # Tạo thư mục logs nếu chưa tồn tại
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Tên file log dựa trên thời gian
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file = os.path.join(log_dir, f"experiment_{timestamp}.csv")
        self.json_file = os.path.join(log_dir, f"experiment_{timestamp}.json")
        
        # Khởi tạo file CSV với header
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss", "accuracy", "embed_dim", "mlp_dim", "patch_size", "heads", "layers"])

    def log_epoch(self, epoch, loss, **params):
        """Ghi loss mỗi epoch vào CSV"""
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{loss:.4f}", "", 
                            params.get("embed_dim", ""), 
                            params.get("mlp_dim", ""), 
                            params.get("patch_size", ""), 
                            params.get("heads", ""), 
                            params.get("layers", "")])

    def log_result(self, accuracy, **params):
        """Ghi kết quả cuối cùng (accuracy) vào CSV và JSON"""
        # Ghi vào CSV
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["final", "", f"{accuracy:.2f}", 
                            params.get("embed_dim", ""), 
                            params.get("mlp_dim", ""), 
                            params.get("patch_size", ""), 
                            params.get("heads", ""), 
                            params.get("layers", "")])
        
        # Ghi vào JSON
        result = {
            "accuracy": accuracy,
            "params": params
        }
        with open(self.json_file, 'w') as f:
            json.dump(result, f, indent=4)

    def get_log_files(self):
        """Trả về đường dẫn file log"""
        return self.csv_file, self.json_file