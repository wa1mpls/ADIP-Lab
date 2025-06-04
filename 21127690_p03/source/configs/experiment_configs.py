# Danh sách các cấu hình thí nghiệm
EXPERIMENTS = [
    {
        "input_size": 32,
        "patch_size": 4,
        "max_len": 100,
        "heads": 8,
        "classes": 10,
        "layers": 6,
        "embed_dim": 256,
        "mlp_dim": 512,
        "dropout": 0.1,
        "num_epochs": 20
    },
    {
        "input_size": 32,
        "patch_size": 8,
        "max_len": 100,
        "heads": 16,
        "classes": 10,
        "layers": 12,
        "embed_dim": 128,
        "mlp_dim": 256,
        "dropout": 0.1,
        "num_epochs": 20
    }
]