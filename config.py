import torch

class Configuration:
    data_path: str = ""
    save_path: str = ""
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
    weight_decay: float = 1e-4
    num_classes: int = 2
    dropout: float = 0.3
    patience: int = 20
    early_stop: bool = True
    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
