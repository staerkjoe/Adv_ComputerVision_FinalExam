from pyexpat import model
import wandb
import yaml
from pathlib import Path
from src.data_loader import DataLoader
from src.model import Model
import torch

def load_config():
    config_path = Path(__file__).resolve().parent / "configs" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)
    
def main():
    config = load_config()
    print("Loaded config:", config)
    data_path = config['data']['path']
    epochs = config['model']['epochs']
    batch_size = config['model']['batch_size']
    imgsz = config['model']['imgsz']
    wandb_project = config['wandb']['project']
    wandb_run_name = config['wandb']['run_name']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataloader = DataLoader(config)
    dataloader.download_dataset()
    dataloader = dataloader  # ...existing code...

    model = Model(config)   # pass device into Model
    model = model.load_model(device=device)  # pass device into load_model
    wandb.init(project="yolov8-training", name="run-1")

    # Train the model with W&B logging enabled
    model.train(data=data_path, epochs=epochs, imgsz=imgsz, batch=batch_size, project=wandb_project ,name=wandb_run_name, save_period=-1, exist_ok=True, wandb=True)

    # Evaluate model (results will also sync to W&B)
    metrics = model.val()
    print(metrics)

    # 5. Optional: upload best model as a W&B artifact
    artifact = wandb.Artifact("best_model", type="model")
    artifact.add_file(model.ckpt_path)
    wandb.log_artifact(artifact)

if __name__ == "__main__":
    main()