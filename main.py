from pyexpat import model
import wandb
import yaml
from pathlib import Path
from src.data_loader import DataLoader
from src.visuals import Visuals
from src.model import YoloModel
import torch
from ultralytics.utils import (SETTINGS)


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
    frozen_layers = config['model']['freeze']

    # Initialize W&B FIRST
    wandb.init(project=wandb_project, name=wandb_run_name)
    SETTINGS["wandb"] = True  # Enable Ultralytics W&B integration

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load Data
    from roboflow import Roboflow
    rf = Roboflow(api_key=config['roboflow']['api_key'])
    project = rf.workspace(config['roboflow']['workspace']).project(config['roboflow']['project'])
    version = project.version(config['roboflow']['version'])
    version.download(config['roboflow']['model_format'])

    # Load Model
    yolo_model = YoloModel(config)
    model = yolo_model.load_model(device=device)
    model.info()

    # Log custom visualization BEFORE training
    visual = Visuals(config, model.model)  # Pass model.model (the actual PyTorch model)
    total_params, trainable_params = visual.count_parameters()
    trainable_param_fig = visual.plot_trainable_parameters(total_params, trainable_params)
    wandb.log({"trainable_parameters_plot": wandb.Image(trainable_param_fig)})

    # Train model (W&B will automatically log metrics during training)
    results = model.train(
        data=data_path, 
        epochs=epochs, 
        imgsz=imgsz, 
        batch=batch_size, 
        save_period=-1, 
        freeze=frozen_layers, 
        exist_ok=True,
        project=wandb_project,  # Optional: specify project
        name=wandb_run_name      # Optional: specify run name
    )

    # Evaluate model (metrics will auto-sync to W&B)
    metrics = model.val()
    print(metrics)

    # Upload best model as W&B artifact
    artifact = wandb.Artifact("best_model", type="model")
    artifact.add_file(model.ckpt_path)
    wandb.log_artifact(artifact)
    
    wandb.finish()

if __name__ == "__main__":
    main()