from pyexpat import model
import wandb
import yaml
from pathlib import Path
from src.data_loader import DataLoader
from src.visuals import Visuals
from src.model import YoloModel
import torch
from ultralytics.utils import SETTINGS


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

    # Enable Ultralytics W&B integration
    SETTINGS["wandb"] = True

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

    # Initialize Visuals class
    visual = Visuals(config, model.model)
    
    # Define custom callback to log all visualizations before W&B finishes
    def on_train_end(trainer):
        """Called when training ends, before W&B closes"""
        print("\n=== Generating Custom Visualizations ===")
        
        # 1. Log parameter comparison
        total_params, trainable_params = visual.count_parameters()
        trainable_param_fig = visual.plot_trainable_parameters(total_params, trainable_params)
        wandb.log({"model/trainable_parameters": wandb.Image(trainable_param_fig)})
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {total_params - trainable_params:,}")
        
        # 2. Log all training visualizations (losses, metrics, and Ultralytics plots)
        visual.log_all_training_visualizations(trainer)
        
        # 3. Upload best model as W&B artifact
        artifact = wandb.Artifact("best_model", type="model")
        artifact.add_file(trainer.best)  # Path to best model weights
        wandb.log_artifact(artifact)
        print(f"Best model saved as artifact: {trainer.best}")
        
        print("=== Custom Visualizations Complete ===\n")

    # Add callback to model
    model.add_callback("on_train_end", on_train_end)

    # Train model (Ultralytics will manage W&B automatically)
    print("\n=== Starting Training ===")
    results = model.train(
        data=data_path, 
        epochs=epochs, 
        imgsz=imgsz, 
        batch=batch_size, 
        save_period=-1, 
        freeze=frozen_layers, 
        exist_ok=True,
        project=wandb_project,
        name=wandb_run_name
    )

    # Evaluate model (metrics auto-logged to W&B by Ultralytics)
    print("\n=== Running Final Validation ===")
    metrics = model.val()
    print(metrics)
    
    print("\n=== Training Complete ===")


if __name__ == "__main__":
    main()