from pathlib import Path
import yaml
from src.model import YoloModel
import wandb
import cv2


def load_config():
    config_path = Path(__file__).resolve().parent / "configs" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)
    

def main():
    config = load_config()
    data_path = config['data']['path_inference']
    yolo_model = YoloModel(config)
    model = yolo_model.load_yolo_from_wandb()

    wandb.init(project=config['wandb']['project'], name=config['wandb']['run_name'])

    # Run validation on inference data
    val_results = model.val(data=data_path, save_json=True, plots=True)
    val_save_dir = Path(val_results.save_dir)
    
    # Log confusion matrix
    confusion_matrix_path = val_save_dir / "confusion_matrix.png"
    if confusion_matrix_path.exists():
        wandb.log({"confusion_matrix": wandb.Image(str(confusion_matrix_path))})
    
    # Log metric plots
    for plot_name in ["P_curve.png", "R_curve.png", "F1_curve.png", "PR_curve.png"]:
        plot_path = val_save_dir / plot_name
        if plot_path.exists():
            wandb.log({plot_name.replace('.png', ''): wandb.Image(str(plot_path))})
    
    # Log prediction images with bounding boxes
    pred_images = []
    val_images_dir = val_save_dir
    for img_path in val_images_dir.glob("*.jpg"):
        pred_images.append(wandb.Image(str(img_path), caption=img_path.name))
        if len(pred_images) >= 20:  # Limit to first 20 images
            break
    
    if pred_images:
        wandb.log({"predictions": pred_images})
    
    # Log metrics summary
    wandb.log({
        "mAP50": val_results.box.map50,
        "mAP50-95": val_results.box.map,
        "precision": val_results.box.mp,
        "recall": val_results.box.mr
    })

    wandb.finish()


if __name__ == "__main__":
    main()