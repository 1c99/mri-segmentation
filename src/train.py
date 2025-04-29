import os
import sys
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import mlflow
import mlflow.pytorch
import wandb
import matplotlib.pyplot as plt
from tqdm import tqdm
import psutil
import ray
import torch.nn.functional as F

print("Starting script...")
print("Python executable:", sys.executable)
print("Current working directory:", os.getcwd())

# New: Ray Tune
from ray import tune
from ray.air import session
from ray.tune.schedulers import ASHAScheduler

# Add Ray AIR imports
from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig

# Initialize tracking tools with fallback
global use_mlflow, use_wandb
use_mlflow = True
use_wandb = True

print("\nInitializing tracking tools...")

# Set up MLflow to use local directory
mlflow_tracking_dir = os.path.abspath("./mlruns")
os.makedirs(mlflow_tracking_dir, exist_ok=True)
mlflow.set_tracking_uri(f"file://{mlflow_tracking_dir}")
mlflow.set_experiment("mri-segmentation")
print(f"MLflow tracking directory: {mlflow_tracking_dir}")

if use_wandb:
    try:
        print("Attempting to initialize W&B...")
        wandb.login()
        print("W&B login successful")
    except Exception as e:
        print(f"W&B login failed: {e}")
        use_wandb = False

print(f"W&B tracking enabled: {use_wandb}")

print("\nChecking data directories...")
print(f"Image directory exists: {os.path.exists('./data/Task01_BrainTumour/imagesTr')}")
print(f"Mask directory exists: {os.path.exists('./data/Task01_BrainTumour/labelsTr')}")
print(f"Number of image files: {len(os.listdir('./data/Task01_BrainTumour/imagesTr'))}")
print(f"Number of mask files: {len(os.listdir('./data/Task01_BrainTumour/labelsTr'))}")

# -------------------------------
# 1. Datasets
# -------------------------------
class MRIDataset(Dataset):
    def __init__(self, image_dir, mask_dir, modalities=['flair', 't1', 't1ce', 't2']):
        print(f"\nInitializing MRIDataset...")
        print(f"Image directory: {image_dir}")
        print(f"Mask directory: {mask_dir}")
        
        self.image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir)])
        self.mask_paths = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir)])
        self.modalities = modalities
        print(f"Found {len(self.image_paths)} images and {len(self.mask_paths)} masks")
        
        # Try loading first image to verify data
        try:
            print(f"\nTesting first image load...")
            first_img = nib.load(self.image_paths[0]).get_fdata()
            print(f"First image shape: {first_img.shape}")
            first_mask = nib.load(self.mask_paths[0]).get_fdata()
            print(f"First mask shape: {first_mask.shape}")
            print("Successfully loaded first image and mask")
            
            # Pre-load first batch to ensure everything works
            print("\nPre-loading first batch to verify data loading...")
            for i in tqdm(range(min(4, len(self.image_paths))), desc="Loading first batch"):
                img = self[i]
                
        except Exception as e:
            print(f"Error during dataset initialization: {e}")
            raise

    def __len__(self):
        return len(self.image_paths)

    def normalize_modality(self, img, eps=1e-8):
        """Normalize each modality independently"""
        mean = np.mean(img)
        std = np.std(img)
        return (img - mean) / (std + eps)

    def __getitem__(self, idx):
        try:
            # Load NIfTI files with memory optimization
            img_proxy = nib.load(self.image_paths[idx])
            mask_proxy = nib.load(self.mask_paths[idx])
            
            # Get the data arrays efficiently
            img = img_proxy.get_fdata(dtype=np.float32)  # Shape: (H, W, D, 4)
            mask = mask_proxy.get_fdata(dtype=np.float32)  # Shape: (H, W, D)
            
            # Clear references to free memory
            img_proxy = None
            mask_proxy = None

            # Print memory usage for debugging
            process = psutil.Process()
            memory_info = process.memory_info()
            # print(f"\nMemory usage after loading item {idx}: {memory_info.rss / 1024 / 1024:.1f} MB")

            # Take middle slice from each modality
            mid_slice = img.shape[2] // 2
            img_slices = []
            
            # Handle both 3D and 4D images
            if img.ndim == 4:
                for mod in range(img.shape[-1]):
                    slice_mod = img[:, :, mid_slice, mod]
                    slice_mod = self.normalize_modality(slice_mod)
                    img_slices.append(slice_mod)
            else:
                slice_mod = img[:, :, mid_slice]
                slice_mod = self.normalize_modality(slice_mod)
                img_slices.append(slice_mod)
            
            # Stack modalities into channels
            img_combined = np.stack(img_slices, axis=0)
            
            # Take middle slice of mask and create one-hot encoding
            mask_slice = mask[:, :, mid_slice]
            
            # Convert to tensors
            img_tensor = torch.tensor(img_combined, dtype=torch.float32)  # Shape: (C, H, W)
            mask_tensor = torch.tensor(mask_slice, dtype=torch.long)  # Shape: (H, W)
            
            # Create one-hot encoding for the mask (4 classes: 0, 1, 2, 3)
            mask_onehot = torch.zeros((4, *mask_tensor.shape), dtype=torch.float32)
            for i in range(4):
                mask_onehot[i] = (mask_tensor == i).float()

            return img_tensor, mask_onehot
        except Exception as e:
            print(f"\nError loading item {idx} from {self.image_paths[idx]}: {e}")
            raise


# -------------------------------
# 2. Simple UNet Model
# -------------------------------
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4):
        super().__init__()
        
        # Encoder
        self.enc1 = self._make_layer(in_channels, 64)
        self.enc2 = self._make_layer(64, 128)
        self.enc3 = self._make_layer(128, 256)
        self.enc4 = self._make_layer(256, 512)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec1 = self._make_layer(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._make_layer(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = self._make_layer(128, 64)
        
        # Final layer
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        
        # Decoder with skip connections
        dec1 = self.dec1(torch.cat([self.up1(enc4), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.up2(dec1), enc2], dim=1))
        dec3 = self.dec3(torch.cat([self.up3(dec2), enc1], dim=1))
        
        # Final layer
        out = self.final(dec3)
        return out

# -------------------------------
# 3. Dice Loss
# -------------------------------
def dice_loss(pred, target, smooth=1.0):
    """Multi-class Dice loss"""
    pred = F.softmax(pred, dim=1)
    num_classes = pred.shape[1]
    
    total_loss = 0
    for cls in range(num_classes):
        pred_cls = pred[:, cls]
        target_cls = target[:, cls]
        
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        total_loss += (1 - dice)
    
    return total_loss / num_classes

# -------------------------------
# 4. Visualization (Save as Image)
# -------------------------------
def visualize_prediction(model, dataset, device, save_path="prediction.png"):
    model.eval()
    img, mask = dataset[0]
    img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(img)
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()

    img = img.squeeze().cpu().numpy()
    pred = pred.squeeze().cpu().numpy()
    mask = mask.squeeze().cpu().numpy()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('MRI Image')
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Ground Truth')
    plt.subplot(1, 3, 3)
    plt.imshow(pred, cmap='gray')
    plt.title('Prediction')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def get_device():
    """
    Automatically detect and return the best available device.
    Priority: CUDA > MPS > CPU
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        # Set memory allocation to be more efficient
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon) device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    return device

# -------------------------------
# 5. Training Loop
# -------------------------------
def train_loop(config):
    # Initialize tracking tools at the start of each trial
    use_mlflow = True
    use_wandb = True
    
    print("\nStarting training function...")
    
    # Convert Ray Tune types to Python types safely
    converted_config = {}
    for k, v in config.items():
        try:
            if k == "batch_size":
                converted_config[k] = int(v)  # Ensure batch_size is integer
            elif k == "epochs":
                converted_config[k] = int(v)  # Ensure epochs is integer
            elif isinstance(v, (float, int)):
                converted_config[k] = float(v)
            else:
                converted_config[k] = v
        except Exception as e:
            print(f"Warning: Could not convert {k}: {v}, using as is. Error: {e}")
            converted_config[k] = v
    
    config = converted_config
    print("Config after conversion:", config)

    # Get data directories from config
    image_dir = config["image_dir"]
    mask_dir = config["mask_dir"]
    
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found at: {image_dir}")
    if not os.path.exists(mask_dir):
        raise FileNotFoundError(f"Mask directory not found at: {mask_dir}")
    
    print(f"Using data directories:\nImages: {image_dir}\nMasks: {mask_dir}")
    
    # Initialize device inside the function
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else 
                         "cpu")
    print(f"Using device: {device}")

    # Set number of workers based on device
    num_workers = 0 if device.type in ['cuda', 'mps'] else 4
    
    # Create model, dataset, and optimizer inside the function
    model = SimpleUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # Initialize MLflow
    if use_mlflow:
        try:
            mlflow.start_run(nested=True)
            mlflow.log_params(config)
            mlflow.log_param("num_workers", num_workers)
        except Exception as e:
            print(f"MLflow logging failed: {e}")
            use_mlflow = False

    # Initialize W&B
    if use_wandb:
        try:
            wandb.init(project="mri-segmentation", config=config, reinit=True)
        except Exception as e:
            print(f"W&B initialization failed: {e}")
            use_wandb = False

    try:
        # Dataset setup
        dataset = MRIDataset(image_dir, mask_dir)
        
        val_split = 0.2
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], 
                                shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], 
                              shuffle=False, num_workers=num_workers)

        best_val_loss = float('inf')
        patience = 5
        counter = 0

        for epoch in range(config["epochs"]):
            # Training phase
            model.train()
            total_loss = 0
            num_batches = 0
            
            for img, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} - Training"):
                img, mask = img.to(device), mask.to(device)
                optimizer.zero_grad()
                output = model(img)
                loss = dice_loss(output, mask)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
            
            train_loss = total_loss / num_batches

            # Validation phase
            model.eval()
            total_val_loss = 0
            num_val_batches = 0
            
            with torch.no_grad():
                for img, mask in tqdm(val_loader, desc="Validation"):
                    img, mask = img.to(device), mask.to(device)
                    output = model(img)
                    loss = dice_loss(output, mask)
                    total_val_loss += loss.item()
                    num_val_batches += 1
            
            val_loss = total_val_loss / num_val_batches

            # Log metrics
            metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch": epoch
            }
            
            if use_mlflow:
                try:
                    mlflow.log_metrics(metrics, step=epoch)
                except Exception as e:
                    print(f"MLflow logging failed: {e}")
                    use_mlflow = False
            
            if use_wandb:
                try:
                    wandb.log(metrics)
                except Exception as e:
                    print(f"W&B logging failed: {e}")
                    use_wandb = False

            # Early stopping and model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                torch.save(model.state_dict(), "best_model.pth")
                if use_mlflow:
                    mlflow.log_artifact("best_model.pth")
                if use_wandb:
                    wandb.save('best_model.pth')
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping triggered after {patience} epochs!")
                    break

            print(f"\nEpoch {epoch+1}/{config['epochs']}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Only report to Ray if we're in a Ray Tune trial
            if "RAY_TUNE_TRIAL_NAME" in os.environ:
                session.report({"val_loss": val_loss})

    except Exception as e:
        print(f"Error during training: {e}")
        raise
    finally:
        if use_mlflow:
            mlflow.end_run()
        if use_wandb:
            wandb.finish()

# -------------------------------
# 6. Entry Point
# -------------------------------
if __name__ == "__main__":
    print("\nStarting main...")
    try:
        # Get absolute paths for data directories
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(os.path.dirname(base_dir))
        data_dir = os.path.join(project_dir, "segmentation", "data", "Task01_BrainTumour")
        image_dir = os.path.join(data_dir, "imagesTr")
        mask_dir = os.path.join(data_dir, "labelsTr")

        print(f"\nUsing data directories:")
        print(f"Image directory: {image_dir}")
        print(f"Mask directory: {mask_dir}")

        # Environment variables to control execution mode
        use_ray = os.getenv("USE_RAY_TUNE", "false").lower() == "true"
        dev_mode = os.getenv("DEV_MODE", "false").lower() == "true"

        if use_ray:
            ray.init()
            
            if dev_mode:
                print("Running in development mode - using minimal config for testing")
                scaling_config = ScalingConfig(
                    num_workers=1,
                    use_gpu=False
                )
                param_space = {
                    "lr": 0.001,
                    "batch_size": 2,
                    "epochs": 2,
                    "image_dir": image_dir,  # Add data paths to config
                    "mask_dir": mask_dir
                }
                num_samples = 2
            else:
                print("Running in full training mode - using complete hyperparameter search")
                scaling_config = ScalingConfig(
                    num_workers=1,
                    use_gpu=True
                )
                param_space = {
                    "lr": tune.loguniform(1e-4, 1e-2),
                    "batch_size": tune.choice([2, 4, 8, 16]),
                    "epochs": 10,
                    "image_dir": image_dir,  # Add data paths to config
                    "mask_dir": mask_dir
                }
                num_samples = 8

            # Create scheduler
            scheduler = ASHAScheduler(
                max_t=param_space["epochs"],
                grace_period=1,
                reduction_factor=2
            )
            
            trainer = TorchTrainer(
                train_loop_per_worker=train_loop,
                train_loop_config=param_space,
                scaling_config=scaling_config
            )
            
            tuner = tune.Tuner(
                trainer,
                tune_config=tune.TuneConfig(
                    metric="val_loss",
                    mode="min",
                    num_samples=num_samples,
                    scheduler=scheduler
                )
            )
            
            try:
                print(f"\nStarting {'development' if dev_mode else 'full'} hyperparameter search...")
                results = tuner.fit()
                best_result = results.get_best_result(metric="val_loss", mode="min")
                print("\nBest trial config:", best_result.config)
                print("Best trial final validation loss:", best_result.metrics["val_loss"])
            except Exception as e:
                print("Error during tuning:", e)
                print("Falling back to default configuration...")
                config = {
                    "lr": 0.001,
                    "batch_size": 4,
                    "epochs": 20,
                    "image_dir": image_dir,
                    "mask_dir": mask_dir
                }
                train_loop(config)
            
            ray.shutdown()
        else:
            # Default single training run
            config = {
                "lr": 0.001,
                "batch_size": 4,
                "epochs": 20,
                "image_dir": image_dir,
                "mask_dir": mask_dir
            }
            train_loop(config)

    except Exception as e:
        print(f"Error in main: {e}")
        raise

    print("Script completed successfully!")