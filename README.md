# Sign Language Recognition with I3D

A production-ready PyTorch implementation for Word-Level American Sign Language (WLASL) recognition using Inflated 3D ConvNet (I3D).

## Features

- **Pretrained I3D Model**: Uses Kinetics-400 pretrained 3D CNN backbones (R3D-18, MC3-18, R(2+1)D-18)
- **Mixed Precision Training**: Automatic Mixed Precision (AMP) for faster training on RTX 3070Ti and similar GPUs
- **Early Stopping**: Prevents overfitting by monitoring validation accuracy
- **LabelEncoder**: Automatic encoding of class labels with sklearn
- **Video Backend**: Supports both decord (fast) and torchvision for video loading
- **Production Ready**: Clean, modular, well-documented code

## Requirements

- Python 3.10+
- PyTorch 2.x
- CUDA-capable GPU (tested on RTX 3070Ti)
- Windows/Linux compatible

## Installation

1. **Create virtual environment:**
```bash
# Windows
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/Mac
python3.10 -m venv venv
source venv/bin/activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Dataset Structure

Organize your dataset as follows:

```
dataset/
├── videos/
│   ├── 07085.mp4
│   ├── 07086.mp4
│   ├── 07087.mp4
│   └── ...
├── train.txt
├── val.txt
└── test.txt
```

**Annotation file format (train.txt, val.txt, test.txt):**
```
video_name.mp4 label
```

Example:
```
07085.mp4 0
07086.mp4 1
07087.mp4 0
07088.mp4 2
```

Where `label` is an integer class ID.

## Training

**Basic training:**
```bash
python train.py --root_dir dataset --train_file dataset/train.txt --val_file dataset/val.txt
```

**Full options:**
```bash
python train.py \
    --root_dir dataset \
    --train_file dataset/train.txt \
    --val_file dataset/val.txt \
    --backbone r3d_18 \
    --epochs 30 \
    --batch_size 8 \
    --lr 0.0001 \
    --num_frames 32 \
    --frame_size 224 \
    --patience 5 \
    --save_dir checkpoints
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--root_dir` | dataset | Root directory of the dataset |
| `--train_file` | dataset/train.txt | Path to training annotations |
| `--val_file` | dataset/val.txt | Path to validation annotations |
| `--backbone` | r3d_18 | Model backbone (r3d_18, mc3_18, r2plus1d_18) |
| `--epochs` | 30 | Number of training epochs |
| `--batch_size` | 8 | Batch size |
| `--lr` | 0.0001 | Learning rate |
| `--num_frames` | 32 | Frames to sample per video |
| `--frame_size` | 224 | Frame resize dimension |
| `--dropout` | 0.5 | Dropout probability |
| `--patience` | 5 | Early stopping patience |
| `--use_amp` | True | Use mixed precision training |
| `--num_workers` | 4 | Data loading workers |
| `--save_dir` | checkpoints | Directory to save models |
| `--resume` | None | Path to checkpoint to resume from |

## Output

After training, the following files are saved in `checkpoints/`:

- `best_model.pth` - Model with best validation accuracy
- `last_model.pth` - Model from the last epoch
- `label_encoder.pkl` - Fitted LabelEncoder for inference
- `checkpoint_epoch_N.pth` - Periodic checkpoints

## Model Architecture

The model uses a 3D ResNet backbone pretrained on Kinetics-400:

- **Input**: Video tensor of shape `(B, C, T, H, W)` = `(batch, 3, 32, 224, 224)`
- **Backbone**: R3D-18 / MC3-18 / R(2+1)D-18 with inflated 3D convolutions
- **Classifier**: Dropout + Linear layer
- **Output**: Class logits of shape `(B, num_classes)`

## Inference

```python
import torch
import pickle
from model import create_model
from dataset import WLASLDataset

# Load label encoder
with open('checkpoints/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load model
checkpoint = torch.load('checkpoints/best_model.pth')
model = create_model(
    num_classes=checkpoint['num_classes'],
    backbone=checkpoint['backbone'],
    pretrained=False
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict
with torch.no_grad():
    # video: tensor of shape (1, 3, 32, 224, 224)
    outputs = model(video)
    pred_idx = outputs.argmax(dim=1).item()
    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    print(f"Predicted class: {pred_label}")
```

## Testing

To test the dataset loading:
```bash
python dataset.py --root_dir dataset --train_file dataset/train.txt
```

To test the model:
```bash
python model.py
```

## Project Structure

```
├── dataset.py          # Custom Dataset class for video loading
├── model.py            # I3D model wrapper
├── train.py            # Training script
├── utils.py            # Utility functions (accuracy, early stopping, etc.)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Performance Tips

1. **Use decord**: Install decord for faster video loading (`pip install decord`)
2. **Increase workers**: Set `--num_workers 8` if you have more CPU cores
3. **Adjust batch size**: RTX 3070Ti can handle batch_size=8 with 32 frames
4. **Mixed precision**: Enabled by default, provides ~2x speedup

## Citation

If you use this code, please cite the WLASL dataset:

```bibtex
@inproceedings{li2020word,
    title={Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison},
    author={Li, Dongxu and Rodriguez, Cristian and Yu, Xin and Li, Hongdong},
    booktitle={The IEEE Winter Conference on Applications of Computer Vision},
    pages={1459--1469},
    year={2020}
}
```

## License

MIT License
