# PyTorch ATCNet Implementation

This directory now contains a **PyTorch version** of ATCNet designed to work with your **cosupervisor's preprocessed data** (128 Hz, 0.5-4.0 s window, 4-40 Hz filter).

## Files Created

1. **`model_pytorch.py`** - PyTorch implementation of ATCNet
   - Converts TensorFlow/Keras model to PyTorch
   - Supports `in_samples=448` (matches cosupervisor preprocessing)
   - Includes attention blocks (MHA, SE) and TCN blocks

2. **`train_pytorch.py`** - Training script for PyTorch ATCNet
   - Loads cosupervisor preprocessed data (`.npz` files)
   - Applies standardization (fit on train, transform train/val/test)
   - Trains with early stopping (patience 60, after epoch 100, on val_accuracy)
   - Saves results, plots, and model weights

3. **`train_atcnet_pytorch_gui.py`** - GUI for PyTorch training
   - Similar layout to `train_atcnet_gui.py` (TensorFlow version)
   - Select subject ID and preprocessed data folder
   - Live training curves, test results, confusion matrix

## Key Differences from TensorFlow Version

| Aspect | TensorFlow (Original) | PyTorch (New) |
|--------|----------------------|---------------|
| **Preprocessing** | Raw BCI2a (250 Hz, 1.5-6.0 s) | Cosupervisor (128 Hz, 0.5-4.0 s) |
| **Input shape** | (N, 1, 22, 1125) | (N, 1, 22, 448) |
| **Model framework** | TensorFlow/Keras | PyTorch |
| **Data source** | Raw `.mat` files | Preprocessed `.npz` files |

## Usage

### Command Line

**Option 1: Cosupervisor preprocessed data (with filtering)**
```bash
python train_pytorch.py --preprocess "/mnt/c/Users/User/Desktop/Thesis/files2/MI dataset" --subject 1
```

**Option 2: Raw BCI2a data (NO filtering, like TensorFlow original)**
```bash
python train_pytorch.py --data "/mnt/c/Users/User/Desktop/Thesis/files2/Four class motor imagery (001-2014)" --subject 1
```

This lets you compare:
- **With filtering** (cosupervisor): 128 Hz, 0.5-4.0 s, 4-40 Hz filter → 448 samples
- **Without filtering** (raw): 250 Hz, 1.5-6.0 s, no filter → 1125 samples

### GUI

```bash
python train_atcnet_pytorch_gui.py
```

Then:
1. Choose **Data source**: 
   - **Cosupervisor preprocessed** (128 Hz, filtered) OR
   - **Raw BCI2a** (250 Hz, NO filtering)
2. Set **Subject ID** (1-9)
3. Browse to the appropriate folder:
   - If preprocessed: **MI dataset** folder
   - If raw: **Four class motor imagery (001-2014)** folder
4. Click **Start training**

## Requirements

Install PyTorch:
```bash
pip install torch torchvision
```

For GPU support (CUDA):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Model Architecture

- **Input**: (batch, 1, 22 channels, 448 samples)
- **Conv Block**: EEGNet-style (depthwise separable convolutions)
- **Attention**: Multi-Head Attention (MHA) or SE block
- **TCN Blocks**: Temporal Convolutional Networks with dilation
- **Output**: (batch, 4 classes) with softmax

## Training Configuration

- **Batch size**: 64
- **Learning rate**: 0.001 (Adam optimizer)
- **Epochs**: 500 (with early stopping)
- **Early stopping**: Patience 60, only after epoch 100, monitors val_accuracy
- **LR scheduler**: ReduceLROnPlateau (factor=0.9, patience=20)
- **Standardization**: Per-channel StandardScaler (fit on train, transform all)

## Results

Results are saved to `results_pytorch/` (CLI) or `results_pytorch_gui/` (GUI):
- `subject_{id}_curves.png` - Training curves (loss and accuracy)
- `subject_{id}_test_confusion_matrix.png` - Confusion matrix
- `subject_{id}_results.json` - All metrics and config
- `saved_models/subject_{id}.pth` - Model weights

## Notes

- The PyTorch model is designed for **448 samples** (cosupervisor preprocessing)
- Subject filtering works if the `.npz` files contain a `subject` array in `meta`
- Standardization is applied automatically (fit on train, transform train/val/test)
- GPU is auto-detected; falls back to CPU if CUDA unavailable
