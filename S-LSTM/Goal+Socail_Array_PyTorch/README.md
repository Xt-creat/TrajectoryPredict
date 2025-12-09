# Social LSTM with Goal and Social Array - PyTorch Implementation

This is a PyTorch implementation of the Social LSTM model with Goal and Social Array modifications, converted from the original TensorFlow version.

## Features

- **Goal Input**: Each pedestrian's final destination (goal) is included as input
- **Social Array**: Replaces grid representation with distance-ordered array of nearby pedestrians
- **PyTorch**: Modern PyTorch implementation with GPU support

## Requirements

- Python 3.7+
- PyTorch 1.8.0+
- NumPy 1.19.0+

Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Structure

The data should be organized as follows:
```
Goal+Socail_Array_PyTorch/
├── data/
│   ├── eth/
│   │   ├── hotel/
│   │   └── univ/
│   └── ucy/
│       ├── zara/
│       │   ├── zara01/
│       │   └── zara02/
│       └── univ/
└── social_lstm/
    ├── social_model.py
    ├── social_train.py
    ├── social_sample.py
    ├── social_utils.py
    └── grid.py
```

## Usage

### Training

```bash
cd Goal+Socail_Array_PyTorch/social_lstm
python social_train.py --leaveDataset 0 --num_epochs 50 --batch_size 16
```

### Sampling/Testing

```bash
python social_sample.py --test_dataset 0 --epoch 0 --obs_length 8 --pred_length 12
```

## Key Differences from TensorFlow Version

1. **Model Definition**: Uses `nn.Module` instead of TensorFlow's graph-based approach
2. **Training Loop**: Standard PyTorch training loop with `optimizer.step()`
3. **Device Management**: Automatic GPU/CPU device selection
4. **Checkpointing**: Uses PyTorch's native checkpoint format (`.pth`)

## Model Architecture

- **Input**: [ID, x, y, goal_x, goal_y] for each pedestrian
- **Social Array**: Distance-ordered array of nearby pedestrians (grid_size × 2)
- **Embeddings**: 
  - Spatial embedding: 4D → embedding_size
  - Social array embedding: grid_size×2 → 1
- **LSTM**: Processes embedded features
- **Output**: 5D distribution parameters (mux, muy, sx, sy, corr)

## Notes

- The model automatically processes data and creates preprocessed pickle files
- Training and validation splits are handled automatically
- Model checkpoints are saved after each epoch

