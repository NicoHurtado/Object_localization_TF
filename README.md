# Object Localization with TensorFlow

This project demonstrates how to train a **Convolutional Neural Network (CNN)** from scratch using **TensorFlow/Keras** to simultaneously:

- **Classify** the emotion shown by an emoji (multiclass classification).
- **Predict** the location of that emoji within a 144×144 image (object localization).

---

## Project Overview

Each image is synthetically generated using OpenMoji emoji assets. A random emoji is pasted on a random background position. The network is trained to:

- Predict one of **9 emoji classes**.
- Regress the **(row, col)** coordinates (center of the emoji) normalized to the image size.

---

## Dataset Generation

There is no external dataset—images are procedurally created using:

- The `create_example()` function generates a synthetic image with:
  - A random emoji from 9 options.
  - Random placement within a 144×144 canvas.
- Each image has:
  - A one-hot encoded class label.
  - A pair of bounding box center coordinates.

---

## Model Architecture

- Input: `{'image': (144, 144, 3)}`
- Outputs:
  - `class_out`: Softmax over 9 emoji classes
  - `box_out`: 2D regression of normalized center (row, col)

### Model compiled with:
- `class_out` loss: `categorical_crossentropy`
- `box_out` loss: `mean squared error (MSE)`
- Optimizer: `Adam`

---

## Data Generator

The custom `data_generator()` yields batches of:

```python
(
  {'image': x_batch}, 
  {'class_out': y_batch, 'box_out': bbox_batch}
)
```

Where:
- `x_batch`: batch of input images
- `y_batch`: one-hot encoded class labels
- `bbox_batch`: normalized row/col coordinates

---

##  Training

Model is trained using:

- `model.fit(...)` with `data_generator()`
- `EarlyStopping` on `box_out_loss`
- `LearningRateScheduler` with decay every 5 epochs
- Custom callback to visualize predictions per epoch

---

##  Example Outputs

At each epoch, the model visualizes:

- Ground truth bounding box (green)
- Predicted bounding box (red)
- Predicted class label and confidence

---

## Requirements

```bash
tensorflow
numpy
matplotlib
Pillow
```

Install with:

```bash
pip install -r requirements.txt
```

---
