# Examples

This directory contains example scripts demonstrating how to use the Deep Stereo project.

## Available Examples

### 1. `visualize_doe.py`
Visualize DOE (Diffractive Optical Element) phase masks and their corresponding PSFs at different depths.

```bash
cd examples
python visualize_doe.py
```

### 2. `inference.py`
Run depth estimation inference on a stereo image pair.

```bash
cd examples
python inference.py
```

**Note:** You need to modify the image paths in the script before running.

## Requirements

Make sure you have installed all dependencies:

```bash
pip install -r ../requirements.txt
pip install matplotlib  # For visualization
```

## Output

Each script will save visualization results as PNG files in the current directory.
