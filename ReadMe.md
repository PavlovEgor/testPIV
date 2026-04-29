# testPIV

A Python framework for testing various Particle Image Velocimetry (PIV) methods. This project generates synthetic particle images from a known flow field, applies PIV algorithms to predict the velocity field, and compares the predictions with the ground truth.

## Overview

The project implements a complete PIV simulation and testing pipeline:

1. **Generate synthetic frames** – Create two consecutive particle images based on:
   - A known **Flow** (analytical velocity field)
   - Initial particle distribution defined by **Particles**
2. **Predict velocity field** – Apply different PIV methods from the **ModelPIV** class
3. **Validate accuracy** – Compare the predicted velocity field with the true flow field


## Key Classes

### `Flow`
Defines the true analytical flow field. Provides velocity `(u, v)` at any point `(x, y)` and time `t`.

### `Particles`
Manages particle distributions including:
- Initial seeding positions
- Particle intensity profiles (e.g., Gaussian)
- Motion tracking based on flow field

### `ModelPIV`
Contains multiple PIV prediction methods. Each method takes two consecutive particle frames and returns an estimated velocity field. Current implementations include:
- https://github.com/NikNazarov/TorchPIV

## Workflow

The typical testing workflow (`main.py`) follows these steps:

```python
# 1. Initialize flow field
flow = Flow.YourFlowType(params)

# 2. Create particles at t=0
particles = Particles.YourParticleType(params)

# 3. Generate two frames (t=0 and t=dt)
particles.evolve(flow, dt)  # move particles according to flow

# 4. Predict velocity field using a ModelPIV method
model = ModelPIV.YourMethod(params)
model.predict(params)

# 5. Compare with ground truth from flow
error = model.error(params)
```


## Installation (Linux)

```
# clone repository
git clone https://github.com/PavlovEgor/testPIV.git
cd testPIV

# activate virtual environment
python3 -m venv venv
source venv/bin/activate

# install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# install the package in development mode
pip install -e .
```

Verify Installation:
```
python3 Tests/simple_test.py
```
