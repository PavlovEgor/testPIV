from torchPIV import OfflinePIV
import torch

piv_gen = OfflinePIV(
    folder="test_images", # Path to experiment
    device="cpu", # Device name
    file_fmt="bmp",
    wind_size=64,
    overlap=32,
    dt=12, # Time between frames, mcs
    scale = 0.02, # mm/pix
    multipass=2,
    multipass_mode="CWS", # CWS or DWS
    multipass_scale=2.0, # Window downscale on each pass
    folder_mode="pairs" # Pairs or sequential frames
)

results = []
for out in piv_gen():
    x, y, vx, vy = out
    results.append(out)