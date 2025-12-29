from ultralytics import YOLO
import ultralytics.nn.tasks as tasks_module
import torch

def patched_torch_safe_load(file):
    return torch.load(file, weights_only=False), file

tasks_module.torch_safe_load = patched_torch_safe_load

model = YOLO('detectors/RCTdetector_v11x_v2.pt')
print('Classes:', model.names)
print('RCT class index:', [k for k, v in model.names.items() if 'Root Canal' in v])
