import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = "cuda" if torch.cuda.is_available() else "cpu"
num_workers = 2
batch_size = 8
pin_memory = True
load_model = True
save_model = True
checkpoint_file = "b7.pth.tar"
weight_decay = 1e-4
learning_rate = 1e-4
num_epochs = 1

basic_transform = A.Compose(
    [
        A.Resize(height=448, width=448),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

