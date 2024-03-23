import load_encoder
from src.models.vision_transformer import vit_huge

model = vit_huge(patch_size=14, num_classes=1000)
load_path = '/home/simone/Downloads/ijepa/IN1K-vit.h.14-300e.pth.tar'
load_encoder.device_cpu(model, load_path)
print(model)
