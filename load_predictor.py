import torch
from torch.utils.tensorboard import SummaryWriter

from src.helper import init_model
from src.models.vision_transformer import vit_predictor

# model = vit_predictor(num_patches=patch_size=16, num_classes=1000)
if torch.cuda.is_available():
    print("Using GPU")
    map_location = None
else:
    print("Using CPU")
    map_location = torch.device('cpu')

encoder, predictor = init_model(map_location)

print(predictor)

writer = SummaryWriter()
writer.add_graph(predictor, torch.rand(1, 3, 224, 224))  # FIXME
writer.close()
