from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
import torch

from load_encoder import load_IN22k_vit_h_14_900e

model = load_IN22k_vit_h_14_900e()

writer = SummaryWriter()

writer.add_graph(model, torch.rand(1, 3, 224, 224))
writer.close()
