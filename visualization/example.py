from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
import torch

# Create a writer instance
writer = SummaryWriter()

# Load a predefined model
model = models.resnet18()

# Write the model graph
writer.add_graph(model, torch.rand(1, 3, 224, 224))
writer.close()
