from src.models.vision_transformer import vit_huge
# Initialize the ViT-H model with the specified patch size and resolution
model = vit_huge(patch_size=14, num_classes=1000)  # Adjust num_classes if needed

import torch
# Load the state dictionary from the file
# state_dict = torch.load('/content/drive/MyDrive/IN1K-vit.h.14-300e.pth.tar')
state_dict = torch.load('/home/simone/Downloads/ijepa/IN1K-vit.h.14-300e.pth.tar')

# Load the state dictionary into the model
model.load_state_dict(state_dict)

# Print the layers/modules of the model for inspection
def print_model_layers(model, prefix=""):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Module):
            module_name = prefix + "." + name if prefix else name
            print(module_name)
            print_model_layers(module, prefix=module_name)

print_model_layers(model)
