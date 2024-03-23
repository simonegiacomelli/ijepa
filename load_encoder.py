import torch


def device_cpu(model, load_path):
    ckpt = torch.load(load_path, map_location=torch.device('cpu'))
    pretrained_dict = ckpt['encoder']

    encoder_mod = {}
    for k, v in pretrained_dict.items():
        new_key = k[len("module."):]
        encoder_mod[new_key] = v
    model.load_state_dict(encoder_mod)

def load_cpu_IN22k_vit_h_14_900e():
    from src.models.vision_transformer import vit_huge

    model = vit_huge(patch_size=14, num_classes=22000)
    load_path = '/home/simone/Downloads/ijepa/IN22K-vit.h.14-900e.pth.tar'
    device_cpu(model, load_path)
    return model


def load_cpu_IN1k_vit_h_14_300e():
    from src.models.vision_transformer import vit_huge

    model = vit_huge(patch_size=14, num_classes=1000)
    load_path = '/home/simone/Downloads/ijepa/IN1K-vit.h.14-300e.pth.tar'
    device_cpu(model, load_path)
    print(model)


