import torch


def _load_cpu_vit_huge(patch_size, num_classes, checkpoint):
    from src.models.vision_transformer import vit_huge
    load_path = './checkpoint/' + checkpoint
    model = vit_huge(patch_size=patch_size, num_classes=num_classes)
    ckpt = torch.load(load_path, map_location=torch.device('cpu'))
    pretrained_dict = ckpt['encoder']

    encoder_mod = {}
    for k, v in pretrained_dict.items():
        new_key = k[len("module."):]
        encoder_mod[new_key] = v
    model.load_state_dict(encoder_mod)

    return model


def load_cpu_IN22k_vit_h_14_900e():
    return _load_cpu_vit_huge(14, 22000, 'IN22K-vit.h.14-900e.pth.tar')


def load_cpu_IN1k_vit_h_14_300e():
    return _load_cpu_vit_huge(14, 1000, 'IN1K-vit.h.14-300e.pth.tar')
