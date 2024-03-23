import torch


def device_cpu(model, load_path):
    ckpt = torch.load(load_path, map_location=torch.device('cpu'))
    pretrained_dict = ckpt['encoder']

    encoder_mod = {}
    for k, v in pretrained_dict.items():
        new_key = k[len("module."):]
        encoder_mod[new_key] = v
    model.load_state_dict(encoder_mod)
