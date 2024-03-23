- script to download checkpoints (see below)
- fix tensorboard pipeline (now the paths do not match)
- clean repo



wget https://github.com/facebookresearch/ijepa/blob/main/configs/in1k_vith14_ep300.yaml
wget https://github.com/facebookresearch/ijepa/blob/main/configs/in1k_vith16-448_ep300.yaml
wget https://github.com/facebookresearch/ijepa/blob/main/configs/in22k_vith14_ep66.yaml
wget https://github.com/facebookresearch/ijepa/blob/main/configs/in22k_vitg16_ep44.yaml

wget https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-logs-rank.0.csv
wget https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.16.448-logs-rank.0.csv
wget https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.h.14-logs-rank.0.csv
wget https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.g.16-logs-rank.0.csv

wget https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar
wget https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.16-448px-300e.pth.tar
wget https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.h.14-900e.pth.tar
wget https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.g.16-600e.pth.tar
