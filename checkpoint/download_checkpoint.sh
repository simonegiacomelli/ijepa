# create function to download checkpoint; then call it with only the filename stem
function download_checkpoint() {
    if [ -f $1 ]; then
        echo "File $1 already exists"
    else
        echo "Downloading $1"
        wget "https://dl.fbaipublicfiles.com/ijepa/$1"
    fi
}

# download checkpoints
download_checkpoint IN1K-vit.h.14-300e.pth.tar
download_checkpoint IN1K-vit.h.16-448px-300e.pth.tar
download_checkpoint IN22K-vit.h.14-900e.pth.tar
download_checkpoint IN22K-vit.g.16-600e.pth.tar



#wget https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar
#wget https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.16-448px-300e.pth.tar
#wget https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.h.14-900e.pth.tar
#wget https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.g.16-600e.pth.tar
