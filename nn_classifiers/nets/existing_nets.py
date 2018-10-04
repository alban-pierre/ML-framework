
# Nets that are already in pytorch


from torchvision.models.alexnet import __all__ as alexnet_nets
from torchvision.models.densenet import __all__ as densenet_nets
from torchvision.models.inception import __all__ as inception_nets
from torchvision.models.resnet import __all__ as resnet_nets
from torchvision.models.squeezenet import __all__ as squeezenet_nets
from torchvision.models.vgg import __all__ as vgg_nets


def help():
    print("Here are the nets that are already in pytorch :")
    print("torchvision.models.alexnet    : ", "  ".join(alexnet_nets))
    print("torchvision.models.densenet   : ", "  ".join(densenet_nets))
    print("torchvision.models.inception  : ", "  ".join(inception_nets))
    print("torchvision.models.resnet     : ", "  ".join(resnet_nets))
    print("torchvision.models.squeezenet : ", "  ".join(squeezenet_nets))
    print("torchvision.models.vgg        : ", "  ".join(vgg_nets))
