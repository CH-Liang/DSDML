from __future__ import print_function, absolute_import



from .Binomial import BinomialLoss
from .LiftedStructure import LiftedStructureLoss
from .multi_similarity_loss import MultiSimilarityLoss
from .Tripletloss import TripletLoss

__factory = {
    'Binomial': BinomialLoss,
    'LiftedStructure': LiftedStructureLoss,
    'triplet': TripletLoss,
    'MS':MultiSimilarityLoss,

}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a loss instance.

    Parameters
    ----------
    name : str
        the name of loss function
    """
    if name not in __factory:
        raise KeyError("Unknown loss:", name)
    return __factory[name](*args, **kwargs)



def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a loss instance.

    Parameters
    ----------
    name : str
        the name of loss function
    """
    if name not in __factory:
        raise KeyError("Unknown loss:", name)
    return __factory[name]( *args, **kwargs)
