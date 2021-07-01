from pyrocko.guts import Tuple, String
from lassie import geo

guts_prefix = 'lassie'


class Receiver(geo.Point):
    codes = Tuple.T(3, String.T())


__all__ = [
    'Receiver',
]
