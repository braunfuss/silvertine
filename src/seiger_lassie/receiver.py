from pyrocko.guts import Tuple, String
from silvertine.seiger_lassie import geo

guts_prefix = 'seiger_lassie'


class Receiver(geo.Point):
    codes = Tuple.T(3, String.T())


__all__ = [
    'Receiver',
]
