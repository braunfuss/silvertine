import logging
from pyrocko import guts
from pyrocko.guts import List, Unicode, Object

from silvertine.version import __version__
from silvertine.meta import silvertineError

guts_prefix = 'silvertine'
logger = logging.getLogger('silvertine.report')


class RunInfo(Object):
    tags = List.T(
        Unicode.T(),
        help='List of user defined labels')

    def add_tag(self, tag):
        if tag not in self.tags:
            self.tags.append(tag)
            self.tags.sort()
        else:
            logger.warn('While adding tag: tag already set: %s' % tag)

    def remove_tag(self, tag):
        try:
            self.tags.remove(tag)
        except ValueError:
            logger.warn('While removing tag: tag not set: %s' % tag)


def read_info(path):
    try:
        info = guts.load(filename=path)
    except OSError:
        raise silvertineError(
            'Cannot read silvertine run info file: %s' % path)

    if not isinstance(info, RunInfo):
        raise silvertineError(
            'Invalid silvertine run info in file "%s".' % path)

    return info


def write_info(info, path):
    try:
        guts.dump(
            info,
            filename=path,
            header='silvertine run info file, version %s' % __version__)

    except OSError:
        raise silvertineError(
            'Cannot write silvertine run info file: %s' % path)
