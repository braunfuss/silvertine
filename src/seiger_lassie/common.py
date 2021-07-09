import os.path as op
from string import Template
from collections import defaultdict

from pyrocko.guts import Object, String
from pyrocko.gf import Earthmodel1D

guts_prefix = 'seiger_lassie'


def data_file(fn):
    return op.join(op.split(__file__)[0], 'data', fn)


class LassieError(Exception):
    pass


class Earthmodel(Object):
    id = String.T()


class CakeEarthmodel(Earthmodel):
    earthmodel_1d = Earthmodel1D.T()


def grouped_by(l, key):
    d = defaultdict(list)
    for x in l:
        d[key(x)].append(x)

    return d


def xjoin(basepath, path):
    if path is None and basepath is not None:
        return basepath
    elif op.isabs(path) or basepath is None:
        return path
    else:
        return op.join(basepath, path)


def xrelpath(path, start):
    if op.isabs(path):
        return path
    else:
        return op.relpath(path, start)


class Path(String):
    pass


class HasPaths(Object):
    path_prefix = Path.T(optional=True)

    def __init__(self, *args, **kwargs):
        Object.__init__(self, *args, **kwargs)
        self._basepath = None
        self._parent_path_prefix = None

    def ichildren(self):
        for (prop, val) in self.T.ipropvals(self):
            if isinstance(val, HasPaths):
                yield val

            elif prop.multivalued and val is not None:
                for ele in val:
                    if isinstance(ele, HasPaths):
                        yield ele

    def set_basepath(self, basepath, parent_path_prefix=None):
        self._basepath = basepath
        self._parent_path_prefix = parent_path_prefix
        for val in self.ichildren():
            val.set_basepath(
                basepath, self.path_prefix or self._parent_path_prefix)

    def get_basepath(self):
        assert self._basepath is not None
        return self._basepath

    def change_basepath(self, new_basepath, parent_path_prefix=None):
        assert self._basepath is not None

        self._parent_path_prefix = parent_path_prefix
        if self.path_prefix or not self._parent_path_prefix:

            self.path_prefix = op.normpath(xjoin(xrelpath(
                self._basepath, new_basepath), self.path_prefix))

        for val in self.ichildren():
            val.change_basepath(
                new_basepath, self.path_prefix or self._parent_path_prefix)

        self._basepath = new_basepath

    def expand_path(self, path, extra=None):
        assert self._basepath is not None

        if extra is None:
            def extra(path):
                return path

        path_prefix = self.path_prefix or self._parent_path_prefix

        if path is None:
            return None
        elif isinstance(path, str):
            return extra(
                op.normpath(xjoin(self._basepath, xjoin(path_prefix, path))))
        else:
            return [
                extra(
                    op.normpath(xjoin(self._basepath, xjoin(path_prefix, p))))
                for p in path]


def expand_template(template, d):
    try:
        return Template(template).substitute(d)
    except KeyError as e:
        raise LassieError(
            'invalid placeholder "%s" in template: "%s"' % (str(e), template))
    except ValueError:
        raise LassieError(
            'malformed placeholder in template: "%s"' % template)


def bark():
    import subprocess
    subprocess.call(['aplay', data_file('bark.wav')])


__all__ = [
    'LassieError',
    'Earthmodel',
    'CakeEarthmodel',
    'HasPaths',
    'Path',
]
