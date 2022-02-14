#!/usr/bin/env python

import time
import os.path as op

from setuptools import setup
from setuptools.command.install import install
from setuptools.command.build_py import build_py

version = '0.1'


class CustomInstallCommand(install):
    def run(self):
        install.run(self)


class NotInAGitRepos(Exception):
    pass


class CustomBuildPyCommand(build_py):

    def git_infos(self):
        '''Query git about sha1 of last commit and check if there are local \
           modifications.'''

        from subprocess import Popen, PIPE
        import re

        def q(c):
            return Popen(c, stdout=PIPE).communicate()[0]

        if not op.exists('.git'):
            raise NotInAGitRepos()

        sha1 = q(['git', 'log', '--pretty=oneline', '-n1']).split()[0]
        sha1 = re.sub(br'[^0-9a-f]', '', sha1)
        sha1 = str(sha1.decode('ascii'))
        sstatus = q(['git', 'status', '--porcelain', '-uno'])
        local_modifications = bool(sstatus.strip())
        return sha1, local_modifications

    def make_info_module(self, packname, version):
        '''Put version and revision information into file src/setup_info.py.'''

        sha1, local_modifications = None, None
        combi = '%s_%s' % (packname, version)
        try:
            sha1, local_modifications = self.git_infos()
            combi += '_%s' % sha1
            if local_modifications:
                combi += '_modified'

        except (OSError, NotInAGitRepos):
            pass

        datestr = time.strftime('%Y-%m-%d_%H:%M:%S')
        combi += '_%s' % datestr

        module_code = '''# This module is automatically created from setup.py
git_sha1 = %s
local_modifications = %s
version = %s
long_version = %s  # noqa
installed_date = %s
''' % tuple([repr(x) for x in (
            sha1, local_modifications, version, combi, datestr)])

        outfile = self.get_module_outfile(
            self.build_lib, ['silvertine'], 'setup_info')

        dir = op.dirname(outfile)
        self.mkpath(dir)
        with open(outfile, 'w') as f:
            f.write(module_code)

    def run(self):
        self.make_info_module('silvertine', version)
        build_py.run(self)


setup(
    cmdclass={
        'build_py': CustomBuildPyCommand,
        'install': CustomInstallCommand,
    },

    name='silvertine',

    description='Monitoring, location and analysis of induced earthquakes',

    version=version,

    author='The silvertine Developers',

    author_email='andreas.steinberg@bgr.de',

    packages=[
        'silvertine',
        'silvertine.apps',
        'silvertine.clustering',
        'silvertine.detector',
        'silvertine.detector.core',
        'silvertine.detector.utils',
        'silvertine.detector.locator',
        'silvertine.scenario',
        'silvertine.locate',
        'silvertine.minimum_1d',
        'silvertine.util',
        'silvertine.plot',
        'silvertine.report',
        'silvertine.seiger_lassie',
        'silvertine.shakemap',
        'silvertine.mechanism',
        'silvertine.monitoring',
        'silvertine.beam_depth',
        'silvertine.seismic_handler',
    ],
    entry_points={
        'console_scripts': [
            'silvertine = silvertine.apps.silvertine:main',
            'seigerdown = silvertine.apps.seigerdown:main',
            'seiger-lassie = silvertine.apps.seiger_lassie:main',

        ]
    },
    package_dir={'silvertine': 'src'},

    package_data={
        'silvertine': [
            'report/app/*.html',
            'report/app/favicon.png',
            'report/app/templates/*.html',
            'report/app/css/*.css',
            'report/app/js/*.js',
            'report/app/configs/*.yaml',
            'mechanism/configs/*.yaml',

            'data/snippets/*.gronf',
            'data/snippets/*.md',
            'data/examples/*/*.*',
            'data/examples/*/*/*.*',
            'detector/model/*h5',
            'detector/json/station_list.json',
            'data/examples/*/*/silvertineown',
            ]},

    data_files=[],

    license='GPLv3',

    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: Implementation :: CPython',
        'Operating System :: POSIX',
        'Operating System :: MacOS',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Application Frameworks',
        ],

    keywords=[
        'seismology, waveform analysis, earthquake modelling, geophysics,'
        ' geophysical inversion'],
    )
