import os
import subprocess
import sys

from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.command.egg_info import egg_info


class SetupAltoDevelop(develop):
    def run(self):
        develop.run(self)


class SetupAltoInstall(install):
    def run(self):
        install.run(self)


class SetupAltoEgg(egg_info):
    def run(self):
        egg_info.run(self)


setup(
    name='tuw-nlp',
    version='0.1',
    description='NLP tools at TUW Informatics',
    url='http://github.com/recski/tuw-nlp',
    author='Gabor Recski,Adam Kovacs',
    author_email='gabor.recski@tuwien.ac.at,adam.kovacs@tuwien.ac.at',
    license='MIT',
    install_requires=[
        'dict-recursive-update',
        'networkx',
        'penman',
        'stanza==1.1.1',
        'nltk',
        "graphviz"
    ],
    packages=find_packages(),
    scripts=['setup.sh'],
    include_package_data=True,
    cmdclass={'develop': SetupAltoDevelop,
              'install': SetupAltoInstall, "egg_info": SetupAltoEgg},
    zip_safe=False)
