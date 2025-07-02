#!/usr/bin/env python

"""
setup.py file for LetsBeRational
"""

#import sys; print("\nPython invoked from %s." % sys.executable);

####################################################################
#                                                                  #
#  Build a binary wheel from prebuilt binaries.                    #
#                                                                  #
####################################################################
#                                                                  #
#  https://pypi.org/project/prebuilt-binaries/                     #
#  License: MIT License                                            #
#  Author: Tim Mitchell                                            #
#                                                                  #
####################################################################

import os,pathlib,warnings
from setuptools.dist import Distribution
from setuptools.command import build_ext
from distutils.file_util import copy_file
from setuptools import setup,Extension,SetuptoolsDeprecationWarning

# The 'keep_name' extension is by Peter Jaeckel, 2023-12-16.

class PrebuiltExtension(Extension):
    def __init__(self, input_file, package = None, keep_name = False):
        name = pathlib.Path(input_file).stem
        if package is not None:
            name = f'{package}.{name}'
        if not os.path.exists(input_file):
            raise ValueError(f'Prebuilt extension file not found\n{input_file}')
        self.input_file = input_file
        self.keep_name = keep_name
        super().__init__(name, ['no-source-needed.c'])


class prebuilt_binary(build_ext.build_ext):
    def run(self):
        for ext in self.extensions:
            if ext.keep_name:
                filename = ext.input_file
            else:
                fullname = self.get_ext_fullname(ext.name)
                filename = self.get_ext_filename(fullname)
            dest_filename = os.path.join(self.build_lib, filename)
            dest_folder = os.path.dirname(dest_filename)
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)
            copy_file(
                ext.input_file, dest_filename, verbose=self.verbose,
                dry_run=self.dry_run
            )
#           print("Added binary {} as {} to wheel.".format(ext.input_file,dest_filename))
        if self.inplace:
            self.copy_extensions_to_source()

####################################################################
#                                                                  #
#  End of code to build binary wheel from prebuilt binaries.       #
#                                                                  #
####################################################################

warnings.filterwarnings("ignore", category=SetuptoolsDeprecationWarning) 
# Suppress bdist_wheel.py:108: RuntimeWarning: Config variable 'Py_DEBUG' is unset, Python ABI tag may be incorrect
warnings.filterwarnings("ignore", category=RuntimeWarning)

module_name = os.getenv('MODULE_NAME')

binary_modules = [PrebuiltExtension(os.getenv('WHEEL_MAIN_BINARY'))]

for f in os.getenv('WHEEL_DEPENDENCY_BINARIES').split():
    binary_modules.append(PrebuiltExtension(f,keep_name=True))

for f in os.getenv('WHEEL_OPTIONAL_BINARIES').split():
    if os.path.isfile(f): # Only attempt to include those that are present.
        binary_modules.append(PrebuiltExtension(f,keep_name=True))

revision = os.getenv('REVISION')

print('Building wheel for {} revision {}.'.format(module_name,revision))

result = setup(
    cmdclass = { 'build_ext': prebuilt_binary },
    has_ext_modules = lambda : True,
    options = {"bdist_wheel": {"universal": True}},
    name = module_name,
    version = '1.0.{}'.format(revision),
    author      = "Peter Jaeckel",
    description = """This is the reference implementation of the implied Black volatility computation algorithm published in "Let's Be Rational" by Peter Jaeckel, © 2013-2023.""",
#    long_description=open('README.txt').read(),
    long_description = """This is the reference implementation of the implied Black volatility computation algorithm published in "Let's Be Rational" by Peter Jaeckel, © 2013-2023.\n\nSee www.jaeckel.org/LetsBeRational.pdf for details of the mathematics published in November 2013; Wilmott, pp 40-53, January 2015.\n\n""",
    url='http://www.jaeckel.org',
    maintainer='Peter Jaeckel',
    maintainer_email='pj@otc-analytics.com',
    license="MIT",
    py_modules = [module_name],
    ext_modules = binary_modules
)

# See https://stackoverflow.com/questions/51939257/how-do-you-get-the-filename-of-a-python-wheel-when-running-setup-py
# print( '{}-{}.whl'.format(result.command_obj['bdist_wheel'].wheel_dist_name,'-'.join(result.command_obj['bdist_wheel'].get_tag())) )
