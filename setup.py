from setuptools import setup

import PMT_Measurments

__author__ = 'Frederik Andersen'
VERSION = "0.0.1"


setup(name='PMT_Measurements',
      version=VERSION,
      url='http://github.com/8me/ducroy/',
      author=__author__,
      author_email='frederik.andersen@fau.de',
      packages=['PMT_Measurment'],
      include_package_data=True,
      platforms='any',
      install_requires=[
          'pyvisa',
          'numpy',
        'matplotlib',
        'scipy',
      ],
      entry_points={
          'console_scripts': [
          ],
      },
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
      ],
)
