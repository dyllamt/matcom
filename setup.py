import os

from setuptools import setup, find_packages

module_dir = os.path.dirname(os.path.abspath(__file__))
setup(name='matcom',
      version='0.1',
      description='community detection in materials databases',
      author='Maxwell Dylla',
      license='MIT',
      packages=find_packages(),
      install_requires=[],
      long_description=open('readme.md').read())
