import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="srp_tomo",
    description="Shortest ray path weld map tomography",
    author="Michal Kalkowski",
    author_email="m.kalkowski@imperial.ac.uk",
    packages=find_packages(exclude=['data', 'references', 'output', 'notebooks']),
    long_description=read('README.md'),
    license='MIT'
)
