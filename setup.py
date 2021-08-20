from setuptools import setup, find_packages

setup(
    name='buster',
    version='0.1.0',
    author='Kendall Park',
    author_email='kendall@cs.wisc.edu',
    packages=find_packages(),
    #  scripts=['bin/script1','bin/script2'],
    #  url='http://pypi.python.org/pypi/PackageName/',
    license='LICENSE',
    description='Something fun',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy", "scipy", "pytest", "pandas", "scikit-optimize", "matplotlib"
    ],
)