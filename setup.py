from setuptools import setup

with open('README.md', 'r') as f:
    long_description=f.read()

setup(
    name='simple-nn',
    version='0.2.0',
    description='A Simple, Handwritten Neural Network',
    long_description=long_description,
    license='MIT',
    author='matt clarson',
    author_email='clarson469@googlemail.com',
    packages=['simplenn'],
    install_requires=['numpy']
)
