from setuptools import setup, find_packages

setup(
    name='gns_tools',
    version='0.1.0',
    description='A general module for Graph Neural Network simulation tools.',
    author='Ashton Cole',
    author_email='ashtoncole1028@gmail.com',
    url='https://github.com/ashtonvcole/TrajAI/tree/main',
    packages=find_packages(),
    install_requires=[
        'torch>=1.10.0',
        'torch-geometric>=2.0.0',
        'torch-scatter>=2.0.0',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License', # Choose your license
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.8',
)
