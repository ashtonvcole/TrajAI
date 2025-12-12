from setuptools import setup, find_packages

setup(
    name='traj_ai',
    version='0.1.0',
    description='A general module for objective dynamics Graph Neural Simulator simulations.',
    author='Ashton Cole',
    author_email='ashtoncole1028@gmail.com',
    url='https://github.com/ashtonvcole/TrajAI/tree/main',
    packages=find_packages(),
    install_requires=[
        'torch>=1.10.0',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.13',
    ],
    python_requires='>=3.8',
)
