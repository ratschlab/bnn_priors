from setuptools import setup

setup(
    name="bnn_priors",
    version="0.1.0",
    description="investigations on BNN priors",
    long_description="",
    author="Vincent Fortuin, Adrià Garriga-Alonso",
    author_email="adria.garriga@gmail.com",
    url="https://github.com/ratschlab/projects2020_BNN-priors/",
    license="MIT license",
    packages=["bnn_priors"],
    install_requires=[
        "torch>=1.5.0<1.6",
        "torchvision>=0.6<0.7",
        "tensorboardX>=2.0<2.1",
        "sacred>=0.8<0.9",
        "gpytorch>=1.0<1.2",
        "pyro-ppl>=1.3<1.4",
        # Coarse dependencies
        "numpy>=1.0<2.0",
        "scipy>=1.5.2<1.6",
        "tqdm>=4.0<5.0",
        "matplotlib>=3.0<4.0",
    ],
    test_suite="testing",
)
