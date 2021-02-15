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
        "h5py>=2.10<2.11",
        "sacred>=0.8<0.9",
        "gpytorch>=1.0<1.2",
        "pyro-ppl>=1.3<1.4",

        # Current metrics storage relies on a bug in numpy 1.18.5 where NaN is
        # automatically cast to -2**64 if written to an empty array of type int64.
        "numpy>=1.18<1.19",

        # Coarse dependencies
        "scipy>=1.5.2<1.6",
        "tqdm>=4.0<5.0",
        "matplotlib>=3.0<4.0",
    ],
    test_suite="testing",
)
