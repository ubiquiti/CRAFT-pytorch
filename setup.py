# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name="craft",
    version="1.0.0",
    description="CRAFT: Character-Region Awareness For Text detection",
    # Choose your license
    license="MIT",
    packages=find_packages(exclude=["figures"]),
    include_package_data=True,
    install_requires=[
        "opencv-python>=4.2.0.32",
        "scikit-image>=0.14.2",
        "scipy>=1.1.0",
    ],
    entry_points={
        "console_scripts":[
            "craft=test_ui:main"
        ],
    },
)
