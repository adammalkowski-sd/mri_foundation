from setuptools import setup, find_packages

setup(
    name="mri_core",
    version="0.0.4",
    packages=find_packages(),
    install_requires=[
        "einops",
        "scikit-image",
        "monai==1.5",
        "torch",
        "torchvision",
        "torchaudio",
    ],
)
