from setuptools import setup, find_packages

setup(
    name="data_assimilation_package",
    version="0.1.0",
    description="Assimilate station data into gridded data using deep learning.",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/your_username/data_assimilation_package",
    packages=find_packages(),
    install_requires=[
        "torch>=1.0.0",
        "numpy",
        "opencv-python",
        "matplotlib",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
