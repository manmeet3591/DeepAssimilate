from setuptools import setup, find_packages

setup(
    name="DeepAssimilate",
    version="0.1.0",
    description="Deep learning framework for assimilating data with deep learning.",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "torch>=1.7.1",
        "numpy",
        "opencv-python",
        "scikit-learn"
    ],
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
