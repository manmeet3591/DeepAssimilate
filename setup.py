from setuptools import setup, find_packages

# Read dependencies from requirements.txt
with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="DeepAssimilate",
    version="0.1.0",
    description="Deep learning framework for assimilating data with SRCNN.",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=required,  # Use the dependencies from requirements.txt
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
