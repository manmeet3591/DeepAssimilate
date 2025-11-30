from setuptools import setup, find_packages

# Read the requirements file
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
requirements_path = os.path.join(current_dir, "requirements.txt")

with open(requirements_path) as f:
    required = f.read().splitlines()

setup(
    name="DeepAssimilate",  # Replace with your package name
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Description of DeepAssimilate package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/DeepAssimilate",
    packages=find_packages(),
    install_requires=required,  # Use the parsed requirements
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
