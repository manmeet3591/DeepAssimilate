from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

# Read requirements file
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    with open(requirements_path, "r", encoding="utf-8") as f:
        # Filter out comments and empty lines
        requirements = []
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                # Remove inline comments
                if " #" in line:
                    line = line.split(" #")[0].strip()
                requirements.append(line)
        return requirements

setup(
    name="deepassimilate",
    version="0.1.0",
    author="DeepAssimilate Team",
    author_email="",  # Update with your email
    description="Modular framework for training unconditional diffusion models and applying them to generative data assimilation (GDA)",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/DeepAssimilate",  # Update with your repo URL
    packages=find_packages(exclude=["legacy", "tests", "examples"]),
    install_requires=read_requirements(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    keywords="diffusion-models, data-assimilation, weather-forecasting, machine-learning, scientific-ml",
)

