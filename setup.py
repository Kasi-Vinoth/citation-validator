from setuptools import setup, find_packages

setup(
    name="citation-validator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "sentence-transformers>=2.2.2",
        "numpy>=1.20.0",
        "google-cloud-aiplatform>=1.38.0",  # Include dependencies
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # Specify Python version
)
