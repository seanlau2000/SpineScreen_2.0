from setuptools import setup, find_packages

setup(
    name="screening-bert",
    version="0.1.0",
    description="ClinicalBERT-based screening model for surgical decision support",
    author="Sean Lau",
    python_requires=">=3.9",

    # This tells setuptools you are using the src/ layout
    package_dir={"": "src"},
    packages=find_packages(where="src"),

    install_requires=[
        "transformers",
        "torch",
        "numpy",
        "pandas",
        "scikit-learn",
        "accelerate",
        "optuna",
        "msoffcrypto-tool",
        "openpyxl",
        "scipy",
        "matplotlib",
    ],
)
