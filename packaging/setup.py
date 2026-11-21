"""
Setup script for the PEARL care management offline routing and audit package.

PEARL: Policy Evolution through Aligned Retrospective Learning.
Basu S, Sheth P, Patel S. Under review, 2026.
"""
from setuptools import setup, find_packages

with open("../requirements.txt") as f:
    requirements = [
        line.strip() for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="pearl-care-management",
    version="2.0.0",
    description=(
        "PEARL: Policy Evolution through Aligned Retrospective Learning — "
        "four-safeguard offline routing and audit framework for care management."
    ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Sanjay Basu",
    author_email="sanjay.basu@ucsf.edu",
    url="https://github.com/sanjaybasu/pearl",
    license="MIT",
    packages=find_packages(where=".."),
    package_dir={"": ".."},
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "pearl-synthetic-demo=scripts.run_v6_synthetic_demo:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=[
        "care management", "next best action", "offline policy evaluation",
        "causal inference", "Medicaid", "intervention misalignment",
        "clinical decision support",
    ],
)
