"""
Setup script for the PEARL care management AI package.

PEARL: Policy Evolution through Aligned Retrospective Learning
Next best action selection for rising-risk ACO care management.

Basu S, et al. PEARL: AI-Guided Next Best Action Selection for
Rising-Risk Medicaid ACO Care Management.
Lancet Digital Health, 2026. [In submission]
"""
from setuptools import setup, find_packages

with open("../requirements.txt") as f:
    requirements = [
        line.strip() for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="pearl-care-management",
    version="1.0.0",
    description=(
        "PEARL: Policy Evolution through Aligned Retrospective Learning — "
        "next best action selection for ACO care management using "
        "within-patient causal identification and tabular IPTW-DPO."
    ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Sanjay Basu",
    author_email="sanjaybasu@waymark.com",
    url="https://github.com/sanjaybasu/pearl",
    license="MIT",
    packages=find_packages(where=".."),
    package_dir={"": ".."},
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "notebook": [
            "jupyter>=1.0",
            "ipykernel>=6.0",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "ruff>=0.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "pearl-pipeline=scripts.run_pipeline:main",
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
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=[
        "care management", "next best action", "reinforcement learning",
        "DPO", "causal inference", "Medicaid", "ACO", "health equity",
        "intervention misalignment", "WPAD",
    ],
)
