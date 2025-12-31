"""Setup script for moonshine-voice package."""

from setuptools import setup, find_packages
import os

# Read the README file


def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    with open(readme_path, "r", encoding="utf-8") as f:
        return f.read()

# Read the LICENSE file


def read_license():
    license_path = os.path.join(os.path.dirname(__file__), "LICENSE")
    with open(license_path, "r", encoding="utf-8") as f:
        return f.read()


# Read the requirements file
def read_requirements():
    requirements_path = os.path.join(
        os.path.dirname(__file__), "src", "moonshine_voice", "requirements.txt"
    )
    with open(requirements_path, "r", encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]


setup(
    name="moonshine-voice",
    version="0.0.4",
    description="Fast, accurate, on-device AI library for building interactive voice applications",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Moonshine AI",
    license="MIT",
    license_files=("LICENSE",),
    python_requires=">=3.8",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "moonshine_voice": [
            "*.dylib",
            "*.so",
            "*.dll",
            "assets/**/*",
        ],
    },
    install_requires=read_requirements(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    keywords=["speech", "transcription", "voice", "ai", "on-device", "stt"],
    url="https://github.com/moonshine-ai/moonshine",
    project_urls={
        "Homepage": "https://github.com/moonshine-ai/moonshine",
        "Documentation": "https://github.com/moonshine-ai/moonshine",
        "Repository": "https://github.com/moonshine-ai/moonshine",
        "Issues": "https://github.com/moonshine-ai/moonshine/issues",
    },
)
