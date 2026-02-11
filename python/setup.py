"""Setup script for moonshine-voice package."""

import os

from setuptools import setup, find_packages, Distribution
from wheel.bdist_wheel import bdist_wheel


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


class PlatformWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        # Mark as platform-specific (not pure Python)
        self.root_is_pure = False

    def get_tag(self):
        python, abi, plat = super().get_tag()
        # Use generic Python tag but platform-specific
        return "py3", "none", plat


def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    with open(readme_path, "r", encoding="utf-8") as f:
        return f.read()


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
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="moonshine-voice",
    version="0.0.48",
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
    cmdclass={"bdist_wheel": PlatformWheel},
    distclass=BinaryDistribution,
)
