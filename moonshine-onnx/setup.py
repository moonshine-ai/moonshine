from pathlib import Path

import pkg_resources
from setuptools import setup


def read_version(fname="src/version.py"):
    my_locals = {}
    exec(
        compile(open(fname, encoding="utf-8").read(), fname, "exec"),
        globals(),
        my_locals,
    )
    return my_locals["__version__"]


setup(
    name="useful-moonshine-onnx",
    packages=["moonshine_onnx"],
    package_dir={"moonshine_onnx": "src"},
    version=read_version(),
    description="Speech recognition for live transcription and voice commands with the Moonshine ONNX models.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    readme="README.md",
    python_requires=">=3.8",
    author="Useful Sensors",
    url="https://github.com/moonshine-ai/moonshine",
    license="MIT",
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            Path(__file__).with_name("requirements.txt").open()
        )
    ],
    include_package_data=True,
)
