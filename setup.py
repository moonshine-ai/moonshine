from pathlib import Path

import pkg_resources
from setuptools import find_packages, setup


def read_version(fname="moonshine/version.py"):
    my_locals = {}
    exec(
        compile(open(fname, encoding="utf-8").read(), fname, "exec"),
        globals(),
        my_locals,
    )
    return my_locals["__version__"]


setup(
    name="useful-moonshine",
    py_modules=["moonshine"],
    version=read_version(),
    description="Speech recognition for live transcription and voice commands.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    readme="README.md",
    python_requires=">=3.8",
    author="Useful Sensors",
    url="https://github.com/moonshine-ai/moonshine",
    license="MIT",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            Path(__file__).with_name("requirements.txt").open()
        )
    ],
    extras_require={
        "tensorflow": ["tensorflow==2.17.0"],
        "jax": ["jax==0.4.34", "keras==3.6.0"],
        "jax-cuda": ["jax[cuda12]", "keras==3.6.0"],
    },
    include_package_data=True,
)
