import pathlib

from setuptools import find_packages, setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="simulator-epi-models",
    version="0.1.1",
    description="Epi models used by the Simulator web app",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/AIforGoodSimulator/epi-models",
    author="Crisis Modelling Devs",
    author_email="billy@aiforgoodsimulator.com",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "scipy",
        "numpy",
        "pandas",
    ],
    zip_safe=False,
    include_package_data=True,
)
