from setuptools import setup


setup(
      name="acoustdsp",
      version="0.0.1",
      author="Hannes Rosseel",
      packages=["acoustdsp", "acoustdsp.utils"],
      install_requires=["numpy", "numba", "matplotlib", "scipy"]
)
