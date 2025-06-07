import setuptools

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="image_inference_project",
    version="0.1",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    python_requires='>=3.6',
)
