import setuptools

# Display message about model checkpoints
print("="*80)
print("IMPORTANT: DSRNet Model Checkpoints")
print("="*80)
print("To use the inference.py script, you need to download the DSRNet model checkpoints manually.")
print("Please find the download links in the README.md file.")
print("Place the downloaded checkpoint files (e.g., dsrnet_s_epoch14.pt, dsrnet_l_epoch18.pt) into the 'weights/' directory.")
print("Ensure the default checkpoint 'weights/dsrnet_s_epoch14.pt' exists or provide a different one via --checkpoint_path.")
print("="*80)
print("\nStarting package setup...")

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="image_inference_project",
    version="0.1",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    python_requires='>=3.6',
    long_description="""\
A project for image inference using DSRNet.
IMPORTANT: Model checkpoints must be downloaded manually. See README.md for details and place them in the 'weights/' directory.
""",
    long_description_content_type="text/plain",
)

print("Package setup complete. If you haven't already, please download the model checkpoints as instructed above.")
