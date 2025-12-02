from setuptools import find_packages, setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(
    name="colorizer",
    version="0.1.0",
    description="Cat Image Colorization with Deep Learning",
    license="MIT",
    author="author Name",
    author_email="your@email.com",
    install_requires=requirements,
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.9",
)
