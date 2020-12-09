import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "drl_negotiation",
    version = "0.0.3",
    author = "naodongbanana",
    author_email = "n1085633848@outlook.com",
    description = "environment for negotiation in negmas",
    long_description = long_description,
    url = "https://github.com/YueNing/tn_source_code",
    packages = setuptools.find_packages(),
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)