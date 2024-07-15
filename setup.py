from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="minihuman",
    version="0.1.2",
    author="human.ai.integration",
    author_email="human.ai.integration@gmail.com",
    description="MiniHuman: A Python Library to Simulate Virtual Human Behaviors and Responses to Environmental Stimuli",
    long_description="MiniHuman: A Python Library to Simulate Virtual Human Behaviors and Responses to Environmental Stimuli",
    long_description_content_type="text/markdown",
    index="https://github.com/songlinxu/MiniHuman-Toolkit",
    project_urls={
        "Bug Tracker": "https://github.com/songlinxu/MiniHuman-Toolkit",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.5",
    include_package_data=True,
    install_requires=['numpy>=1.17.2','pandas>=1.1.5','scikit-learn'],
)