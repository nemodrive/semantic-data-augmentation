from setuptools import setup

dependency_links=[
	'git+https://github.com/nemodrive/semantic_data_augmentation',
]

setup(
    name="roadpackage",
    version="0.0.1",
    author="Dragos",
    author_email="draogs@example.com",
    description="A small example package",
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=['roadpackag'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)