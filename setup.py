import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="biobert-embedding",
    packages=['biobert_embedding'],
    version="0.1.4",
    author="Jitendra Jangid, Ariel Lubonja",
    author_email="jitujangid38@gmail.com, ariellubonja@live.com",
    description="Embeddings from BioBERT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ariellubonja/biobert_embedding",
    download_url="https://github.com/ariellubonja/biobert_embedding/archive/v0.1.3.tar.gz",
    install_requires=[
          'torch==2.1.2',
          'pytorch-pretrained-bert==0.6.2',
      ],

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
