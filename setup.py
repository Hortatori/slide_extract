from setuptools import setup

setup(name='slide_extract',
      description='text extraction from news',
      url='https://github.com/Hortatori/slide_extract',
      author='Marjolaine Ray',
      author_email='marjolaine.ray@ens.psl.eu',
      install_requires=[
        "numpy==2.2.5",
        "pandas==2.2.3",
        "pytz==2022.1",
        "scikit_learn==1.6.1",
        "sentence_transformers==3.2.0",
        "torch==2.4.1",
        "tqdm==4.66.4"
      ],
      zip_safe=False)