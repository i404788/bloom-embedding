from setuptools import setup, find_packages

setup(
  name = 'bloom-embedding',
  packages = find_packages(exclude=[]),
  version = '0.1.0',
  license='MIT',
  description = 'Bloom Embedding - Pytorch',
  author = 'Ferris Kwaijtaal',
  author_email = 'ferris+gh@devdroplets.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/i404788/bloom-embedding',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism',
    'audio generation'
  ],
  install_requires=[
    'torch>=1.13',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)

