import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = [
  'transformers',
  'torch',
  'pillow',
  'timm',
  'numpy',
  'opencv-python',
  'pytesseract'
]


setuptools.setup(
    name="clipcrop",
    version="2.4.3",
    author="Vishnu Nandakumar",
    author_email="nkumarvishnu25@gmail.com",
    description="Extract sections from your image by using OpenAI CLIP and Facebooks Detr implemented on HuggingFace Transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url = 'https://github.com/Vishnunkumar/clipcrop/',
    packages=[
        'clipcrop',
    ],
    package_dir={'clipcrop': 'clipcrop'},
    package_data={
        'clipcrop': ['clipcrop/*.py']
    },
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='clipcrop',
    classifiers=(
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    ),
)
