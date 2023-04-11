# from distutils.core import setup

# setup(
#   name = 'clipcrop',         # How you named your package folder (MyLib)
#   packages = ['clipcrop'],   # Chose the same as "name"
#   version = '1.0',      # Start with a small number and increase it with every change you make
#   license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
#   description = "Extract sections from your image by using OpenAI CLIP and Facebooks Detr implemented on HuggingFace Transformers",   # Give a short description about your library
#   author = 'Vishnu N',                   # Type in your name
#   author_email = 'vishnunkumar25@gmail.com',      # Type in your E-Mail
#   url = 'https://github.com/Vishnunkumar/clipcrop/',   # Provide either the link to your github or to your website
#   download_url ='https://github.com/Vishnunkumar/clipcrop/archive/refs/tags/v-1.0.tar.gz',    # I explain this later on
#   keywords = ['Documents', 'Machine learning', 'NLP', 'Deep learning', 'Computer Vision'],   # Keywords that define your package best
#   install_requires = [            # I get to this in a second
#           'transformers',
#           'torch',
#           'pillow',
#           'timm',
#           'numpy',
#           'opencv-python'
#   ],
#   classifiers=[
#     'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
#     'Intended Audience :: Developers',      # Define that your audience are developers
#     'Topic :: Software Development :: Build Tools',
#     'License :: OSI Approved :: MIT License',   # Again, pick a license
#     'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
#     'Programming Language :: Python :: 3.4',
#     'Programming Language :: Python :: 3.5',
#     'Programming Language :: Python :: 3.6',
#   ],
# )


import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = [
  'transformers',
  'torch',
  'pillow',
  'timm',
  'numpy',
  'opencv-python'
]


setuptools.setup(
    name="clipcrop",
    version="1.0.3",
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
