from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='Road',
    url='https://github.com/nemodrive/semantic_data_augmentation',
    author='Dragos Homner',
    author_email='dragoshomner@yahoo.com',
    # Needed to actually package something
    packages=['road-package'],
    # Needed for dependencies
    install_requires=['numpy'],
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='ABC',
    description='Overlay people on the road',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)