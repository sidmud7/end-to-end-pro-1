from setuptools import find_packages, setup
from typing import List

# Define a constant for the '-e .' flag in requirements
HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    '''
    This function will return the list of requirements.
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        # Remove newline characters and the '-e .' entry
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
    name='MLproject',
    version='0.0.1',
    author='Siddhartha',
    author_email='siddhartha.main@gmail.com', # Replace with your email
    packages=find_packages(), # Automatically finds packages (folders with __init__.py)
    install_requires=get_requirements('requirements.txt') # Installs dependencies from requirements.txt
)