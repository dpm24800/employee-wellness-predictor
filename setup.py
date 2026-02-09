from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj: # as temporary file object
        requirements = file_obj.readlines()
        
    requirements = [req.replace("\n", "") for req in requirements]

    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
    name='employee-burnout-score-prediction-system',
    version='0.0.1',
    description='An End-to-end ML Project',
    author='Dipak Pulami Magar',
    author_email='dpm.it24800@gmail.com',
    packages=find_packages(),  # Automatically discover and include all packages
    # install_requires=['pandas', 'numpy', 'seaborn']
    install_requires=get_requirements('requirements.txt')
    # Other potential arguments: long_description, license, url, entry_points
)