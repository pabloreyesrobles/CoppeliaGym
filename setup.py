from setuptools import setup, find_packages
import sys
import os.path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CoppeliaGym'))

setup(
	name='CoppeliaGym', # Replace with your own username
	version='0.0.1',
	author='Pablo Reyes Robles',
	author_email='pabloreyes500@gmail.com',
	description='A CoppeliaGym implementation of OpenAI for custom scenes.',
	packages=find_packages(),
	classifiers=[
		'Programming Language :: Python :: 3',
		'License :: OSI Approved :: MIT License',
		'Operating System :: OS Independent',
	],
	install_requires=[
		'numpy',
		'gym'
	],
	python_requires='>=3.8',
    include_package_data=True,
    package_data={'CoppeliaGym': ['envs/robots/*.ttt']}
)