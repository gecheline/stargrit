from setuptools import setup, find_packages

setup(name='stargrit',
      version='0.1.2',
      description='STellAR General RadiatIve Transfer',
      url='https://github.com/gecheline/stargrit',
      author='Angela Kochoska',
      author_email='a.kochoska@gmail.com',
      license='MIT License',
      packages=find_packages(),
      package_data={'stargrit': ['geometry/tables/*.npy', 'geometry/tables/*.csv', 'atmosphere/tables/*.csv']},
      zip_safe=False)