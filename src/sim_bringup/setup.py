from setuptools import find_packages, setup
import glob

package_name = 'sim_bringup'

data_files = [
    ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
]

# Install launch files
launch_files = glob.glob('launch/*.py')
if launch_files:
    data_files.append(('share/' + package_name + '/launch', launch_files))

config_files = glob.glob('config/*.yaml')
if config_files:
    data_files.append(('share/' + package_name + '/config', config_files))

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    package_dir={package_name: 'src'},
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='somo',
    maintainer_email='sunnycat_158@qq.com',
    description='Sentry simulation bringup package',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={'console_scripts': []},
)
