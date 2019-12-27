from setuptools import setup

setup(name='gym_kuka_multi_blocks',
      version='0.0.2',
      install_requires=['gym', 'pybullet==2.5.5', 'numpy', 'ray[all]', 'tensorflow>=2.0', 'boto3', 'pandas', 'scipy',
                        'matplotlib', 'seaborn']
)
