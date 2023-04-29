'''
main reference : https://www.gymlibrary.dev/content/environment_creation/

==============================================
setup 방법
==============================================
1. my_gym_env 폴더로 이동
2. >> pip install -e .

==============================================
정의한 environment 사용 방법
==============================================
1. python에서 다음을 실행
import gym
import veca
env = gym.make("veca/[environment id]", ...)   # environment id는 veca/__init__.py에서 정의됨
'''

from setuptools import setup

setup(
    name = "veca",
    version = "0.0.1",
    #install_requires = ["gym==0.26.0", "pygame==2.1.0"],
    install_requires = ["gym==0.21.0", "pygame==2.1.0"],
)