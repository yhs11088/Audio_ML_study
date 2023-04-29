'''
test custom environment
NOTE. custom environment must be registered to gym before running main.py
      (for more details, see setup.py)
'''

import gym
import veca

from stable_baselines3 import A2C, PPO

def main(env_name = "veca/Playground-v0", model_name = "A2C", mode = "train"):

    assert env_name in ["veca/Playground-v0", "veca/Playground-v1"]
    assert model_name in ["A2C", "PPO", "random"]
    assert mode in ["train", "test"]

    #---------------------------------------------------------------
    # 하이퍼파라미터
    # 1) environment
    room_width = 15
    room_height = 12

    # 2) 모델 학습
    total_timesteps = 15000
    model_path = f"{model_name}_{env_name.split('/')[-1]}"

    # 3) 테스트 경과 확인 파라미터
    test_iteration = 300
    print_interval = 10
    cum_reward = 0
    cum_count = 0

    #---------------------------------------------------------------
    if mode == "train":

        if model_name == "random":
            return

        # 학습 환경
        env = gym.make(
            env_name,
            room_width = room_width,
            room_height = room_height
        )

        # 모델 정의
        model = eval(f"{model_name}('MlpPolicy', env, verbose = True)")

        # 모델 학습
        model.learn(total_timesteps = total_timesteps)

        # 모델 저장
        model.save(model_path)
        del model

    #---------------------------------------------------------------
    else:

        # 모델 불러오기
        if model_name != "random":
            try:
                model = eval(f"{model_name}.load(model_path)")
            except:
                print(f"+++ {model_path}.zip does not exist! +++")

        # 테스트 환경
        env = gym.make(
            env_name,
            render_mode = "human",
            room_width = room_width,
            room_height = room_height
        )

        # 초기화
        if isgym21:
            observation = env.reset()  
        else:
            observation, info = env.reset() 


        # 테스트 환경 진행
        for i in range(test_iteration):

            if model_name == "random":
                # 랜덤 action
                action = env.action_space.sample()
            else:
                # 모델 action
                action, _states = model.predict(observation); action = int(action)

            # 진행
            if isgym21:
                observation, reward, done, info = env.step(action)
            else:
                observation, reward, terminated, truncated, info = env.step(action)
                done = (terminated or truncated)

            # 누적 reward
            cum_reward += reward
            cum_count += 1
            if (i+1) % print_interval == 0:
                print(f"Iteration {i+1} : recent mean reward = {cum_reward / cum_count:.3f}")
                cum_reward = 0
                cum_count = 0

            # 재초기화
            if done:
                if isgym21:
                    observation = env.reset() 
                else:
                    observation, info = env.reset() 

                # 누적 reward 초기화
                if cum_count > 0:
                    print(f"Iteration {i+1} : recent mean reward = {cum_reward / cum_count:.3f}")
                cum_reward = 0
                cum_count = 0
                print("-" * 20)

        env.close()

if __name__ == "__main__":

    # 하이퍼파라미터
    isgym21 = gym.__version__ == "0.21.0"
    env_name = "veca/Playground-v0" if isgym21 else "veca/Playground-v1"
    model_name = "random"

    # 학습
    print(" Train ".center(50, "="))
    main(env_name, model_name, mode = "train")

    # 테스트
    print(" Test ".center(50, "="))
    main(env_name, model_name, mode = "test")