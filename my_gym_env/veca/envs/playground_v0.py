'''
Simple playground environment v0
- assumes gym==0.21.0
- stable-baselines3 models can be applied

main reference : https://www.gymlibrary.dev/content/environment_creation/
'''

import sys
import numpy as np
import pygame
import gym
from gym.spaces import Box, Discrete

class PlaygroundEnv_v0(gym.Env):
    #------------------------------------------------------
    metadata = {
        "render_mode" : ["human", "rgb_array"],
        "render_fps" : 4
    }

    #------------------------------------------------------
    def __init__(
            self, render_mode = None,
            room_width = 10, room_height = 10,
        ):

        assert (render_mode is None) or (render_mode in self.metadata['render_mode'])

        self.render_mode = render_mode   # 시각화 모드
        self.room_width  = room_width    # 방 가로 길이 (단위 : 칸)
        self.room_height = room_height   # 방 세로 길이 (단위 : 칸)
        self.grid_size   = 25            # 한 칸 길이 (단위 : pixel)

        # observation : [x_agent, y_agent, x_target, y_target, dx, dy, dx2, dy2]
        # - x_agent : agent의 x 좌표 (= 0, 1, ..., room_width-1)
        #   y_agent : agent의 y 좌표 (= 0, 1, ..., room_height-1)
        #   x_target : target의 x 좌표 (= 0, 1, ..., room_width-1)
        #   y_target : target의 y 좌표 (= 0, 1, ..., room_height-1)
        #   dx : agent의 x 방향 이동 좌표 (= -1, 0, 1)
        #   dy : agent의 y 방향 이동 좌표 (= -1, 0, 1)
        #   dx2 : agent의 이전 x 방향 이동 좌표 (= -1, 0, 1)
        #   dy 2: agent의 이전 y 방향 이동 좌표 (= -1, 0, 1)
        self.observation_space = Box(
            low   = np.array([0, 0, 0, 0, -1, -1, -1, -1]),
            high  = np.array([room_width-1, room_height-1, room_width-1, room_height-1, 1, 1, 1, 1]),
            shape = (8,),
            dtype = int
        )

        # action : {0, 1, 2, 3} 중 하나
        # - 0 = 상 / 1 = 하 / 2 = 좌 / 3 = 우
        self.action_space = Discrete(4)

        # (action -> 방향벡터) 변환 dictionary
        self._action_to_direction = {
            0 : np.array([0, 1]),
            1 : np.array([0, -1]),
            2 : np.array([-1, 0]),
            3 : np.array([1, 0]),
        }
        
        # agent & target 위치
        self._agent_location  = np.array([0, 0], dtype = int)  # [x_agent, y_agent]
        self._target_location = np.array([0, 0], dtype = int)  # [x_target, y_target]

        # agent 현재 & 이전 방향벡터
        self._direction = np.zeros(2)           # 원래는 상/하/좌/우 중 하나이지만 0벡터로 초기화
        self._previous_direction = np.ones(2)   # 원래는 상/하/좌/우 중 하나이지만 1벡터로 초기화

        self.window = None
        self.clock = None

    #------------------------------------------------------
    def _get_obs(self):
        return np.concatenate([self._agent_location, self._target_location, self._direction, self._previous_direction])

    #------------------------------------------------------
    def reset(self, seed = None):
        # gym.Env.np_random에 seed 전달
        #super().reset(seed = seed)      # for gym==0.21.0

        # agent 위치 초기화
        #self._agent_location = self.np_random.integers(
        self._agent_location = np.random.randint(
            low   = np.array([0, 0]),
            high  = np.array([self.room_width, self.room_height]),
            size  = 2,
            dtype = int
        )

        # target 위치 초기화 (agent 위치와 다르게)
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            #self._agent_location = self.np_random.integers(
            self._agent_location = np.random.randint(
                low   = np.array([0, 0]),
                high  = np.array([self.room_width, self.room_height]),
                size  = 2,
                dtype = int
            )

        # agent 현재 & 이전 방향벡터 초기화
        self._direction = np.zeros(2)           # 원래는 상/하/좌/우 중 하나이지만 0벡터로 초기화
        self._previous_direction = np.ones(2)   # 원래는 상/하/좌/우 중 하나이지만 1벡터로 초기화

        # observation
        observation = self._get_obs()

        # info
        info = {}

        # PyGame window 반영
        if self.render_mode == "human":
            self._render_frame()
        
        return observation       # for gym==0.21.0
    
    #------------------------------------------------------
    def _get_reward(self, previous_agent_location):

        # action이 진동하는지 여부
        oscillating = np.allclose(self._direction, -self._previous_direction, atol = 1e-8)

        # 정지해있는지 여부
        stopped = np.array_equal(self._agent_location, previous_agent_location)

        # 택시 거리
        previous_dist = abs(self._target_location - previous_agent_location).sum()
        dist          = abs(self._target_location - self._agent_location).sum()
        max_dist      = self.room_width + self.room_height

        # reward 정의
        # - 방법 1
        #reward = 1 if terminated else 0
        # - 방법 2
        #reward = np.exp(-1. * dist)
        # - 방법 3
        #reward = 1 - (dist / max_dist)
        # - 방법 4 
        '''
        if previous_dist < dist:
            #reward = -0.1
            reward = - (dist - previous_dist) / max_dist
        elif previous_dist > dist:
            reward = 1 - (dist / max_dist)
        else:
            reward = 0
        '''
        # 방법 5
        if oscillating or stopped:
            reward = -0.25
        elif dist > previous_dist:
            reward = -0.75
        else:
            reward = 1 - dist / max_dist
        
        return reward

    #------------------------------------------------------
    def step(self, action):

        # (optional) 이전 위치 & 방향벡터 저장 (reward 계산시 사용)
        previous_agent_location = self._agent_location
        self._previous_direction[:] = self._direction[:]

        # (action -> 방향벡터) 변환
        self._direction = self._action_to_direction[action]

        # agent 위치 이동 (room 범위를 벗어날 시 clip)
        self._agent_location = np.clip(
            self._agent_location + self._direction,
            a_min = [0, 0],
            a_max = [self.room_width-1, self.room_height-1]
        )

        # terminated
        terminated = np.array_equal(self._agent_location, self._target_location)

        # observation
        observation = self._get_obs()

        # reward
        reward = self._get_reward(previous_agent_location)
    
        # truncated
        truncated = False

        # info
        info = {}

        # PyGame window 반영
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, info     # for gym==0.21.0
    
    #------------------------------------------------------
    #def render(self, mode = None):
    def render(self, mode = None):          # for gym==0.21.0
        if self.render_mode == "rgb_array":
            return self._render_frame()

    #------------------------------------------------------  
    def _render_frame(self):

        grid_size = self.grid_size  # 격자 한 칸 길이 (단위 : pixel)

        if self.render_mode == "human":
            # window를 정의하지 않았다면 PyGame window 초기화
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode(
                    size = (self.room_width * grid_size, self.room_height * grid_size)
                )

            # 시계를 정의하지 않았다면 시계 초기화
            if self.clock is None:
                self.clock = pygame.time.Clock()

        # canvas 초기화
        canvas = pygame.Surface(
            size = (self.room_width * grid_size, self.room_height * grid_size)
        )
        canvas.fill(color = (255, 255, 255)) # 흰색

        # agent 그리기
        x_agent, y_agent = self._agent_location
        pygame.draw.rect(
            canvas,
            color = (255, 0, 0),
            rect  = pygame.Rect(
                (x_agent * grid_size, y_agent * grid_size),
                (grid_size, grid_size)
            )
        )

        # target 그리기
        x_target, y_target = self._target_location
        pygame.draw.rect(
            canvas,
            color = (0, 0, 255),
            rect  = pygame.Rect(
                (x_target * grid_size, y_target * grid_size),
                (grid_size, grid_size)
            )
        )

        # 격자선 그리기
        # 1) 세로선
        for x in range(self.room_width+1):
            pygame.draw.line(
                canvas,
                color     = (0, 0, 0),
                start_pos = (x * grid_size, 0),
                end_pos   = (x * grid_size, self.room_height * grid_size),
                width     = 1
            )
        # 2) 가로선
        for y in range(self.room_height+1):
            pygame.draw.line(
                canvas,
                color     = (0, 0, 0),
                start_pos = (0, y * grid_size),
                end_pos   = (self.room_width * grid_size, y * grid_size),
                width     = 1
            )

        if self.render_mode == "human":

            # **** test ****
            # source : https://stackoverflow.com/questions/1997710/pygame-error-display-surface-quit-why
            running = True
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    sys.exit()
    
            if running:
                # canvas 내용을 PyGame window로 옮기기
                self.window.blit(
                    source = canvas,
                    dest   = canvas.get_rect()
                )
                pygame.event.pump()
                pygame.display.update()

                # 설정된 frame rate에 맞게 시간 이동
                self.clock.tick(self.metadata['render_fps'])

        elif self.render_mode == "rgb_array":
            # 각 픽셀의 색깔 배열
            # - shape = (가로 격자 개수, 세로 격자 개수, 3)
            pixel_colors = np.array(pygame.surfarray.pixels3d(canvas)) 

            # 픽셀 색깔 배열을 transpose해서 return
            # - shape = (세로 격자 개수, 가로 격자 개수, 3)
            return np.transpose(pixel_colors, axes = (1, 0, 2))
        
    #------------------------------------------------------  
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
