import time
import random
import pygame
import numpy as np
from collections import deque

class SnakeEnv:
    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

    def __init__(self, grid_size=(10, 10), window_size=(300, 300), gui=False):
        self.grid_size = grid_size
        self.window_size = window_size
        self.gui = gui
        self.reset()
        if self.gui:
            self._initialize_gui()

    def _initialize_gui(self):
        pygame.init()
        self.window = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("Snake Game")

    def reset(self):
        n, m = self.grid_size
        self.snake = deque([(n // 2, m // 2)])
        self.food = self._generate_food()
        self.done = False
        self.steps_without_food = 0
        return self._get_observation()

    def step(self, action):

        if self.done:
            return self._get_observation(), 0, True, {}

        new_head = self._calculate_new_head(action)
        if not self._is_valid_position(new_head):
            self.done = True
            return self._get_observation(), -10, True, {}

        self.snake.append(new_head)
        threshold = self.grid_size[0] * self.grid_size[1]

        if new_head == self.food:
            self.food = self._generate_food()
            self.steps_without_food = 0
            reward = 1
        else:
            self.snake.popleft()
            self.steps_without_food += 1
            reward = -1 / threshold / 2

        done = self.steps_without_food > threshold
        mask = self._calculate_mask()
        if not any(mask):
            done = True
            reward = -10
            mask = np.ones(4)
        
        return self._get_observation(), reward, done, {'mask': mask}

    def render(self):
        if not self.gui or self.done:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        cell_size = self.window_size[0] // self.grid_size[0]
        self.window.fill((255, 255, 255))

        # Draw food
        pygame.draw.rect(self.window, (255, 0, 0),
                         (self.food[1] * cell_size, self.food[0] * cell_size, cell_size, cell_size))

        # Draw snake
        for segment in self.snake:
            pygame.draw.rect(self.window, (0, 0, 0),
                             (segment[1] * cell_size, segment[0] * cell_size, cell_size, cell_size))

        pygame.display.flip()

    def _calculate_mask(self):
        mask = [self._is_valid_position(self._calculate_new_head(a)) for a in range(4)]
        return np.array(mask)

    def _is_valid_position(self, pos):
        return 0 <= pos[0] < self.grid_size[0] and 0 <= pos[1] < self.grid_size[1] and pos not in self.snake

    def _generate_food(self):
        cells = {(x, y) for x in range(self.grid_size[0]) for y in range(self.grid_size[1])}
        cells -= set(self.snake)
        return random.choice(tuple(cells)) if cells else None

    def _calculate_new_head(self, action):
        head, (dx, dy) = self.snake[-1], self.ACTIONS[action]
        return (head[0] + dx, head[1] + dy)

    def _get_observation(self):
        grids = np.zeros((2, *self.grid_size), dtype=np.float32)

        # snake
        snake_length = len(self.snake)
        grids[0] += -1
        for i, segment in enumerate(self.snake, start=1):
            grids[0][segment] = i / snake_length

        # food
        if self.food:
            grids[1][self.food] = 1

        return grids

    def close(self):
        if self.gui:
            pygame.quit()

if __name__ == '__main__':
    gui = True
    env = SnakeEnv(gui=gui)
    mask = np.ones(4)
    s = env.reset()

    for _ in range(1000):
        valid_actions = [a for a in range(4) if mask[a]]
        action = random.choice(valid_actions)
        s, reward, done, info = env.step(action)
        mask = info.get('mask', np.ones(4))

        env.render()
        time.sleep(0.01)
        print(np.round(s[0], 1))

        if done:
            break

    env.close()
