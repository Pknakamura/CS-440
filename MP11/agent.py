import numpy as np
import utils
from operator import or_

class Agent:
    def __init__(self, actions, Ne=40, C=40, gamma=0.7, display_width=18, display_height=10):
        # HINT: You should be utilizing all of these
        self.actions = actions
        print(self.actions)
        self.Ne = Ne  # used in exploration function
        print(self.Ne)
        self.C = C
        self.gamma = gamma
        self.display_width = display_width
        self.display_height = display_height
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()

        self.checked_actions = [0,0,0,0]
        
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self, model_path):
        utils.save(model_path, self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

        print(self.checked_actions)

    # Load the trained model for evaluation
    def load_model(self, model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None
        self.a = None
    
    def update_n(self, state, action):
        # TODO - MP11: Update the N-table. 
        self.N[state][action] += 1

    def update_q(self, s, a, r, s_prime):
        # TODO - MP11: Update the Q-table. 
        lr = self.C / (self.C + self.N[s][a])
        self.Q[s][a] += lr * (r + self.gamma * max(self.Q[s_prime]) - self.Q[s][a])     

    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        '''

        s_prime = self.generate_state(environment)

        if self._train and self.s != None:
            reward = -1 if dead else 1 if points > self.points else -0.1

            self.update_n(self.s, self.a)
            self.update_q(self.s, self.a, reward, s_prime)
        
        best_action = self.actions[-1]
        f_max = 1 if self.N[s_prime][best_action] < self.Ne and self._train else self.Q[s_prime][best_action]
        for a in self.actions[-2::-1]:
            f_a = 1 if self.N[s_prime][a] < self.Ne and self._train else self.Q[s_prime][a]
            if f_a > f_max:
                best_action = a
                f_max = f_a
        
        self.checked_actions[best_action] += 1

        if self._train:
            if dead:
                self.reset()
            else:
                self.s = s_prime
                self.a = best_action
                self.points = points

        return best_action
        

    def generate_state(self, environment):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        '''
        # TODO - MP11: Implement this helper function that generates a state given an environment 
        snake_head_x = environment[0]
        snake_head_y = environment[1]
        snake_body = environment[2]
        food_x = environment[3]
        food_y = environment[4]
        rock_x = environment[5]
        rock_y = environment[6]

        wall_up    = snake_head_y == 1 or (rock_x, rock_y) == (snake_head_x, snake_head_y - 1) or (rock_x, rock_y) == (snake_head_x - 1, snake_head_y - 1) 
        wall_down  = snake_head_y == self.display_height - 2  or (rock_x, rock_y) == (snake_head_x, snake_head_y + 1) or (rock_x, rock_y) == (snake_head_x - 1, snake_head_y + 1)
        wall_left  = snake_head_x == 1 or (rock_x, rock_y) == (snake_head_x - 2, snake_head_y)
        wall_right = snake_head_x == self.display_width - 2 or (rock_x, rock_y) == (snake_head_x + 1, snake_head_y)

        food_dir_x = 1 if snake_head_x > food_x else 2 if snake_head_x < food_x else 0
        food_dir_y = 1 if snake_head_y > food_y else 2 if snake_head_y < food_y else 0
        adjoining_wall_x = 0 if not wall_left and not wall_right else 2 if not wall_left and wall_right else 1
        adjoining_wall_y = 0 if not wall_up and not wall_down else 2 if not wall_up and wall_down else 1
        adjoining_body_top = int((snake_head_x, snake_head_y - 1) in snake_body)
        adjoining_body_bottom = int((snake_head_x, snake_head_y + 1) in snake_body)
        adjoining_body_left = int((snake_head_x - 1, snake_head_y) in snake_body)
        adjoining_body_right = int((snake_head_x + 1, snake_head_y) in snake_body)

        return (food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, 
                adjoining_body_bottom, adjoining_body_left, adjoining_body_right)
