# CS481 Final Project Snake AI
# Created by Vince Bjazevic, Mathew Nguyen, Luis Valle-Arellanes
# Description: Implements a neural network-based artificial intelligence for playing the Snake game 
#      using the Pygame library. The project involves defining a feedforward neural network, handling 
#      model parameters and evaluation, and recall for reinforcement learning. The AI learns to play 
#      the game by interacting with its environment and adjusting its behavior based on feedback.

import torch
import random
import numpy as np
from collections import deque, namedtuple
import pygame
import random
from IPython import display
import os

class ForwardFeedingNN(torch.nn.Module):
	def __init__(self, n_input, n_hidden, n_output):
		super().__init__()
		# Define the neural network layers		
		self.layer1 = torch.nn.Linear(n_input, n_hidden)
		self.layer3 = torch.nn.Linear(n_hidden, n_hidden)        
		self.layer2 = torch.nn.Linear(n_hidden, n_output)

	def forward(self, x):
		# Define the forward pass of the neural network
		x = torch.nn.functional.relu(self.layer1(x))
		x = self.layer2(x)
		return x

class Model_Parameters: # Class to handle model parameters and evaluation
	def __init__(self, model):
		self.model = model
		self.MSE_Loss = torch.nn.MSELoss()
		self.optimizer = torch.optim.Adam(model.parameters(), 0.001) # Learning rate


	def evaluate(self, current_state, move, renforcement, next_state, game_over):
		# Convert data to PyTorch tensors
		current_state = torch.tensor(current_state, dtype=torch.float)
		next_state = torch.tensor(next_state, dtype=torch.float)
		move = torch.tensor(move, dtype=torch.long)
		renforcement = torch.tensor(renforcement, dtype=torch.float)

		if len(current_state.shape) == 1:
            # Unsqueezing to add a batch dimension if it's a single sample
			current_state = torch.unsqueeze(current_state, 0)
			next_state = torch.unsqueeze(next_state, 0)
			move = torch.unsqueeze(move, 0)
			renforcement = torch.unsqueeze(renforcement, 0)
			game_over = (game_over, )

        # Forward pass through the model
		predicted = self.model(current_state)

		actual = predicted.clone()
        # Update the Q-values based on the renforcements and game_over		
		for i in range(len(game_over)):
			eval = renforcement[i]
			if not game_over[i]:
				eval = renforcement[i] + 0.8 * torch.max(self.model(next_state[i]))

			actual[i][torch.argmax(move[i]).item()] = eval

        # Calculate and backpropagate the loss
		self.optimizer.zero_grad()
		loss = self.MSE_Loss(actual, predicted)
		loss.backward()

		self.optimizer.step()



class AI_Snake: # Main class for the Snake AI

	def __init__(self):

        # Initialize game parameters, snake, model, game board and neural network
		self.width = 640
		self.height = 480
		self.total_games = 0
		self.score = 0
		self.route = 1

		self.memory = deque(maxlen=100_000) # popleft()
		self.model = ForwardFeedingNN(11, 256, 3)
		self.param = Model_Parameters(self.model)

        # Initialize display and game elements
		self.display = pygame.display.set_mode((self.width, self.height))
		pygame.display.set_caption('Snake AI (Please give the AI at least 1 minute to train or 100 games...)')
		self.clock = pygame.time.Clock()
		
		# Place Inital snake, and orientation
		self.front = Grid_Square(self.width/2, self.height/2)
		self.snake = [self.front,
					  Grid_Square(self.front.x-20, self.front.y),
					  Grid_Square(self.front.x-(2*20), self.front.y)]

		# Place goal onto board
		self.goal = None
		self.goal = Grid_Square(random.randint(0, (self.width-20)//20 )*20, random.randint(0, (self.width-20)//20 )*20)
		self.frame_iteration = 0


	def get_current_gamestate(self):
        # Obtain the current game state as input for the neural network
		front = self.snake[0]
		current_state = []
		orientation_l, orientation_r, orientation_u, orientation_d = False, False, False, False

		if self.route == 1:
			orientation_r = True
		elif self.route == 2:
			orientation_l = True
		elif self.route == 3:
			orientation_u = True
		elif self.route == 4:
			orientation_d = True

		# Obstacle right of snake front
		if orientation_r and self.is_edge(Grid_Square(front.x + 20, front.y)):
			current_state.append(True)
		elif orientation_l and self.is_edge(Grid_Square(front.x - 20, front.y)):
			current_state.append(True)
		elif orientation_u and self.is_edge(Grid_Square(front.x, front.y - 20)):
			current_state.append(True)
		elif orientation_d and self.is_edge(Grid_Square(front.x, front.y + 20)):
			current_state.append(True)
		else:
			current_state.append(False)
		# Obstacle forward of snake front
		if orientation_u and self.is_edge(Grid_Square(front.x + 20, front.y)):
			current_state.append(True)
		elif orientation_d and self.is_edge(Grid_Square(front.x - 20, front.y)):
			current_state.append(True)
		elif orientation_l and self.is_edge(Grid_Square(front.x, front.y - 20)):
			current_state.append(True)
		elif orientation_r and self.is_edge(Grid_Square(front.x, front.y + 20)):
			current_state.append(True)
		else:
			current_state.append(False)
		# Obstacle left of snake front
		if orientation_d and self.is_edge(Grid_Square(front.x + 20, front.y)):
			current_state.append(True)
		elif orientation_u and self.is_edge(Grid_Square(front.x - 20, front.y)):
			current_state.append(True)
		elif orientation_r and self.is_edge(Grid_Square(front.x, front.y - 20)):
			current_state.append(True)
		elif orientation_l and self.is_edge(Grid_Square(front.x, front.y + 20)):
			current_state.append(True)
		else:
			current_state.append(False)
		
		# Direction flags
		current_state.append(orientation_l)
		current_state.append(orientation_r)
		current_state.append(orientation_u)        
		current_state.append(orientation_d)

		# Relative position to goal cell
		current_state.extend([
			self.goal.x < self.front.x,self.goal.x > self.front.x,
			self.goal.y < self.front.y,self.goal.y > self.front.y])
				
		return np.array(current_state, dtype=int)

	def recall_past_data(self, current_state, move, renforcement, next_state, game_over):
		# Store past data in memory for training
		self.memory.append((current_state, move, renforcement, next_state, game_over)) 

	def train_on_state(self, current_state, move, renforcement, next_state, game_over):
        # Train the model on a single experience		
		self.param.evaluate(current_state, move, renforcement, next_state, game_over)

	def get_new_current_state(self, current_state):
        # Decide on the next move based on the current state and the trained model
		final_move = [0,0,0]
		if random.randint(0, 200) < 80 - self.total_games: # choosing between random and predictediction
			move = random.randint(0, 2)
			final_move[move] = 1
		else:
			current_state0 = torch.tensor(current_state, dtype=torch.float)
			predictediction = self.model(current_state0)
			move = torch.argmax(predictediction).item()
			final_move[move] = 1

		return final_move


	def next_frame(self, move):
        # Update the game state for the next frame
		self.frame_iteration += 1
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				quit()

		orientation = [1, 4, 2, 3] # 1: Right, 2: Left, 3: Up, 4: Down
		i = orientation.index(self.route)

		if np.array_equal(move, [1, 0, 0]):
			snake_turn = orientation[i] # no change in orientation
		elif np.array_equal(move, [0, 1, 0]):
			snake_turn = orientation[(i + 1) % 4] # right turn r -> d -> l -> u
		else: # [0, 0, 1]
			snake_turn = orientation[(i - 1) % 4] # left turn r -> u -> l -> d

		self.route = snake_turn

		if self.route == 1:
			self.front = Grid_Square(self.front.x + 20, self.front.y)
		elif self.route == 2:
			self.front = Grid_Square(self.front.x - 20, self.front.y)
		elif self.route == 4:
			self.front = Grid_Square(self.front.x, self.front.y + 20)
		elif self.route == 3:
			self.front = Grid_Square(self.front.x, self.front.y - 20)

		# Populate new cell with snake body
		self.snake.insert(0, self.front)
		
		renforcement = 0
		game_over = False
		if self.is_edge() or self.frame_iteration > 100*len(self.snake):
			game_over = True
			# For loss renforce with -5 points
			renforcement = -5
			return renforcement, game_over, self.score

		if self.front == self.goal:
			self.score += 1
			# For goal renforce with 5 points
			renforcement = 5
			self.goal = Grid_Square(random.randint(0, (self.width-20)//20 )*20, random.randint(0, (self.height-20 )//20)*20)
		else:
			# Remove last snake body cell
			self.snake.pop()
		
		self.display.fill((72, 65, 78))

		flip_x = False
		for x in range(0, self.width, 20):
			flip_y = False
			for y in range(0, self.height, 20):
				if flip_y != flip_x:
					pygame.draw.rect(self.display, (64, 59, 70), pygame.Rect(x, y, 20, 20))
				flip_y = not flip_y
			flip_x = not flip_x

		
		# Draw display
		for pt in self.snake:
			pygame.draw.rect(self.display, (63, 123, 242), pygame.Rect(pt.x, pt.y, 20, 20))

		pygame.draw.rect(self.display, (255,255,255), pygame.Rect(self.front.x + 4, self.front.y + 4, 12, 12))
		pygame.draw.rect(self.display, (244, 74, 39), pygame.Rect(self.goal.x + 4, self.goal.y + 4, 12, 12))
		pygame.draw.rect(self.display, (16, 195, 66), pygame.Rect(self.goal.x + 10, self.goal.y + 2, 6, 4))


		text = font.render("Score: " + str(self.score), True, (245, 245, 245))
		self.display.blit(text, [0, 0])
		pygame.display.flip()

		# Set speed of games
		if self.total_games >= 100:
			self.clock.tick(20)
		else:
			self.clock.tick(1000)

		return renforcement, game_over, self.score


	def is_edge(self, pt=None):
        # Check if a given point is an edge in the game		
		if pt is None:
			pt = self.front

		if pt.x > self.width - 20:
			return True
		if pt.x < 0:
			return True
		if pt.y > self.height - 20:
			return True
		if pt.y < 0:
			return True 

		if pt in self.snake[1:]:
			return True

		return False
	
if __name__ == '__main__':
	# Main game loop, snake and states
	pygame.init()
	Grid_Square = namedtuple('Grid_Square', 'x, y')
	font = pygame.font.Font('Roboto-BlackItalic.ttf', 25)
	record = 0
	snake = AI_Snake()
	while True:
        # Game loop iteration
		previous_current_state = snake.get_current_gamestate()
		final_move = snake.get_new_current_state(previous_current_state)
		renforcement, game_over, score = snake.next_frame(final_move)
		current_state_new = snake.get_current_gamestate()

        # Update the neural network with the new experiences
		snake.train_on_state(previous_current_state, final_move, renforcement, current_state_new, game_over)
		snake.recall_past_data(previous_current_state, final_move, renforcement, current_state_new, game_over)

		if game_over:

			# Reset
			snake.route = 1
			snake.front = Grid_Square(snake.width/2, snake.height/2)
			snake.snake = [snake.front,
					  Grid_Square(snake.front.x-20, snake.front.y),
					  Grid_Square(snake.front.x-(2*20), snake.front.y)]
			snake.score = 0
			snake.goal = None
			snake.goal = Grid_Square(random.randint(0, (snake.width-20)//20 )*20, random.randint(0, (snake.height-20 )//20)*20)
			snake.frame_iteration = 0
			snake.total_games += 1
			# Train the model on retained information

			# Update score
			if score > record:
				record = score
			print('Game:', snake.total_games, '\t' 'Score:', score, '\t','Record:', record)