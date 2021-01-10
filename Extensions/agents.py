
import numpy as np
from collections import OrderedDict,namedtuple
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distrib


class StupidElevator():
	"""
	Elevator that just goes every floor opening the doors
	"""
	def __init__(self,number_floors):
		self.cursor = 0
		self.actions = [4,0]*(number_floors-1)+ [4,1]*(number_floors-1)
	def select_action(self,state):
		action = self.actions[self.cursor]
		self.cursor =(self.cursor+1)%len(self.actions)
		return action

class IntelligentElevator():
	#elevator making the actions in a greedy manner, it goes to the highest demand floor, picks up everyone and deliver them to their wanted floor

	def __init__(self):
		self.will=-1

	def select_action(self,state_uncoded):
		floor_filling,elevator_position,elevators_destination,floors_calls,time = state_uncoded
		loaded = bool(np.max(elevators_destination))

		if self.will == -1 and not loaded:
			
			floor_call_per_floor = np.sum(floors_calls,axis=1)
			if np.max(floor_call_per_floor)==0:
				self.will=-1
				return 4
			else :
				self.will = np.argmax(floor_call_per_floor)
		if self.will ==-1 and loaded:
			self.will = np.argmax(elevators_destination)

		position = np.argmax(elevator_position[0])
		if self.will == position:
			self.will=-1
			return 4
		if self.will < position:
			return 1
		if self.will > position:
			return 0



class RandomElevator():
	"""
	Elevator acting randomly
	"""
	def __init__(self):
		pass
	def select_action(self,state):
		return random.randint(0,4)



#Transition for the RL algorithm for the replay memory
Transition = namedtuple('Transition',
						('state', 'action', 'reward'))


class ReplayMemory(object):
	"""
	Memory class of the RL algorithm to enable experience replay
	"""
	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0

	def push(self, *args):
		"""Saves a transition."""
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = Transition(*args)
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def reset(self):
		self.memory = []
		self.position = 0

	def __len__(self):
		return len(self.memory)



#RL algorithm using Sarsa for weight updates
class AgentLinearSarsa(nn.Module):
	def __init__(self,state_dim,action_dim,epsilon,epsilon_decay,gamma):
		super(AgentLinearSarsa,self).__init__()
		#2 layer neural network
		self.lin = nn.Sequential(nn.Linear(state_dim,400),nn.ReLU(),nn.Linear(400,400),nn.ReLU(),nn.Linear(400,action_dim))

		# time for the decay of epsilon
		self.time = 0
		self._epsilon=epsilon
		self.epsilon_decay = epsilon_decay
		self.action_dim = action_dim
		self.gamma = gamma

	#Epsilon decayed
	@property
	def epsilon(self):
		return max(self._epsilon*self.epsilon_decay**self.time,0.02)

	@epsilon.setter
	def epsilon(self,value):
		self._epsilon = value
		self.time=0

	#function to decay epsilon
	def decay(self):
		self.time +=1

	#forward pass of the neural network
	def forward(self,states):
		return self.lin(states)

	#selecting action in a epsilon greedy manner
	def select_action(self,state):
		if random.random() < self.epsilon:
			return random.randint(0,self.action_dim-1)
		else:
			with torch.no_grad():
				processed = self(state)
				return torch.argmax(processed).item()

	#Update using experience replay and Sarsa
	def update(self,memory,batch_size,gamma,optimizer,target_net):
		transitions = memory.sample(batch_size)
		batch = Transition(*zip(*transitions))
		non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
										  batch.next_state)), dtype=torch.bool,device=self.lin[0].weight.device)
		non_final_next_states = torch.cat([s for s in batch.next_state
													if s is not None]).to(device=self.lin[0].weight.device)
		state_batch = torch.cat(batch.state)
		action_batch = torch.cat(batch.action).to(device=self.lin[0].weight.device)
		reward_batch = torch.cat(batch.reward).to(device=self.lin[0].weight.device)

		# Compute V(s_{t+1}) for all next states.
		next_state_values = torch.zeros(batch_size,device=self.lin[0].weight.device)
		with torch.no_grad():
			next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
			# Compute the expected Q values
			expected_state_action_values = (next_state_values * gamma) + reward_batch



		state_action_values = self(state_batch).gather(1, action_batch)

		# Compute Huber loss

		loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

		# Optimize the model
		optimizer.zero_grad()
		loss.backward()
		for param in self.parameters():
			param.grad.data.clamp_(-1, 1)
		optimizer.step()


class PolicyGradientAgent(nn.Module):
	def __init__(self,state_dim,action_dim,epsilon,epsilon_decay,gamma):
		super(PolicyGradientAgent,self).__init__()
		#2 layer neural network
		self.lin = nn.Sequential(nn.Linear(state_dim,400),nn.ReLU(),nn.Linear(400,400),nn.ReLU(),nn.Linear(400,action_dim))

		# time for the decay of epsilon
		self.time = 0
		self._epsilon=epsilon
		self.epsilon_decay = epsilon_decay
		self.action_dim = action_dim
		self.gamma = gamma

	#Epsilon decayed
	@property
	def epsilon(self):
		return max(self._epsilon*self.epsilon_decay**self.time,0.02)

	@epsilon.setter
	def epsilon(self,value):
		self._epsilon = value
		self.time=0

	#function to decay epsilon
	def decay(self):
		self.time +=1

	#forward pass of the neural network
	def forward(self,states):
		return self.lin(states)

	#selecting action in a epsilon greedy manner
	def select_action(self,state):
		if random.random() < self.epsilon:
			return random.randint(0,self.action_dim-1)
		else:
			with torch.no_grad():
				processed = self(state)
				return torch.argmax(processed).item()

	#Update using experience replay and Sarsa
	def update(self,memory,gamma,optimizer):
		transitions = memory.memory
		batch = Transition(*zip(*transitions))
		non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
										  batch.next_state)), dtype=torch.bool,device=self.lin[0].weight.device)
		non_final_next_states = torch.cat([s for s in batch.next_state
													if s is not None]).to(device=self.lin[0].weight.device)
		state_batch = torch.cat(batch.state)
		action_batch = torch.cat(batch.action).to(device=self.lin[0].weight.device).squeeze()
		reward_batch = torch.cat(batch.reward).to(device=self.lin[0].weight.device)
		discounted_reward = torch.stack([sum([gamma**i*reward_batch[k:][i] for i in range(reward_batch[k:].shape[0])])for k in range(reward_batch.shape[0])],dim=0)
		# print(discounted_reward)
		discounted_reward = (discounted_reward-torch.mean(discounted_reward))
		
		# exit()

		quality = discounted_reward*F.cross_entropy(self(state_batch), action_batch, reduction='none')




		loss = -torch.mean(quality)

		# Optimize the model
		optimizer.zero_grad()
		loss.backward()
		for param in self.parameters():
			param.grad.data.clamp_(-1, 1)
		optimizer.step()
		memory.reset()

#policy learner general class 
class ReinforceAgent(object):
	def __init__(self,temperature, temperature_decay):
		self._temperature = temperature
		self.temperature_decay = temperature_decay
		self.time = 0

	#for selecting random actions
	@property
	def temperature(self):
		return max(self._temperature*self.temperature_decay**self.time,1)
	
	@temperature.setter
	def epsilon(self,value):
		self._temperature = value
		self.time=0

	def decay(self):
		self.time +=1

	# this one is sampling action the the policy
	def select_action(self,state):
		probs = self.probs_from_state(state)
		probs = np.array([round(prob.item(),2) for prob in probs])
		probs = probs/ np.sum(probs)

		return np.random.choice(range(probs.shape[0]),p=probs)

	def probs_from_state(self,state):
		raise NotImplementedError("The process function has to be implemented")



class AgentReinforceTorch(ReinforceAgent):
	"""
	Policy agent that uses reinforce
	"""
	def __init__(self,state_dim,action_dim,temperature, temperature_decay):
		ReinforceAgent.__init__(self,temperature, temperature_decay)
		#module to evaluate the probabilities
		self.decision_module = nn.Sequential(nn.Linear(state_dim,500),nn.ReLU(),nn.Linear(500,action_dim))
		self.decision_module[0].weight.data.normal_(0,1/np.sqrt(50))
		self.decision_module[0].bias.data.normal_(0,1/np.sqrt(50))
		self.decision_module[2].weight.data.normal_(0,1/np.sqrt(50))
		self.decision_module[2].bias.data.normal_(0,1/np.sqrt(50))


	#smooth the probabilities with temparature (not needed here)
	def probs_from_state(self,state):
		with torch.no_grad():
			return F.softmax(self.decision_module(state)/self.temperature)

	def gradient_probits(self,state):
		return F.softmax(self.decision_module(state))

	#function to update the network following the reinforce algorithm
	def update_network(self,memory,optimizer):
		#taking alkl episode
		transitions = memory.memory
		#batching
		batch = Transition(*zip(*transitions))
		
		state_batch = torch.cat(batch.state)
		action_batch = torch.cat(batch.action)
		reward_batch = torch.cat(batch.reward)
		#negative log probability multiplied with the reward and then averaged
		loss= torch.mean(F.cross_entropy(self.decision_module(state_batch),action_batch,reduction='none')*reward_batch)
		#nullify gradients
		optimizer.zero_grad()
		#backward
		loss.backward()
		#optimization step
		optimizer.step()