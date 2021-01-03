import numpy as np
from collections import OrderedDict,namedtuple
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distrib


#intentions are an hour and a floor meaning that at that time people need to be at this floor
class Person():
	def __init__(self,arrival,departure):
		self.intentions=OrderedDict([(arrival,0),(departure,0)])
		self.position=0
		self.arrival = arrival
		self.departure = departure
		self.here=False
	#function to add an extra intention
	def add_new_intention(self,time,floor):
		self.intentions=OrderedDict(sorted([*self.intentions.items(),(time,floor)],key=lambda x: x[0]))

	#function to check the will of the person at a given time ie where that person wants to go
	def will(self,time):

		current_intention=0
		for intention_time in self.intentions:
			if intention_time>time:
				return current_intention
			else:
				current_intention = self.intentions[intention_time]
		return current_intention

class Distribution():
	"""
	Wrapper of a distribution, virtual class
	"""
	def sample(self):
		"""
		virtual function
		"""
		raise NotImplementedError

## Population class to sample different individuals every day so as to enforce generalisation

class PtDistribution(Distribution):
	def __init__(self,pytorch_distrib):
		self.m = pytorch_distrib

	def sample(self):
		return self.m.sample().item()


def sample_if_possible(item):
	if isinstance(item,Distribution):
		return item.sample()
	else:
		return item



class Population():
	"""
	We only modelise in laws 
	"""
	def __init__(self,number, arrival_law,departure_law,floor_law):
		self.number = number
		self.arrival_law = arrival_law
		self.departure_law = departure_law
		self.floor_law = floor_law

	def sample_population(self,intentions=None):
		pop = []
		for k in range(self.number):
			arrival = int(self.arrival_law.sample())
			departure = int(self.departure_law.sample())
			floor_hour = arrival + 20
			floor = int(self.floor_law.sample())
			individual = Person(arrival,departure)
			individual.add_new_intention(floor_hour,floor)
			
			if intentions is not None:
				intends = intentions[k]
				for time,floor in intends:
					time = sample_if_possible(time)
					floor = sample_if_possible(floor)
					individual.add_new_intention(time,floor)
			pop.append(individual)
		return pop





class Node():
	"""
	General class to handle containing persons and checking the will of the persons inside it
	"""
	def __init__(self,ID):
		self.id = ID
		#the content will be the persons in the container
		self.content=[]

	def check_will(self,time):
		return [person.will(time) for person in self.content]

class Tower(Node):
	"""
	Main environement that will we probed by an agent
	It needs: a list of Persons , a number of floors and a number of elevators
	"""
	def __init__(self,persons,nb_floors,nb_elevators):
		#calling constructor of super class
		super(Tower,self).__init__(0)
		#the content of the tower is the persons given to the constructor
		self.content = persons
		#setting ther position to floor 0
		for person in self.content:
			person.position = 0
			#and by default they are absent
			person.here = False

		#Initialize the floors as nodes	
		self.nb_floors = nb_floors
		self.floors = [Node(i) for i in range(nb_floors)]

		#initalize elevators as nodes
		self.nb_elevators = nb_elevators
		self.elevators = [Node(i) for i in range(nb_elevators)]
		#the position of the elevators at the begining is floor 0
		self.position_elevators = [0]*nb_elevators
		#begin time 7 AM count in decimal units
		self.time=700
		#end of the day
		self.stop_hour =2000

	def reset(self):
		"""
		Function to reboot the system (same as constructeer basically )
		"""
		for person in self.content:
			person.position = 0
			person.here=False
		for elevator in self.elevators:
			elevator.content = []
		for floor in self.floors:
			floor.content = []
		self.time = 700
		self.position_elevators = [0]*nb_elevators


	def check_intentions(self,node):
		"""
		Useful function that tell for the given node the people in it that are at the wrong floor (ie their position is different than their will)
		It allows to count the unsatisfied people and the ones that are calling the elevators
		"""
		intentions=[]
		for index,person in enumerate(node.content):
			will = person.will(self.time)
			#if the will of the person is different than his position (if the person is in the tower)
			if person.position != will and person.here:
				#we append it to the intention list that lists unsatisfied  people of the floor
				intentions.append((index,person.position,will))
		return intentions

	def update_arrival_departure(self):
		"""
		Function to add people to floor 0 when their arrival hour is due
		It removes them if the time is over their departure
		"""
		for person in self.content:
			#if the person is not here and its arrival time is passed
			if person.position == 0 and not person.here and person.arrival<self.time and person.departure>self.time:
				# We add it to floor 0
				self.floors[0].content.append(person)
				person.here = True
			#if the person wants to leave and is at floor 0
			if person.position == 0 and person.here and person.departure<self.time:
				#we pop it from the content of floor 0
				self.floors[0].content.pop(self.floors[0].content.index(person))
				person.here = False

	#the state will be composed of:filling of the floors
	#							   if people waiting
	#							   floor of the elevator
	#							   filling of the elevator
	#                              destinations of the elevator
	#							   destination of the persons at the floor
	# This will be given to the RL algorithm
	
	@property
	def state(self):
		#filling of the floors 
		floor_filling = np.array([len(floor.content) for floor in self.floors])/len(self.content)
		elevator_position = np.zeros((self.nb_elevators,self.nb_floors))
		#elevators destination
		elevators_destination = np.zeros((self.nb_elevators,self.nb_floors))
		for index_el,el in enumerate(self.elevators):
			elevator_position[index_el,self.position_elevators[index_el]]+=1
			intentions=self.check_intentions(el)
			for _,_,will in intentions:
				elevators_destination[index_el,will]+=1
		#filling not taken into account yet

		#destination of the persons
		floors_calls = np.zeros((self.nb_floors,self.nb_floors))
		for index_floor,floor in enumerate(self.floors):
			intentions=self.check_intentions(floor)
			for _,_,will in intentions:
				floors_calls[index_floor,will]+=1
		return floor_filling,elevator_position,elevators_destination,floors_calls,np.array([self.time])/self.stop_hour




	

	def step(self,*args):
		"""
		Main function of the elevator: args lists all the actions of the elevatorS (with an S)
		5 actions are possible for the elevators: go up, go down, open for people going up, open for people going down, open for everyone
		0: Going up
		1: Going down
		2: Picking the persons going up and dropping
		3: Picking the persons going down and dropping
		4: Picking and dropping 
		"""
		#2 minutes passed (2/60 minutes actually)
		self.time +=2
		#making people arrive or laeve the tower
		self.update_arrival_departure()

		#If the number of orders is not the same than the number of elevators we stop the program
		if len(args) != len(self.elevators):
			raise Exception

		#Dealing with all the actions
		for order_nb,arg in enumerate(args):
			#elevator is going up
			if arg == 0:
				self.position_elevators[order_nb] = min(self.nb_floors-1,self.position_elevators[order_nb]+1)
			#going down
			if arg == 1:
				self.position_elevators[order_nb] = max(0,self.position_elevators[order_nb]-1)

			#If the door opens the person that want to leave at this floor are leaving the elevator
			if arg == 2 or arg ==4 or arg==3:
				#taking different instances as they will be used a lot in this scope
				floor_index =self.position_elevators[order_nb]
				floor = self.floors[floor_index]
				elevator = self.elevators[order_nb]
				#checking the intentions of the people in the elevator
				intentions = self.check_intentions(elevator)
				#reversing because we are using pop on a list that changes the indexing of the intention list
				#except if we pop the bigger indexes first (hence the reverse to start with biggests indexes)
				intentions.reverse()
				for index,position,will in intentions:
					#if the elevator is at the right floor
					if will == floor_index:
						#the person leaves the elevator
						person =elevator.content.pop(index)
						# his/her (no sexism here) new position is the floor
						person.position= floor_index
						#he/she belongs to the floor now
						floor.content.append(person)
				# Now the actions to pick up the persons concerned
				#just like before the intentions are reversed
				intentions = self.check_intentions(floor)
				intentions.reverse()
				for index,position,will in intentions:
					#If we pick the persons for for going up
					if arg == 2 or arg ==4:
						#only the ones going up are entering
						if position < will:
							#pop and append
							elevator.content.append(floor.content.pop(index))

					if arg == 3 or arg ==4:
						#only the ones going down are entering
						if position > will:
							#pop and append
							elevator.content.append(floor.content.pop(index))
		#the reward is minus the number of people waiting
		reward = - len(self.check_intentions(self))
		return self.state,reward,(self.time>=self.stop_hour)


class StupidElevator():
	"""
	Elevator that just goes every floor opening the doors (hardcoded for 3 floors)
	"""
	def __init__(self):
		self.cursor = 0
		self.actions = [4,0,4,0,4,1,4,1]
	def select_action(self,state):
		action = self.actions[self.cursor]
		self.cursor =(self.cursor+1)%8
		return action

class IntelligentElevator():
	def __init__(self):
		self.will=-1

	def select_action(self,state_uncoded):
		floor_filling,elevator_position,elevators_destination,floors_calls,time = state_uncoded
		loaded = bool(np.max(elevators_destination))

		if self.will == -1 and not loaded:
			
			floor_call_per_floor = np.sum(floors_calls,axis=1)
			if np.max(floor_call_per_floor)==0:
				self.will=-1
				return -1
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

#Util function to cast state into torch tensor
def translate_state(state):
	return torch.tensor(np.concatenate([array.reshape(-1) for array in state],axis=0)).float()

#Transition for the RL algorithm for the replay memory
Transition = namedtuple('Transition',
						('state', 'action', 'next_state', 'reward'))


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

	def __len__(self):
		return len(self.memory)



#RL algorithm using Sarsa for weight updates
class AgentLinearSarsa(nn.Module):
	def __init__(self,state_dim,action_dim,epsilon,epsilon_decay,gamma):
		super(AgentLinearSarsa,self).__init__()
		#2 layer neural network
		self.lin = nn.Sequential(nn.Linear(state_dim,50),nn.ReLU(),nn.Linear(50,action_dim))

		# time for the decay of epsilon
		self.time = 0
		self._epsilon=epsilon
		self.epsilon_decay = epsilon_decay
		self.action_dim = action_dim
		self.gamma = gamma

	#Epsilon decayed
	@property
	def epsilon(self):
		return max(self._epsilon*self.epsilon_decay**self.time,0.05)

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
										  batch.next_state)), dtype=torch.bool)
		non_final_next_states = torch.cat([s for s in batch.next_state
													if s is not None])
		state_batch = torch.cat(batch.state)
		action_batch = torch.cat(batch.action)
		reward_batch = torch.cat(batch.reward)
		

		# Compute V(s_{t+1}) for all next states.
		next_state_values = torch.zeros(batch_size)
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

def initialize_population(number,arrival_law,departure_law,floor_law,intentions=None):
	pop = Population(number, arrival_law,departure_law,floor_law)
	persons = pop.sample_population(intentions)
	return  persons


def test_linear_sarsa():
	arrival_law = PtDistribution(distrib.uniform.Uniform(700,901))
	departure_law= PtDistribution(distrib.uniform.Uniform(1800,1870))
	floor_law = PtDistribution(distrib.categorical.Categorical(probs=torch.Tensor([0.,0.3,0.7])))
	number =7

	persons = initialize_population(number,arrival_law,departure_law,floor_law)


	# Intialize Tower


	nb_floors=3
	nb_elevators=1
	# persons=[Person(arrival,departure) for arrival,departure in zip(range(700,1050,50),range(1800,1870,10))]

	# for index,person in enumerate(persons):
	# 	person.add_new_intention(person.arrival+index*10,index%nb_floors)
	
	tower = Tower(persons,nb_floors,nb_elevators)


	#parameters agent
	state_dim = translate_state(tower.state).shape[0]
	action_dim = 5
	epsilon = 0.9
	epsilon_decay=0.98
	gamma=0.999
	learning_rate=0.01

	#initializing memory
	capacity = 100000
	batch_size = 128
	memory = ReplayMemory(capacity)

	#init agent
	agent = AgentLinearSarsa(state_dim,action_dim,epsilon,epsilon_decay,gamma)
	# agent_elev = IntelligentElevator()
	# a target net to avoid over-fitting the current policy
	target_net = AgentLinearSarsa(state_dim,action_dim,epsilon,epsilon_decay,gamma)
	target_net.load_state_dict(agent.state_dict())
	#using RMSprop but Adam works perfectly
	optimizer = optim.RMSprop(agent.parameters(), lr=learning_rate)


	#run trials
	number_of_trials =50000
	for trial in range(number_of_trials):

		#get state
		state = tower.state
		#cast it in torch
		state = translate_state(state)
		#metric: length waited
		summed_rwd=0
		#bool for final state
		done = False
		#run episode
		while not done:
			#select action
			action = agent.select_action(state)
			#process action
			new_state,reward,done = tower.step(action)
			#cast new state
			new_state = translate_state(new_state)
			#adding number of  unsatisfied people to the metric
			summed_rwd += reward
			#adding transition to the memory
			memory.push(state.float().view(1,-1), torch.tensor(action).view(1,1).long(), new_state.float().view(1,-1), torch.tensor([reward]).float())
			state = new_state
			#updating agent
			if trial >=1:
				agent.update(memory,batch_size,gamma,optimizer,target_net)

		print("Trial Nb: ",trial)
		print("Number of minutes waited: ",-summed_rwd)
		print("Percentage of random actions: ", agent.epsilon)
		# tower.reset()
		persons = initialize_population(number,arrival_law,departure_law,floor_law)
		tower = Tower(persons,nb_floors,nb_elevators)

		# assigning network to the target net (for value iteration)
		if trial%2 == 0:
			target_net.load_state_dict(agent.state_dict())
		agent.decay()


def test_intelligent():
	arrival_law = PtDistribution(distrib.uniform.Uniform(700,901))
	departure_law= PtDistribution(distrib.uniform.Uniform(1800,1870))
	floor_law = PtDistribution(distrib.categorical.Categorical(probs=torch.Tensor([0.,0.3,0.7])))
	number =7

	persons = initialize_population(number,arrival_law,departure_law,floor_law)


	# Intialize Tower


	nb_floors=3
	nb_elevators=1
	# persons=[Person(arrival,departure) for arrival,departure in zip(range(700,1050,50),range(1800,1870,10))]

	# for index,person in enumerate(persons):
	# 	person.add_new_intention(person.arrival+index*10,index%nb_floors)
	
	tower = Tower(persons,nb_floors,nb_elevators)


	agent = IntelligentElevator()


	#run trials
	results = []
	number_of_trials =50000
	for trial in range(number_of_trials):

		#get state
		state = tower.state
		#metric: length waited
		summed_rwd=0
		#bool for final state
		done = False
		#run episode
		while not done:
			#select action
			action = agent.select_action(state)
			#process action
			new_state,reward,done = tower.step(action)
			#adding number of  unsatisfied people to the metric
			summed_rwd += reward

			state = new_state
		results.append(-summed_rwd)


		# tower.reset()
		persons = initialize_population(number,arrival_law,departure_law,floor_law)
		tower = Tower(persons,nb_floors,nb_elevators)
	print("Average performance:",np.mean(results))




def test_stupid():
	arrival_law = PtDistribution(distrib.uniform.Uniform(700,901))
	departure_law= PtDistribution(distrib.uniform.Uniform(1800,1870))
	floor_law = PtDistribution(distrib.categorical.Categorical(probs=torch.Tensor([0.,0.3,0.7])))
	number =7

	persons = initialize_population(number,arrival_law,departure_law,floor_law)


	# Intialize Tower


	nb_floors=3
	nb_elevators=1
	# persons=[Person(arrival,departure) for arrival,departure in zip(range(700,1050,50),range(1800,1870,10))]

	# for index,person in enumerate(persons):
	# 	person.add_new_intention(person.arrival+index*10,index%nb_floors)
	
	tower = Tower(persons,nb_floors,nb_elevators)


	agent = StupidElevator()


	#run trials
	results = []
	number_of_trials =50000
	for trial in range(number_of_trials):

		#get state
		state = tower.state
		#metric: length waited
		summed_rwd=0
		#bool for final state
		done = False
		#run episode
		while not done:
			#select action
			action = agent.select_action(state)
			#process action
			new_state,reward,done = tower.step(action)
			#adding number of  unsatisfied people to the metric
			summed_rwd += reward

			state = new_state
		results.append(-summed_rwd)


		# tower.reset()
		persons = initialize_population(number,arrival_law,departure_law,floor_law)
		tower = Tower(persons,nb_floors,nb_elevators)
	print("Average performance:",np.mean(results))

if __name__ == "__main__":
	test_stupid()



