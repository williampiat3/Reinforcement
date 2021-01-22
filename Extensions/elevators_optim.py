import numpy as np
from collections import OrderedDict,namedtuple
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distrib
from environement import Person,Tower,Node,Environement
from agents import StupidElevator,RandomElevator,IntelligentElevator,AgentLinearSarsa,ReplayMemory,PolicyGradientAgent,AgentReinforceTorch
import matplotlib.pyplot as plt



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
	We only modelise in laws for sampling different timesets at every turn
	"""
	def __init__(self,number, arrival_law,departure_law,floor_law):
		#number of individuals to modelise
		self.number = number
		#arrival law
		self.arrival_law = arrival_law
		#departure law
		self.departure_law = departure_law
		#repartition on the floors law
		self.floor_law = floor_law

	def sample_population(self,intentions=None):
		#function to sample the individuals behaviors
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





def initialize_population(number,arrival_law,departure_law,floor_law,intentions=None):
	pop = Population(number, arrival_law,departure_law,floor_law)
	persons = pop.sample_population(intentions)
	return  persons

#Util function to cast state into torch tensor
def translate_state(state,device=None,dtype=torch.float):
	new_state = torch.tensor(np.concatenate([array.reshape(-1) for array in state],axis=0))
	return new_state.to(device=device,dtype=dtype)

def test_linear_sarsa():
	device = torch.device("cuda")
	arrival_law = PtDistribution(distrib.uniform.Uniform(700,901))
	departure_law= PtDistribution(distrib.uniform.Uniform(1800,1870))
	floor_law = PtDistribution(distrib.categorical.Categorical(probs=torch.Tensor([0.,0.3,0.1,0.1,0.4,0.,0.,0.2])))
	number =10

	persons = initialize_population(number,arrival_law,departure_law,floor_law)


	# Intialize Tower


	nb_floors=8
	nb_elevators=1
	# persons=[Person(arrival,departure) for arrival,departure in zip(range(700,1050,50),range(1800,1870,10))]

	# for index,person in enumerate(persons):
	# 	person.add_new_intention(person.arrival+index*10,index%nb_floors)
	
	tower = Tower(persons,nb_floors,nb_elevators)


	#parameters agent
	state_dim = translate_state(tower.state).shape[0]
	action_dim = 5
	epsilon = 0.9
	epsilon_decay=0.9998
	gamma=0.99
	learning_rate=0.01

	#initializing memory
	capacity = 100000
	batch_size = 512
	memory = ReplayMemory(capacity)

	#init agent
	agent = AgentLinearSarsa(state_dim,action_dim,epsilon,epsilon_decay,gamma)
	agent.to(device)
	# agent_elev = IntelligentElevator()
	# a target net to avoid over-fitting the current policy
	target_net = AgentLinearSarsa(state_dim,action_dim,epsilon,epsilon_decay,gamma)
	target_net.load_state_dict(agent.state_dict())
	target_net.to(device)
	#using RMSprop but Adam works perfectly
	optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
	results= []

	agent_intel = IntelligentElevator()
	#run trials
	number_of_trials =100000
	for trial in range(number_of_trials):



		### creating memories
		for _ in range(2):

			#get state
			state = tower.state
			#metric: length waited
			#bool for final state
			done = False
			#run episode
			while not done:
				#select action
				action = agent_intel.select_action(state)
				#process action
				new_state,reward,done = tower.step(action)
				#adding number of  unsatisfied people to the metric
				reward = -np.log1p(-reward)
				state = new_state
				memory.push(translate_state(state,device=device).float().view(1,-1), torch.tensor(action).view(1,1).long(), translate_state(new_state,device=device).float().view(1,-1), torch.tensor([reward]).float())


			tower.reset()


		#get state
		state = tower.state
		#cast it in torch
		state = translate_state(state,device=device)
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
			new_state = translate_state(new_state,device=device)
			#adding number of  unsatisfied people to the metric
			summed_rwd += reward
			#adding transition to the memory
			reward = -np.log1p(-reward)
			memory.push(state.float().view(1,-1), torch.tensor(action).view(1,1).long(), new_state.float().view(1,-1), torch.tensor([reward]).float())
			state = new_state
			#updating agent
			if trial >=1:
				agent.update(memory,batch_size,gamma,optimizer,target_net)

		print("Trial Nb: ",trial)
		print("Number of minutes waited: ",-summed_rwd)
		print("Percentage of random actions: ", agent.epsilon)
		results.append(-summed_rwd)
		# tower.reset()
		persons = initialize_population(number,arrival_law,departure_law,floor_law)
		tower = Tower(persons,nb_floors,nb_elevators)

		# assigning network to the target net (for value iteration)
		if trial%2 == 0:
			target_net.load_state_dict(agent.state_dict())
		agent.decay()
	return results




def test_policy_gradient():
	device = torch.device("cuda")
	arrival_law = PtDistribution(distrib.uniform.Uniform(700,901))
	departure_law= PtDistribution(distrib.uniform.Uniform(1800,1870))
	floor_law = PtDistribution(distrib.categorical.Categorical(probs=torch.Tensor([0.,0.3,0.1,0.1,0.4,0.,0.,0.2])))
	number =10

	persons = initialize_population(number,arrival_law,departure_law,floor_law)


	# Intialize Tower

	nb_floors=8
	nb_elevators=1
	# persons=[Person(arrival,departure) for arrival,departure in zip(range(700,1050,50),range(1800,1870,10))]

	# for index,person in enumerate(persons):
	# 	person.add_new_intention(person.arrival+index*10,index%nb_floors)
	
	tower = Tower(persons,nb_floors,nb_elevators)


	#parameters agent
	state_dim = translate_state(tower.state).shape[0]
	action_dim = 5
	epsilon = 0.9
	epsilon_decay=0.998
	gamma=0.9
	learning_rate=0.01

	#initializing memory
	capacity = 100000
	batch_size = 512
	memory = ReplayMemory(capacity)

	#init agent
	agent = PolicyGradientAgent(state_dim,action_dim,epsilon,epsilon_decay,gamma)
	agent.to(device)
	# agent_elev = IntelligentElevator()
	#using RMSprop but Adam works perfectly
	optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
	results= []

	# agent_intel = IntelligentElevator()
	#run trials
	number_of_trials =100000
	for trial in range(number_of_trials):



		# ### creating memories
		# for _ in range(2):

		# 	#get state
		# 	state = tower.state
		# 	#metric: length waited
		# 	#bool for final state
		# 	done = False
		# 	#run episode
		# 	while not done:
		# 		#select action
		# 		action = agent_intel.select_action(state)
		# 		#process action
		# 		new_state,reward,done = tower.step(action)
		# 		#adding number of  unsatisfied people to the metric
		# 		state = new_state
		# 		memory.push(translate_state(state,device=device).float().view(1,-1), torch.tensor(action).view(1,1).long(), translate_state(new_state,device=device).float().view(1,-1), torch.tensor([reward]).float())
		# 	agent.update(memory,gamma,optimizer)

		# 	tower.reset()


		#get state
		state = tower.state
		#cast it in torch
		state = translate_state(state,device=device)
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
			new_state = translate_state(new_state,device=device)
			#adding number of  unsatisfied people to the metric
			summed_rwd += reward
			#adding transition to the memory
			memory.push(state.float().view(1,-1), torch.tensor(action).view(1,1).long(), new_state.float().view(1,-1), torch.tensor([reward]).float())
			state = new_state
			#updating agent
		agent.update(memory,gamma,optimizer)

		print("Trial Nb: ",trial)
		print("Number of minutes waited: ",-summed_rwd)
		print("Percentage of random actions: ", agent.epsilon)
		results.append(-summed_rwd)
		# tower.reset()
		persons = initialize_population(number,arrival_law,departure_law,floor_law)
		tower = Tower(persons,nb_floors,nb_elevators)

		agent.decay()
	return results


def test_intelligent():
	arrival_law = PtDistribution(distrib.uniform.Uniform(700,901))
	departure_law= PtDistribution(distrib.uniform.Uniform(1800,1870))
	floor_law = PtDistribution(distrib.categorical.Categorical(probs=torch.Tensor([0.,0.3,0.1,0.1,0.4,0.,0.,0.2])))
	number =10

	persons = initialize_population(number,arrival_law,departure_law,floor_law)


	# Intialize Tower


	nb_floors=8
	nb_elevators=1
	# persons=[Person(arrival,departure) for arrival,departure in zip(range(700,1050,50),range(1800,1870,10))]

	# for index,person in enumerate(persons):
	# 	person.add_new_intention(person.arrival+index*10,index%nb_floors)
	
	tower = Tower(persons,nb_floors,nb_elevators)


	agent = IntelligentElevator()


	#run trials
	results = []
	number_of_trials =1000
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
	return results




def test_stupid():
	arrival_law = PtDistribution(distrib.uniform.Uniform(700,901))
	departure_law= PtDistribution(distrib.uniform.Uniform(1800,1870))
	floor_law = PtDistribution(distrib.categorical.Categorical(probs=torch.Tensor([0.,0.3,0.1,0.1,0.4,0.,0.,0.2])))
	number =10

	persons = initialize_population(number,arrival_law,departure_law,floor_law)


	# Intialize Tower


	nb_floors=8
	nb_elevators=1
	# persons=[Person(arrival,departure) for arrival,departure in zip(range(700,1050,50),range(1800,1870,10))]

	# for index,person in enumerate(persons):
	# 	person.add_new_intention(person.arrival+index*10,index%nb_floors)
	
	tower = Tower(persons,nb_floors,nb_elevators)


	agent = StupidElevator(nb_floors)


	#run trials
	results = []
	number_of_trials =1000
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
	return results

def manual_debug():



	# Intialize Tower


	nb_floors=3
	nb_elevators=1
	persons=[Person(arrival,departure) for arrival,departure in zip(range(700,1050,50),range(1800,1870,10))]

	for index,person in enumerate(persons):
		person.add_new_intention(person.arrival+index*10,index%nb_floors)
	
	tower = Tower(persons,nb_floors,nb_elevators)




	#run trials
	results = []
	number_of_trials =100
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
			action = int(input("action"))
			#process action
			new_state,reward,done = tower.step(action)
			#adding number of  unsatisfied people to the metric
			summed_rwd += reward

			state = new_state
			print(new_state)
			print(reward)
		results.append(-summed_rwd)


		tower.reset()
		
	return results


def test_simpler_env():

	#initialize environement
	dim = 2
	min_value = -0.3
	max_value = 0.3
	drag = 0.9
	gamma = 0.93
	env = Environement(dim,min_value,max_value,drag)
	env.reset()


	#initialise temparature
	temperature = 1
	temperature_decay = 0.998

	agent = AgentReinforceTorch(env.state_dim,env.action_dim,temperature,temperature_decay)

	# define the optimizer
	learning_rate = 0.001
	optimizer = optim.Adam(agent.decision_module.parameters(),lr=learning_rate)

	# Building memory
	capacity = 200000
	batch_size=500
	memory = ReplayMemory(capacity)

	nb_iterations=10000

	lens=[]
	#iterate
	for iteration in range(nb_iterations):
		state = env.state
		done = False
		history = []
		summed_rwd= 0
		t=0
		
		while not done:
			action = agent.select_action(state)
			new_state, reward, done = env.step(action)
			history.append((state,action,reward))
			state = new_state
			t+=1
			if t>1600:
				break


		#discounting and normalizing values
		values=[]
		#discount and sum
		for i in range(len(history)):
			values.append(sum([x[2]*gamma**t for t,x in enumerate(history[i:])]))
		values = np.array(values)
		#normalize
		values = (values-np.mean(values))/(np.std(values)+0.001)
		#adding to memory
		for i in range(len(history)):
			memory.push(history[i][0].view(1,-1),torch.tensor([history[i][1]]).long(),torch.tensor([values[i]]).float())
		agent.update_network(memory,optimizer)
		#empty memory
		memory.memory=[]
		memory.position=0

		#plots and prints
		print("minutes waited:",t)
		print("agent temperature:",agent.temperature)
		print("________")
		env.reset()
	# # plt.plot(test_stupid(),c="k",label="Naive agent")


def test_simpler_env2():
	device = torch.device("cuda")
	gamma = 0.93
	arrival_law = PtDistribution(distrib.uniform.Uniform(700,901))
	departure_law= PtDistribution(distrib.uniform.Uniform(1800,1870))
	floor_law = PtDistribution(distrib.categorical.Categorical(probs=torch.Tensor([0.,0.3,0.1,0.1,0.4,0.,0.,0.2])))
	number =15

	persons = initialize_population(number,arrival_law,departure_law,floor_law)


	# Intialize Tower


	nb_floors=8
	nb_elevators=1
	# persons=[Person(arrival,departure) for arrival,departure in zip(range(700,1050,50),range(1800,1870,10))]

	# for index,person in enumerate(persons):
	# 	person.add_new_intention(person.arrival+index*10,index%nb_floors)
	
	tower = Tower(persons,nb_floors,nb_elevators)


	#initialise temperature
	temperature = 1
	temperature_decay = 0.9994
	state_dim = translate_state(tower.state).shape[0]
	action_dim = 5

	agent = AgentReinforceTorch(state_dim,action_dim,temperature,temperature_decay)
	agent.decision_module.to(device)

	# define the optimizer
	learning_rate = 0.0001
	optimizer = optim.Adam(agent.decision_module.parameters(),lr=learning_rate)

	# Building memory
	capacity = 200000
	batch_size=500
	memory = ReplayMemory(capacity)

	nb_iterations=1000000

	lens=[]
	#iterate
	for iteration in range(nb_iterations):
		state = translate_state(tower.state,device=device)
		done = False
		history = []
		summed_rwd= 0
		t=0
		
		while not done:
			action = agent.select_action(state)
			new_state, reward, done = tower.step(action)
			history.append((state,action,reward))
			state = translate_state(new_state,device=device)
			summed_rwd+=reward


		#discounting and normalizing values
		values=[]
		#discount and sum
		for i in range(len(history)):
			values.append(sum([x[2]*gamma**t for t,x in enumerate(history[i:])]))
		values = np.array(values)
		#normalize
		values = (values-np.mean(values))/(np.std(values)+0.001)
		#adding to memory
		for i in range(len(history)):
			memory.push(history[i][0].view(1,-1),torch.tensor([history[i][1]]).long().to(device),torch.tensor([values[i]]).float().to(device))
		agent.update_network(memory,optimizer)
		#empty memory
		memory.memory=[]
		memory.position=0

		#plots and prints
		print("minutes waited:",-summed_rwd)
		print("agent temperature:",agent.temperature)
		print("________")
		#agent.decay()
		persons = initialize_population(number,arrival_law,departure_law,floor_law)
		tower = Tower(persons,nb_floors,nb_elevators)



if __name__ == "__main__":
	#manual_debug()
	# test_policy_gradient()
	# plt.plot(test_linear_sarsa(),c="b",label="Sarsa")
	# # plt.plot(test_stupid(),c="k",label="Naive agent")
	# plt.plot(test_intelligent(),c="r",label="Smart agent")
	# plt.legend()
	# plt.xlabel("Trial")
	# # plt.ylim(0,400)
	# plt.show()
	test_simpler_env2()



	# # plt.plot(test_stupid(),c="k",label="Naive agent")
