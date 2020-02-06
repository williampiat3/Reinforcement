import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import itertools
from bandits import UCBAgent
from functools import wraps

plt.ion()

# Disclaimer: This script uses a lot of RAM it creates billions of UCB Agents for selecting actions 

class UCBAgentForTree(UCBAgent):
	"""
	UCB adapted for the nodes of the tree
	it takes by default the action to stop changing the state
	However this approach works fine with a small number of actions
	"""
	def __init__(self,K,c):
		UCBAgent.__init__(self,K,c)
	#this function just force to try to stop once and then to explore other moves
	def sample_regardless_count(self):
		if np.sum(self.count)==0:
			return self.nb_actions -1
		else:
			action = np.argmax(self.means + self.c*np.sqrt(np.log(self.time))/np.sqrt(self.count+1))
			self.time +=1
			return action

#decorator to remove gradient from the computations: this should allow less memory consumption
def no_grad(func):
	@wraps(func)
	def wrapper(*args,**kwargs):
		with torch.no_grad():
			return func(*args,**kwargs)
	return wrapper


class LoadingPattern(object):
	"""
	Class environement to model a loading pattern
	"""
	def __init__(self,size,kernels):
		with torch.no_grad():
			self.size = size
			#initialization of an unbalanced core 
			self.state = torch.randn(1,1,self.size,self.size)
			self.state[:,:,self.size//2:,:]+= torch.rand(1,1,self.size//2,self.size)*2
			self.state[:,:,:self.size//2,:]-= torch.rand(1,1,self.size//2,self.size)*2
			self.state = self.state.clamp(-1,1)

			#intialization of the actions
			self.actions = {}
			for i,perm in enumerate(itertools.combinations(itertools.product(range(self.size),repeat=2),2)):
				self.actions[i]=perm
			#adding an action to stop permuting the assemblies
			self.actions[len(self.actions)]="EOF"
			#clone the state to be able to come back to the initial state (used for models)
			self.save = self.state.clone()
			self.kernels = kernels
			# assess starting quality so a to improve it
			self.initial_quality = self.assess_quality()

	# Assess quality by computing the norm of the convolution of the core with a gaussian kernel
	@no_grad
	def assess_quality(self):
		
		return sum([torch.sum(F.conv2d(self.state, kernel,padding=3)**2).item() for kernel in self.kernels])

	# Assessing improvement as the improvement in quality
	@no_grad	
	def assess_improvement(self):
		return self.initial_quality - self.assess_quality()

	# Plot function to display the state 
	def display_state(self):
		plt.imshow(self.state.numpy().squeeze())
		plt.show()
		plt.pause(0.01)

	#backup function to get back to the saved state
	@no_grad
	def restart(self):
		self.state = self.save.clone()


	#Function to completly reboot the environement if you want to run multiple episodes
	@no_grad
	def reboot(self):
		self.state = torch.randn(1,1,self.size,self.size)
		self.state[:,:,self.size//2:,:]+= torch.rand(1,1,self.size//2,self.size)*2
		self.state[:,:,:self.size//2,:]-= torch.rand(1,1,self.size//2,self.size)*2
		self.state = self.state.clamp(-1,1)
		self.save = self.state.clone()
		self.initial_quality = self.assess_quality()


	# functin to overwritte the saved state with the current state
	@no_grad	
	def store(self):
		self.save = self.state.clone()


	#function to restore a given state
	@no_grad	
	def restore_from_state(self,state):
		self.state=state

	#The step here is the permutation of two assemblies
	@no_grad
	def step(self,action):
		
		if action != (len(self.actions)-1):
			loc1,loc2= self.actions[action]
			temp = self.state[:,:,loc1[0],loc1[1]].clone()
			self.state[:,:,loc1[0],loc1[1]] = self.state[:,:,loc2[0],loc2[1]]
			self.state[:,:,loc2[0],loc2[1]] = temp
			return self.state,-0.7
		else:
			return self.state,self.assess_improvement()




class MonteCarloTreeSearch():
	"""
	Class for itering in the action tree and be able to cut branch smoothly to exploit them afterwards
	"""
	def __init__(self,number_actions,gamma,root=False):
		#branches of our tree: as many as the number of actions
		self.childs = dict.fromkeys(range(number_actions),None)
		#the decision is taken by a UCB agent
		self.decision = UCBAgentForTree(number_actions,17)

		self.number_actions=number_actions
		self.gamma = gamma
		self.root=root

	#Function to run an episode recursively  and return he expected summed reward form each action to update the values of our nodes
	def run_episode(self,depth,env):
		#if end depth reached: return the improvment 
		if depth ==0:
			return env.assess_improvement()
		#else take an action
		action = self.decision.sample_regardless_count()
		new_state,reward=env.step(action)
		#the next move have to be handled by the child for the attempted action
		#if it is not existing we have to create it
		if self.childs[action] is None:
			new_child = self.__class__(self.number_actions,self.gamma)
			self.childs[action]=new_child
		#Running the episode of the child
		result=self.childs[action].run_episode(depth-1,env)
		#If the action is not final
		if action != (self.number_actions-1):
			q_value = reward + self.gamma*np.max([self.childs[action].decision.means[i] for i in range(self.number_actions) if self.childs[action].decision.count[i]>0])
		#if the action is terminating
		else:
			q_value=reward
		#updating the value of the UCB agent
		self.decision.update(action,q_value)
		return q_value

	#Select greedily the action
	def return_best_action(self):
		best_index = np.argmax(self.decision.means)
		return best_index

	#function to get the branch of the tree so as to exploit it the next time step
	def __getitem__(self,key):
		branch = mcts.childs[key]
		branch.root = True
		return branch



if __name__ == "__main__":
	#Gaussian kernels
	kernels=[torch.ones(1,1,4,4)/16,torch.ones(1,1,2,2)/16]
	#loading pattern of size 10
	load = LoadingPattern(10,kernels)
	#little display
	load.display_state()
	gamma = 0.99
	#running a lot of sample before each decision of a certain depth
	number_of_episodes=15000
	depth=3
	#initialize tree
	mcts = MonteCarloTreeSearch(len(load.actions),gamma,root=True)
	#number of total actions
	number_of_iterations=200


	for iteration in range(number_of_iterations):
		# storing state to prepare it to be probed by the tree
		load.store()
		#run episodes
		for _ in range(number_of_episodes):
			mcts.run_episode(depth,load)
			load.restart()
		#select best action
		action = mcts.return_best_action()
		#taking the step
		load.step(action)
		#display
		load.display_state()
		if action == (len(load.actions)-1):
			break
		mcts = mcts[action]
	# print(load.assess_improvement())



	


	