import numpy as np
import random
import matplotlib.pyplot as plt

#simple implementation of a K armed bandit
class KArmedBandits():
	def __init__(self,K):
		self.means = np.random.randn(K)

	def step(self,action):
		#sampling the arm as presented in Sutton and Barto
		return np.random.randn(1)+ self.means[action]

class RandomAgent():
	"""
	Random agent that counts average reward
	"""
	def __init__(self,nb_actions):
		self.means = np.zeros(nb_actions)
		self.nb_actions = nb_actions
		self.count = np.zeros(nb_actions)

	def sample(self):
		return random.randint(0,self.nb_actions-1)

	#function to update the running mean (using exact mean iteration)
	def update(self,action,reward):
		self.count[action] += 1
		self.means[action] += (reward-self.means[action])/self.count[action]

class EpsilonGreedyAgent(RandomAgent):
	"""
	Agent that act in a epsilon greedy manner given the possible arms it can pull
	epsilon is for the developer to choose (to decay if need be):
	with probability epsilon the agent picks a random action
	ortherwise it acts greedily
	"""
	def __init__(self,nb_actions,eps):
		
		RandomAgent.__init__(self,nb_actions)
		self.eps = eps


	#function to samplle an action
	def sample(self):
		if random.random()<self.eps:
			#making random action
			return random.randint(0,self.nb_actions-1)
		else:
			#greedy action
			return np.argmax(self.means)



class UCBAgent(EpsilonGreedyAgent):
	"""
	UCB: Upper Confidence Bound Action selection, it gives more probability to the actions that were not tried
	allows for a better assessment of the mean
	"""
	def __init__(self,nb_actions,c):

		EpsilonGreedyAgent.__init__(self,nb_actions,0.)
		self.time=1
		self.c= c
	def sample(self):
		if np.min(self.count)==0:
			action = np.argmin(self.count)
		else:
			action = np.argmax(self.means + self.c*np.sqrt(np.log(self.time))/np.sqrt(self.count))
		self.time +=1
		return action


#main greedy function (can be replace by UCB)
def main_greedy(arms,epsilon,steps,verbose=True):
	bandits = KArmedBandits(arms)
	eps_agent = EpsilonGreedyAgent(arms,epsilon)


	for step in range(steps):
		action=eps_agent.sample()
		reward = bandits.step(action)
		eps_agent.update(action,reward)

	print(bandits.means)
	print(eps_agent.means)



#iterator to get the data about the trials
def main_greedy_plots(arms,epsilon,steps,agent_cls,verbose=True,**kwargs):
	bandits = KArmedBandits(arms)
	eps_agent = agent_cls(**kwargs)
	#for plots
	number_of_optimal_actions=0
	rewards=[]
	#classic loop of action/reaction
	for step in range(steps):
		action=eps_agent.sample()
		reward = bandits.step(action)
		eps_agent.update(action,reward)
		if action == np.argmax(bandits.means):
			number_of_optimal_actions+=1
		#for plots
		
		yield np.mean(reward)/np.max(bandits.means),number_of_optimal_actions/(step+1)
	if verbose:
		print(bandits.means)
		print(eps_agent.means)

#function to create plots about the choices of the algorti
def trials(nb_trials,agent_cls,**kwargs):
	arms = 10
	steps = 1000
	epsilon = 0.1
	total_rws = []
	total_bests=[]
	for k in range(nb_trials):
		av_rws,av_bests = zip(*[(av_rw,av_best) for (av_rw,av_best) in main_greedy_plots(arms,epsilon,steps,agent_cls,verbose=False,**kwargs)])
		total_rws.append(av_rws)
		total_bests.append(av_bests)
	total_rws = np.array(total_rws)
	total_bests = np.array(total_bests)
	plt.plot(total_rws.mean(0))
	plt.title("Average reward received")
	plt.figure()
	plt.plot(total_bests.mean(0))
	plt.title("Percentage of optimal actions")
	plt.show()


	

if __name__ == "__main__":
	# main_greedy(10,4000,0.1)
	#eps_agent =  RandomAgent(arms)
	#eps_agent = EpsilonGreedyAgent(arms,epsilon)
	#eps_agent = UCBAgent(arms,2)
	trials(1000,RandomAgent,nb_actions=10)
