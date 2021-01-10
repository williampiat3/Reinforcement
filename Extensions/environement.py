import numpy as np
from collections import OrderedDict,namedtuple
import random
import torch


#intentions are an hour and a floor meaning that at that time people need to be at this floor
class Person():
	def __init__(self,arrival,departure):
		# intention dict describing the will of the person
		self.intentions=OrderedDict([(arrival,0),(departure,0)])
		#position in the building
		self.position=0
		#arival and departure time
		self.arrival = arrival
		self.departure = departure
		#bolean if at work
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

	def __len__(self):
		return len(self.content)

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
		#elevator capacity
		self.capacity=13

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
		self.position_elevators = [0]*len(self.elevators)


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

			if all([not (person in floor.content )for floor in self.floors]) and all([not (person in floor.content )for floor in self.elevators]) and person.here:
				print("waza")

			if person.position == 0 and not person.here and person.arrival<self.time and person.departure>self.time:
				# We add it to floor 0
				self.floors[0].content.append(person)
				person.here = True
			#if the person wants to leave and is at floor 0
			if person.position == 0 and person.here and person.departure<self.time and person in self.floors[0].content:
				#we pop it from the content of floor 0
				self.floors[0].content.pop(self.floors[0].content.index(person))
				person.here = False
		
	#the state will be composed of:filling of the floors
	#							   if people waiting
	#							   floor of the elevator
	#							   filling of the elevator
	#							  destinations of the elevator
	#							   destination of the persons at the floor
	# This will be given to the RL algorithm
	
	@property
	def state(self):
		#filling of the floors 
		floor_filling = np.array([len(floor) for floor in self.floors])/len(self)
		elevator_position = np.zeros((self.nb_elevators,self.nb_floors))
		#elevators destination
		elevators_destination = np.zeros((self.nb_elevators,self.nb_floors))
		elevators_filling = np.zeros(self.nb_elevators)
		for index_el,el in enumerate(self.elevators):
			elevators_filling[index_el]= len(el)/self.capacity
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
		self.time +=10
		#making people arrive or laeve the tower
		self.update_arrival_departure()
		reward=0


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


class Environement(object):
	"""
	The environement here is a table where a ball has to reach a hole
	The destination and the position of the ball are initialized randomly
	"""
	def __init__(self,dim,min_value,max_value,drag):
		#dimension of te table
		self.dim = dim
		# max dimensions of the table
		self.min_value = min_value
		self.max_value = max_value
		
		#defining action and state space
		self.action_dim = self.dim*2
		self.state_dim = self.dim*2
		self.drag = drag
	
	def reset(self):
		"""
		function to restart another episode
		"""
		self.destination = torch.rand(self.dim)*(self.max_value-self.min_value)+self.min_value
		self.position = torch.rand(self.dim)*(self.max_value-self.min_value)+self.min_value
		self.momentum = torch.zeros(self.dim)

	@property
	def state(self):
		"""
		State that would be given to the agent: relative position + speed
		"""
		return torch.cat([(self.position-self.destination),self.momentum],dim=0)
	

	def step(self,action):
		"""
		Main function for executing a timestep
		"""
		with torch.no_grad():
			#action dimension
			dim = action //2
			#action way
			toward = (action%2)*2 -1
			
			#decaying momentum
			self.momentum *= self.drag
			#adding action to momentum
			self.momentum[dim] += toward*0.02
			#moving 
			self.position = self.position+self.momentum
			#if hitting a wall nullify speed and clamp position
			for i in range(self.position.shape[0]):
				if self.position[i]>self.max_value or self.position[i]<self.min_value:
					self.position[i]=torch.clamp(self.position[i],self.min_value,self.max_value)
					self.momentum[i]=0
			
			state = self.state
			#checking if we arrived
			distance = torch.sqrt(torch.sum((self.position-self.destination)**2))
			if distance.item() < 0.01:
				reward = 1
				done=True
			else:
				reward = -0.01
				done = False
			return state,reward,done

