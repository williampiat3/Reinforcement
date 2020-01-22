# Introduction

## About the lesson
The course presented here is a computational approach to reinforcement learning but RL is a subject of many sciences (Psychology, Economics, neurosciences...)

### Main ideas
This is a general introduction of Reinforcement Learning with some important notions that should be kept in mind before passing to the other lessons:
* Reinforcement learning is the closest paradigm to "natural" learning:  our agent learns by interacting with its environement, using its perception of the environement to build progressively a policy to optimize the rewards it will receive in a not so distant future.
* This is not supervised learning where we know the result and try to have it infered by an algorithm, we have a state and reward feed and try to infer information on the state feed to act on the reward feed. This is the goal of reinforcement: _Maximizing the total future reward_ (No one cares about the past, all that matter is what you can do from now on to improve your future)
* The notion of Markovian state is central: "The future is independant of the past given the present" There is enough information in the current state to predit all that could happen in the future: we can therefore build some planification on the current state only.

### What is an agent

There might be multiple components of an agent (not all of them are necessary) for him to interact with the environement:
* A policy: A function that indicates which behavior the agent must have
* A value function: A metric of appreciation of each possible state and/or actions
* A model: A representation of the real world that the agent can improve and probe if necessary

The agent has to find the right compromise between :

* **Exploitation** where it is using its knowledge to select, according to it, the best action

* **Exploration** where the agent tries new actions in the hope to obtain a better outcome than what would have brought a greedy action

