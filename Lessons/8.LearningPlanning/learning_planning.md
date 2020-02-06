# Integrating learning and planning 

This lesson introduces the concept of a model based RL agent: the agent has the ability to simulate the experiences (although approximatly) and to learn from simulated experience. The model of the environement is formaly a transition model coupled with a regression model that predicts the reward the agent will get. In zero summed two players games it can be simulated using a rollout policy (see for instance the [AlphaGo](https://vk.com/doc-44016343_437229031?dl=56ce06e325d42fbc72) paper on how the rollout policy is used)

It allows using Monte Carlo Tree Search (MCTS) to explore the best actions that could be performed in a short distant future

MCTS was, before DeepMind's value network, the state of the art in terms of Go playing programs: it samples the most promising actions using the different rollouts that the algorithm performed from the current state. 

I implemented a simple MCTS algorithm in the file [here](https://github.com/williampiat3/Reinforcement/blob/master/Lessons/8.LearningPlanning/monte_carlo.py) however it consumes a lot of RAM especially if you want to increase the depth of the tree, the environement that I coded is the one of an unbalanced board, the tree has to figure out what are the best permutations that is has to perform to make it balanced (16 Gb of RAM are necessary for the default parameters)
