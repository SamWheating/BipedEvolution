import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('BipedalWalker-v2')
env.reset()

initial_action = [0.0, 0.0, 0.0, 0.0]

# Basic Algorithm Plan:
# FIrst round is 500 specimens, totally random. Mass extinction leaves the ten most fit
# Successive rounds, 50 specimens / generation
# Elitism favours top 10, unaltered to next generation
# Muatated clones of the top 10 make up another 40 samples

# going to use a consistent topology because tensorflow is unfriendly with dynamic or irregular topologies
# (Apparently pytorch is better for this)
# This is more or less randomly designed, this is an experiment after all.

# Room for improvement:
# Parallelization of the search space.
# Evaluation of learning / mutation rates
# Genetic crossover (Wouldnt Necessarily Work, but worth trying)
# New Scoring metric based only on distance.
# Better detection of cyclic or recurring behaviour.

# NEURAL NETWORK ARCHITECTURE

n_hidden_1 = 24
n_hidden_2 = 24
n_hidden_3 = 24
num_input = 24 		# output from the walker, number of items in input layer of ANN
num_outputs = 4 	# output size (4 motors speeds)

# CONTROL VARIABLES

SIMULATION_STEPS = 600 		# how many frames to render before timeout
INITIAL_POPULATION = 500	# massive starting population	
POPULATION_SIZE = 50		# successive population size
ELITISM = 10				# best specimens carried to the next round unaltered
MULTIPLICATION_FACTOR = 5 	# rate at which elite multiply on each generation
MUTATION_PROBABILITY = 0.1	# chance of a given connection being mutated

class Genome:

	weights = {}
	biases = {}

	# Each genome has 24*24 + 24*24 + 24*24 + 24* 4 = 1824 weights
	# Each genome has 24 + 24 + 4 = 52 biases
	# = 508 tunable parameters

	def __init__(self):

		self.weights = {
			'h1': np.zeros((num_input, n_hidden_1), dtype='float32'),		# weights of first hidden layer
			'h2': np.zeros((n_hidden_1, n_hidden_2), dtype='float32'),		# weights of second hidden layer
			'h3': np.zeros((n_hidden_1, n_hidden_2), dtype='float32'),		# weights of second hidden layer
			'out': np.zeros((n_hidden_2, num_outputs), dtype='float32')		# weights of output layer connections 
		}

		# Biases may not be necessary here.
		self.biases = {
			'b1': np.zeros(n_hidden_1, dtype='float32'),	# biases of first hidden layer
			'b2': np.zeros(n_hidden_2, dtype='float32'),	# biases of second hidden layer
			'out': np.zeros(num_outputs, dtype='float32'),	# biases of output layer
		}

def mutate(genome, magnitude):

	# Randomized mutations to avoid local maximums

	# Randomize a small fraction of biases 
	for key, value in genome.biases.items():
		for row in range(value.shape[0]):
			if np.random.uniform(0, 1) < MUTATION_PROBABILITY:
				value[row] += np.random.uniform((-1*magnitude), magnitude)


	# Randomize a small fraction of weights
	for key, value in genome.weights.items():
		for row in range(value.shape[0]):
			for col in range(value.shape[1]):
				if np.random.uniform(0, 1) < MUTATION_PROBABILITY:
					value[row][col] += np.random.uniform((-1*magnitude), magnitude)

	return genome


def fitness_test(genome, render=False):

	env.reset()

	# tf Graph input
	X = tf.placeholder("float", [1, num_input])

	# Store layers weight & biases

	def neural_net(x):
		layer_1 = tf.add(tf.matmul(x, genome.weights['h1']), genome.biases['b1'])
		layer_2 = tf.add(tf.matmul(layer_1, genome.weights['h2']), genome.biases['b2'])
		layer_3 = (tf.matmul(layer_2, genome.weights['h3']))
		out_layer = tf.matmul(layer_3, genome.weights['out']) + genome.biases['out']

		return out_layer

	logits = neural_net(X)  # assembles NN logic

	init = tf.global_variables_initializer()

	previous_state = [[0.0] for _ in range(num_input)]
	previous_reward = -200.0

	sess = tf.Session()

	sess.run(init)

	detector = [] # used for deadlock prevention

	for j in range(SIMULATION_STEPS):

		# TODO: Deadlock prevention

		if render: env.render()
		#env.render()

		if j == 0:
			observation, reward, done, info = env.step(initial_action)

		else:

			prediction = sess.run(logits, feed_dict={X: np.transpose(np.array(previous_state))})  # take action dictated by neural network
			observation, reward, done, info = env.step(prediction[0])

			indicator = (round(observation[0], 4))
			 								
			detector.append(indicator)

			if len(detector) > 20:
				detector.pop(0)

		if len(detector) > 10:
			if len(np.unique(detector)) == 1:
				done = True

		if done == True:
			previous_reward = -100

#		tmp = 0

		for item in range(len(observation)):
			previous_state[item][0] = list(observation)[item]	# save input for calculating next move
#			tmp += 1

		if done == True or j == (SIMULATION_STEPS-1):
			print("simulation ended at step %i with reward %f" % (j, previous_reward))
			return reward
			sess.close()
			break 

		previous_reward = reward

# main optimization code

if __name__ == "__main__":

	population = [Genome() for i in range(INITIAL_POPULATION)]

	# Initialize genomes to random parameters

	for genome in population:
		np.random.seed()
		genome.weights['h1'] = np.random.uniform(-1.0, 1.0, size=genome.weights['h1'].shape).astype('float32')
		genome.weights['h2'] = np.random.uniform(-1.0, 1.0, size=genome.weights['h2'].shape).astype('float32')
		genome.weights['h3'] = np.random.uniform(-1.0, 1.0, size=genome.weights['h3'].shape).astype('float32')
		genome.weights['out'] = np.random.uniform(-1.0, 1.0, size=genome.weights['out'].shape).astype('float32')

		genome.biases['b1'] = np.random.uniform(-1.0, 1.0, size=genome.biases['b1'].shape).astype('float32')
		genome.biases['b2'] = np.random.uniform(-1.0, 1.0, size=genome.biases['b2'].shape).astype('float32')
		genome.biases['out'] = np.random.uniform(-1.0, 1.0, size=genome.biases['out'].shape).astype('float32')

	generation_number = 0

	average_strengths = []
	max_strengths = []
	deviations = []

	rates = [0.2, 0.1, 0.05, 0.01, 0.01, 0.01]  # hardcoded mutation rates

	NUM_GENERATIONS = len(rates)

	best_sample = Genome()
	best_strength = 0

	while(generation_number < NUM_GENERATIONS):

		rate = rates[generation_number]

		next_population = []

		strengths = [0 for i in range(len(population))]

		# Test every member of population
		
		render = (generation_number >= 1)  # No need to see the first generation.

		for i in range(len(population)):
			if generation_number  == (NUM_GENERATIONS-1): render = True
			strengths[i] = fitness_test(population[i], render)
			print("concluded simulation generation %i, species %i" % (generation_number, i))

		print("\n new generation \n")

		print(strengths, "\n\n")

		next_population = []

		strengths2 = [value for value in strengths if value >= -50.0]

		average_strengths.append(sum(strengths2)/len(strengths2))
		deviations.append(np.std(strengths2))

		# take the 10 strongest into the next generation (unaltered)
		strengths_sorted = sorted(strengths)
		max_strengths.append(strengths_sorted[-1])
		
		for strength in strengths_sorted[(len(strengths_sorted)-10):]:
			print(strength)
			if strength > best_strength:
				best_strength = strength
				best_sample = population[strengths.index(strength)]

			# append n copies of each of the strongest candidates
			for _ in range(MULTIPLICATION_FACTOR):
				next_population.append(population[strengths.index(strength)])

		# apply mutations to all but one copy of each candidate
		for i in range(len(next_population)):
			if i % MULTIPLICATION_FACTOR != 0:
				next_population[i] = mutate(next_population[i], rate)

		# sanity checking
		print("Next generation size: " + str(len(next_population)))

		# replace population
		population = next_population

		generation_number += 1

	# After all evolution is done, plot the progress over time.

	avg_strength = plt.plot(average_strengths, 'r', label='Average fitness of population')
	std_deviation = plt.plot(deviations, 'g', label='std deviation of population fitness')
	max_streng = plt.plot(max_strengths, 'k', label='max fitness of population')
	plt.legend()

	plt.title(('Progression of' + str(NUM_GENERATIONS) + "generations"))
	plt.ylabel('Generation Number')

	plt.show()

	# play out the strongest specimen after testing
	SIMULATION_STEPS = 1800

	while(True):
		print("strongest observed sample:", best_strength)
		a = fitness_test(best_sample, True)
