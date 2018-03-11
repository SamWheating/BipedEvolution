import gym
import tensorflow as tf
import numpy as np

env = gym.make('BipedalWalker-v2')
env.reset()

initial_action = [0.0, 0.0, 0.0, 0.0]

# Basic Algorithm Plan:
#
# 50 specimens / generation
# Elitism favours top 10, unaltered to next generation
# 10 new, random specimens are generated for each generation
# Bottom 20 are dropped
# Crossover between random surviving samples fils the other 30 
# Mutations on some percentage of survivig samples (10%?) (not elites)


# gonna use a consistent topology because tensorflow is wack

n_hidden_1 = 24
n_hidden_2 = 24
num_input = 24 # output from the walker, number of items in input layer of ANN
num_outputs = 4 # output size (4 motors speeds)

SIMULATION_STEPS = 200

NUM_GENERATIONS = 10
POPULATION_SIZE = 50
MUTATION_RATE = 0.10
INNER_MUTATION_RATE = 0.05
ELITISM = 10			# best specimens carried to the next round unaltered
KILL_RATE = 10			# worst specimens removed each generation
SPAWN_RATE = 10			# New, random specimens added each generation

class Genome:

	weights = {}
	biases = {}

	# Each genome has 24*12 + 12*12 + 12*4 = 480 weights
	# Each genome has 12 + 12 + 4 = 28 biases
	# = 508 tunable parameters

	def __init__(self):

		self.weights = {
			'h1': np.zeros((num_input, n_hidden_1), dtype='float32'),		# weights of first hidden layer
			'h2': np.zeros((n_hidden_1, n_hidden_2), dtype='float32'),		# weights of second hidden layer
			'out': np.zeros((n_hidden_2, num_outputs), dtype='float32')		# weights of output layer connections 
		}

		self.biases = {
			'b1': np.zeros(n_hidden_1, dtype='float32'),	# biases of first hidden layer
			'b2': np.zeros(n_hidden_2, dtype='float32'),	# biases of second hidden layer
			'out': np.zeros(num_outputs, dtype='float32'),	# biases of output layer
		}


def crossover(genome1, genome2):
	# Randomized splicing of strong genomes to produce (theoretically) superior offspring
	# Initial Version: Simply take random weight and bias arrays and combine into a new genome

	# ToDo: better crossover

	genome3 = Genome()

	draw = np.random.uniform(0, 1.0)
	
	if draw > 0.5:
		genome3.weights = genome1.weights
	else: 
		genome3.weights = genome2.weights

	draw = np.random.uniform(0, 1.0)
	if draw > 0.5:
		genome3.biases = genome1.biases
	else: 
		genome3.biases = genome2.biases	

	return genome3

def mutate(genome1):

	# Randomized mutations to avoid local maximums

	# Randomize a small fraction of biases 
	for key, value in genome1.biases.items():
		for row in range(value.shape[0]):
				draw = np.random.random_sample()
				if draw < INNER_MUTATION_RATE:
					value[row] = np.random.uniform(-1.0, 1.0)


	# Randomize a small fraction of weights
	for key, value in genome1.weights.items():
		for row in range(value.shape[0]):
			for col in range(value.shape[1]):
				draw = np.random.random_sample()
				if draw < INNER_MUTATION_RATE:
					value[row][col] = np.random.uniform(-1.0, 1.0)

	return genome1


def fitness_test(genome1, render=False):

	env.reset()

	# tf Graph input
	X = tf.placeholder("float", [1, num_input])

	# Store layers weight & biases

	def neural_net(x):
		# Hidden fully connected layer with 12 neurons
		layer_1 = tf.add(tf.matmul(x, genome1.weights['h1']), genome1.biases['b1'])
		# Hidden fully connected layer with 12 neurons
		layer_2 = tf.add(tf.matmul(layer_1, genome1.weights['h2']), genome1.biases['b2'])
		# Output fully connected layer with 4 neurons to control the four inputs of the biped.
		out_layer = tf.matmul(layer_2, genome1.weights['out']) + genome1.biases['out']

		return out_layer

	logits = neural_net(X)  # assembles NN logic

	init = tf.global_variables_initializer()

	previous_state = [[0.0] for _ in range(num_input)]
	previous_reward = -200.0

	sess = tf.Session()

	sess.run(init)

	for j in range(SIMULATION_STEPS):

		# TODO: Deadlock prevention

		env.render()

		if j == 0:
			observation, reward, done, info = env.step(initial_action)

		else:

			prediction = sess.run(logits, feed_dict={X: np.transpose(np.array(previous_state))})
			observation, reward, done, info = env.step(prediction[0]) 								# take action dictated by neural network
			#print(observation)

		tmp = 0

		for item in range(len(observation)):
			previous_state[item][0] = list(observation)[item]
			tmp += 1

		if done == True or j == (SIMULATION_STEPS-1):
			print("simulation ended at step %i with reward %f" % (j, previous_reward))
			return previous_reward
			sess.close()
			break 


		previous_reward = reward

# main optimization code

population = [Genome() for i in range(POPULATION_SIZE)]

# Initialize genomes to random parameters

for genome in population:
	np.random.seed()
	genome.weights['h1'] = np.random.uniform(-1.0, 1.0, size=genome.weights['h1'].shape).astype('float32')
	genome.weights['h2'] = np.random.uniform(-1.0, 1.0, size=genome.weights['h2'].shape).astype('float32')
	genome.weights['out'] = np.random.uniform(-1.0, 1.0, size=genome.weights['out'].shape).astype('float32')
	genome.biases['b1'] = np.random.uniform(-1.0, 1.0, size=genome.biases['b1'].shape).astype('float32')
	genome.biases['b2'] = np.random.uniform(-1.0, 1.0, size=genome.biases['b2'].shape).astype('float32')
	genome.biases['out'] = np.random.uniform(-1.0, 1.0, size=genome.biases['out'].shape).astype('float32')

generation_number = 0

average_strengths = []

while(generation_number < NUM_GENERATIONS):

	strengths = [0 for i in range(POPULATION_SIZE)]

	# Test every member of population
	
	render = False

	for i in range(POPULATION_SIZE):
		if generation_number  == (NUM_GENERATIONS-1): render = True
		strengths[i] = fitness_test(population[i], render)
		print("concluded screening simulation %i" % i)

	next_population = []

	average_strengths.append(sum(strengths)/len(strengths))

	# kill the 10 weakest

	strengths_sorted = sorted(strengths)

	for strength in strengths_sorted[0:KILL_RATE]:
		population.pop(strengths.index(strength))
		strengths.pop(strengths.index(strength))

	print(len(next_population))

	# Generate some by crossover

	remaining_population = POPULATION_SIZE - KILL_RATE

	for _ in range(POPULATION_SIZE - (ELITISM + SPAWN_RATE)):
		next_population.append(crossover(population[np.random.randint(0, remaining_population)], population[np.random.randint(0, remaining_population)]))

	print(len(next_population))

	# apply mutations
	for i in range(POPULATION_SIZE - (ELITISM+SPAWN_RATE)):
		draw = np.random.random_sample()
		if draw < MUTATION_RATE:
			next_population[i] = mutate(next_population[i])


	print(len(next_population))

	# take the 10 strongest into the next generation (unaltered)

	strengths_sorted = sorted(strengths)
	
	for strength in strengths_sorted[(len(strengths_sorted)-10):]:
		next_population.append(population[strengths.index(strength)])


	print(len(next_population))


	# Spawn new specimens

	for _ in range(SPAWN_RATE):
		a = Genome()
		np.random.seed()
		a.weights['h1'] = np.random.uniform(-1.0, 1.0, size=genome.weights['h1'].shape).astype('float32')
		a.weights['h2'] = np.random.uniform(-1.0, 1.0, size=genome.weights['h2'].shape).astype('float32')
		a.weights['out'] = np.random.uniform(-1.0, 1.0, size=genome.weights['out'].shape).astype('float32')
		a.biases['b1'] = np.random.uniform(-1.0, 1.0, size=genome.biases['b1'].shape).astype('float32')
		a.biases['b2'] = np.random.uniform(-1.0, 1.0, size=genome.biases['b2'].shape).astype('float32')
		a.biases['out'] = np.random.uniform(-1.0, 1.0, size=genome.biases['out'].shape).astype('float32')
		next_population.append(a)


	print(len(next_population))

	population = next_population

	generation_number += 1

print(average_strengths)