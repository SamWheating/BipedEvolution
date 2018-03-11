import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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

SIMULATION_STEPS = 400

INITIAL_POPULATION = 500

NUM_GENERATIONS = 10
POPULATION_SIZE = 50

ELITISM = 10			# best specimens carried to the next round unaltered
MULTIPLICATION_FACTOR = 5 # rate at which elite multiply on each generation

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

def mutate(genome1, magnitude):

	# Randomized mutations to avoid local maximums

	# Randomize a small fraction of biases 
	for key, value in genome1.biases.items():
		for row in range(value.shape[0]):
			value[row] += np.random.uniform((-1*magnitude), magnitude)


	# Randomize a small fraction of weights
	for key, value in genome1.weights.items():
		for row in range(value.shape[0]):
			for col in range(value.shape[1]):
				value[row][col] += np.random.uniform((-1*magnitude), magnitude)

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

	detector = [] # used for deadlock prevention

	for j in range(SIMULATION_STEPS):

		# TODO: Deadlock prevention

		if render: env.render()
		#env.render()

		if j == 0:
			observation, reward, done, info = env.step(initial_action)

		else:

			prediction = sess.run(logits, feed_dict={X: np.transpose(np.array(previous_state))})
			observation, reward, done, info = env.step(prediction[0])

			indicator = (round(observation[0], 4))
			 								# take action dictated by neural network
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

population = [Genome() for i in range(INITIAL_POPULATION)]

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
max_strengths = []
deviations = []

rates = [0.1, 0.1, 0.01, 0.01, 0.01, 0.01]  # hardcoded af

NUM_GENERATIONS = len(rates)

best_sample = Genome()
best_strength = 0

while(generation_number < NUM_GENERATIONS):

	rate = rates[generation_number]

	next_population = []

	strengths = [0 for i in range(len(population))]

	# Test every member of population
	
	render = False

	for i in range(len(population)):
		#if generation_number  == (NUM_GENERATIONS-1): render = True
		strengths[i] = fitness_test(population[i], render)
		print("concluded simulation generation %i, species %i" % (generation_number, i))

	print("\n new generation \n")

	print(strengths, "\n\n")

	next_population = []

	average_strengths.append(sum(strengths)/len(strengths))
	deviations.append(np.std(strengths))

	# take the 10 strongest into the next generation (unaltered)
	strengths_sorted = sorted(strengths)
	max_strengths.append(strengths_sorted[-1])
	
	for strength in strengths_sorted[(len(strengths_sorted)-10):]:
		print(strength)
		if strength > best_strength:
			best_strength = strength
			best_sample = population[strengths.index(strength)]

		for _ in range(MULTIPLICATION_FACTOR):
			next_population.append(population[strengths.index(strength)])

	# apply mutations
	for i in range(len(next_population)):
		if i % MULTIPLICATION_FACTOR != 0:
			next_population[i] = mutate(next_population[i], rate)


	print(len(next_population))

	population = next_population

	generation_number += 1

avg_strength = plt.plot(average_strengths, 'r', label='Average fitness of population')
std_deviation = plt.plot(deviations, 'g', label='std deviation of population fitness')
max_streng = plt.plot(max_strengths, 'k', label='max fitness of population')
plt.legend()
#plt.legend(handles=[avg_strength, std_deviation, max_streng])
plt.title(('Progression of' + str(NUM_GENERATIONS) + "generations"))
plt.ylabel('Generation Number')

plt.show()

while(True):
	print("strongest observed sample:", best_strength)
	for i in range(len(next_population)):
		a = fitness_test(next_population[i], True)
