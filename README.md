# BipedEvolution

Evolutionary optimization of neural networks implemented in Python

Created as a personal challenge for HackUVIC 2018 Hackathon.

Original plan: try to beat the OpenAI walking Biped challenge using genetic optimization of Artificial Neural Networks, more specifically the NEAT algorithm.

https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies

Algorithm Specification taken from here:
http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf

## Challenge:

Create a system for genetic optimization of neural networks using the NEAT algorithm which can learn to walk given the simple 2D biped model from the openAI gym.

https://gym.openai.com/envs/BipedalWalker-v2/

## Results

I had trouble building custom topologies in TensorFlow, so I switched to genetic optimization of weights and biases in a fixed-size network. 

Then I had issues writing an effective genetic crossover for neural-network based genomes, so I switched to a survival-of-the-fittest and mutation-based approach:

  1) Generate 500 samples with random weighting and biases.
  2) Test all of them, keep the 10 top performers.
  3) Make 5 copies of each and apply small, random changes.
  4) Go to (2), repeat for number of generations. Reduce amplitude of mutations for each generation.

The results were kind of promising as the biped moved from a totally uncontrolled state in the last generation to a more consistent state after seven generations.

![Comparison of first and seventh generation](https://github.com/SamWheating/BipedEvolution/blob/master/sidebyside.gif?raw=true)
