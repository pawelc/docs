# Neural Turing Machine
Nice experiments tesing if network can learn to copy, use associative memory, model n-gram. Simple differentiable controller which can read and write to 
memory using location addressing (shift) and associative addressing. Controller learns to output weights over the memory and its output is the output of the 
model or can be mixed with output of some other net?

# Hybrid computing using a neural network with dynamic external memory
This is the Neural Turing Machine with improved addressing (link Matrix keeping the order in which memory was accessed, removing convolutional shift). 
The link matrix can be sparse without decrease of the performance. Model is learned using supervised learning and also reinforce algorithms.
DNC also keeps track of used memory and can free memory when no more needed.
The tests are on graphs (underground map), family tree model and inference of relations, block puzzle where model has to learn how to meet different constraints
for the blocks, also finding shortest path 
