# Learning context-free grammars: Capabilities and limitations of a recurrent neural network with an external stack memory
Im this paper from 1992 authors already used RNN with external memory (stack in this case). RNN at each time step receives previous state,
input and value from the stack. The output is the output, state and the categorical action which is used to decide if we should push, pop, or no-op
from the stack. It looks they also used curriculum learning. They called it Neural Network Pushdown Automaton.

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

# Neural Random-Access Machines
There is a controller, modules (operations), registers. Controller orchestrates operations over registers using operations and the results are written back to the 
registers.
The circuits are like mini algorithms (subroutine) that are executed step by step. More difficult problems are hard to learn possibly because of weak optimization. 

# Reinforcement learning neural Turing machines 
This model uses hard attention to address the memory. Most external interfaces are discrete so very useful. It combines backpropagation with reinforce.
Important because continuous-discrete learning. This model is very difficult to train for more complex problems. 
Curriculum learning is very important to be able to learn. It is also important to decrease the variance of the gradient estimator.
Reinforce i.e. policy gradient is used to construct the cost function but it is simply expectation of the actions over the cumulative rewards.
They had to hack controller a bit to be able to solve the tasks. They call it direct access controller because it can for example copy directly input to memory
modulated only by variable from the controller.
Can read further how they make method more stable and how to implement it.

# Memory networks
Very generic architecture for memory enhanced models. Components I (input),G(generalization i.e. memory access/write), O (output feature from memory given input), 
R (response). The memory stores all sequences (if words are given there is a model to discover sequences). 
Then memory is compared to each sequence in order in the test set . Authors use hashing of the memory to improve performance. The model is applied to QA tasks.

# Weakly Supervised Memory Networks
The same application as Memory Networs but doesn't hint which intermediate sentences help with answering question hence weak (no supervision of the supporting facts). 
Like Differentiable Neural Computer they try to keep temporal information for memory access.
They try to fight the local minima by first running the model with all non-linear operations removed and later when 
there is no improvement they enable non-linearity (they call it linear start).

# End-To-End Memory Networks
The authors propose differentiable architecture for question answering. All sentences are encoded into internal representation, the output is encoded
separately. Based on internal representation we create weight vector over the output. Question is also embedded and this embedding is summed with output.
This computation can be chained like in RNN but input at each level is output from the previous layer and all the input sentences. Authors
try different kinds of parameter tyeing between layers. The layers represent number of think times before network computes the answer.

# Recent Advances in Recurrent Neural Networks
Nice review of RNN from the historic models up to the state of the art like memory networks.

# GENERATIVE TEMPORAL MODELS WITH MEMORY

This paper combines stochastic variational inference with memory-augmented recurrent neural networks. 
The authors test 4 variants of their models against the Variational Recurrent Neural Network on 7 artificial tasks requiring long term memory. 
The reported log-likelihood lower bound is not obviously improved by the new models on all tasks but is slightly better on tasks requiring high capacity memory.

# Variational Memory Addressing in Generative Models
Generative model enhanced with memory trained using Using multi sample stochastic lower bound for log likelihood. The memory
is addressed by discrete random variable which also have approximate posterior. So it uses hard attention. Tehy mention that their addressing
would be plugin replacement for soft addressing used in "GENERATIVE TEMPORAL MODELS WITH MEMORY" 

There was a workshop about memomy https://nips2015.sched.com/event/4G4h/reasoning-attention-memory-ram-workshop