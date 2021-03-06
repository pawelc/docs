# http://www.deeplearningpatterns.com/doku.php?id=memory 
Website for the future book about DL. It contain references and summary of papers for each aspect of DL including memory enhanced models.

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

# [Key-Value Memory Networks for Directly Reading Documents](https://arxiv.org/abs/1606.03126)
Authors propose new architecture that enables learning QA system directly from the documents. Other possibilities like building Knowledge Bases
are problematic because we need to create algorithms to create automatically KB which are often too rigid or build them manually. 
It is based on end to end memory network architecture. The main difference is that array used for addressing is composed of keys and second array used for output is composed of 
values. Authors point out importance of the hashing for model performance in terms of speed of data retrieval. 
The final output is multiplied by each candidate output and the best is chosed as an answer.
Authors test different ways of of encoding keys/values and compare the methods of directly reading documents or KB. 

# [Associative Long Short-Term Memory](https://arxiv.org/abs/1602.03032)
Authors improve LSTM models by adding associative memory and not increasing the number of parameters.
They use ideas from the Holographic Reduced Representations which uses fixed sized array to store
associative array. It uses complex algebra to store and retrieve memories (multiplication of complex vectors and
inverse of the key times the memory to retrieve value). Authors reduce the retrieval noise by
soring multiple copies and when retrieving taking an average.
Compared to NTM it is not necessary to look for free locations.
The Error is controled by items stored in the memory and number of copies.
The update equations are similar to regular LSTM but we also have additional key vectors and also function that bounds
comples vector by constraining them to be with maximum modulus of 1.
Copy task as ultimate basic task to check. If it is failed that model is very poor.
They use XML prediction task to check if model can forget correctly.
The other task is sequence of "set variable to value" instructions followed by query command to recall asigned value.
Authors compare their model to NTM. NTM can show very unstable learning compared to the Associative LSTM. But
NTM can show better generalization on longer sequences in algorithmic tasks.

# [Grid Long Short-Term Memory](https://arxiv.org/abs/1507.01526)
Grid LSTMs are different than multidmiensional LSTM or stacked LSTM. Here the memory and hidden vectors are passed across all dimensions. The input to each dimension is
concatenation of the all hidden vectors from the previous blocks. It doesn't suffer from the exponential eplosion as multidimensional LSTM. We can have any number of dimensions.
In a task to sum up 2 integers network while predicting is not given previous predicted digit as input so it makes the task more chalanging becuase it has to remember previous
digits. The tied grid 2-LSTM achieves much better results than stack LSTM.
This paper shows many tests to check performance of sequence model.

# [Ask Me Anything: Dynamic Memory Networks for Natural Language Processing](https://arxiv.org/abs/1506.07285)
Authors propose model composed of modules: input module, question module, episodic memory module, answer module.
Input module encodes the sentence(s) using RNN using output of hidden states per word if sentence and hidden per sentence if multiple
sentences. The question module also uses RNN but return only the last hidden state. Both input and question modules
use the same embeding matrix. Episodic memory module is composed of the NN which takes input special features vector representing
similarities between question, memory and input representation. The output of this network acts as gating mechanism for hidden
states of the GRU network which takes input codes and memory as its input. The memory stops iterating when it chooses 
terminal symbol. The answer module is also GRU network which at each time step takes input as question, previous hidden state and
previous predicted output. The answer module can be run at the end of memory processing or at each iteration of memory pass.
They had inspiration from the neuroscience. They perform experiments on QA, POS tagging and sentiment analysis. All can be
done by this model.

# [Inferring Algorithmic Patterns with Stack-Augmented Recurrent Nets](https://arxiv.org/abs/1503.01007)
Authors enhanced RNN with memory (stack or list) to show that it helps solve some simple tasks not tackled by previous
architectures. They generate sequences with some algorithms (context free and context sensitive grammars) so models can try to learn it. 
Because sequences are generated algorithmically some of the symbols can be predicted deterministically. 
Model presented has unbounded memory which is different for example from NTM. Authors also notice the in the NTM
paper authors didn't rationalize the usage of such complex model to the problems they studied.
The first model is almost the same as Pushdown Automaton from 90s. The difference is that we don't use values saying how many elements have
to be poped or pushed (stack in the old model contained value and value indicating the weight of the value).
Authors also show parametrization of the model with linked-list which has the head which movement, read and write operations are parametrized by NN.
Authors also find that SGD is not enough for complex tasks and they need to use search algorithms. They also use discretization of conntinuous operations
on stacks during test time to improve results which they call rounding. They hypothetize that learning more complex algorithms possibly would
require combining continuous and dicrete optimization algorithms.
On simple examples authors set recurrent (in RNN) matrix to 0 to isolate the effects of the memory.
Based on memorization and binary addition tasks authors conclude the stack enhanced RNN generalized better than list enhanced RNNs.
Both models are generalizing better (train test with up 20 length sequences and test set with up 60 length) than regular RNN and LSTM.
For the addition task the same way of checking generalization. Authors visualise internals by showing how stacks are used to remeber summands and do
operations like carry. Albeit the proposed model is worse than LSTM and SRCN in language modeling task.