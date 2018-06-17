# Context-dependent pre-trained deep neural networks for large-vocabulary speech recognition
HMM - DNN model which combines neural network and HMM. Here NN provides observation probabilities.
# Unsupervised Neural Hidden Markov Models
Performing inference in HMM using NN. Replaces transition and emission matrices with output of the NN. The posteriors are still computed using Baum-Welch algorithm and they use 2 stage EM algorithm. The posteriors are used to weight gradients of the emission and transition which can be computed using NN. It would be interesting to compare it to our model.
Using CNN for observation distribution and RNN (LTSM) for transition model (effectively replacing 1 step Markov with dependence on all previous data) … but does it mean that data have to be fed sequentially and we cannot do stochastic optimization like we do.
Lessons learnt by authors could be important to our previous work because they found that when using NN in unsupervised model it is more important to use dropout and proper weight initialization compared to supervised models.
# A survey of hybrid ANN/HMM models for automatic speech recognition
Such models were developed already in 80s and beginning of 90s. Authors described different combinations if HMM and ANN and different ways of training. 
Separate (HMM and ANN params are learned separately) and global schemes are described (all parameters are learned together). 
Are hybrid systems worth considering a research topic or obsolete?
# A Connectionist Approach to Speech Recognition
Integration prior knowledge with learning: constraints on architecture, constraints that guide training, prior knowledge used to initialize free parameters. 
Different interesting ideas I haven’t heard before like: adding special outputs, smoothing the inputs to help with regularization. 
Partitioning of large system into smaller networks is described and training it globally and each part separately. 
Different architectures for processing sequences, more interesing: TDNN, RNN

# Dynamic Routing Between Capsules
Capsules are routed using Hebbian rules i.e. the connection between capsules on the consecutive layers are enforced if their outputs agree. Capsules are applied in the convolutional way. Don’t fully grasp the architecture so have to revisit it later.
# Transforming Auto-encoders
Network of capsules that learns to transform images. Capsule has recognition and generative units. It has also boolean gate to recognize if entity is on the picture. When applied in CNN way there is no weight sharing?
# Matrix capsules with EM routing
Reach source of references on how to improve CNNs by adding other types of invariances. Only read beginning.
# Adding Gradient Noise Improves Learning for Very Deep Networks
This uses annealed noise added to gradient. Maybe it can be compared to QA. Added gradient noise is similar to simulated annealing. 
Adding noise only helps with very deep architectures h > 5. For simple FFNN It looks that using good init and grad clipping achieves the same results as the same with 
grad noise. Grad noise helps when only simple init. But it is still useful when network is complex and we are not sure about proper init. But this is not true for more complex architectures where finding perfect init values is much harder.
# Quantum machine learning : what quantum computing means to datamining / Peter Wittek
Started but in the end was too hard to follow. Maybe later.

# Learning to Execute
LSTM that learns what is the output of the short program. The authors also test the LSTM model on copy task and notice that improvement can come from few heuristics 
like inversing the input or doubling it. Authors also notice that learning is only possible when applying their new curriculum learning (mixing simple curriculum  
and some random that can select also difficult examples like in Reinforcement learning neural Turing machines). It looks that combined curriculum  learning is mandatory 
and naive one can be worse the data distribution one.
Authors generate programs automatically that meet some criteria  regarding the difficulty. Overall this article seems to be weaker model than for example NTM.

# Mastering the game of Go with deep neural networks and tree search
Use value, policy networks together with Monte Carlo Tree Search and other techniques to achieve the best ever AI go player. First policy network is trained supervised on database of games. It is seed to policy network which is learnt using RL with self play. On these self play games value network is trained. 
# http://karpathy.github.io/2016/05/31/rl/
Karpaty also thinks that Policy gradient is much better than DQN. Policy gradient is sum A_i P(y_i|x_i) where y_i is sampled action, x_i is input and A_i is 
advantage i.e. the reward at the end. So we discourage all moves which in the end made us lost and encourage all movements that led to win. So here RL
is like supervised learning. When discounting all the rewards we can standadize them so we encourage half of the actions and discourage half of the actions.
This decreases variance of the policy gradient estimator. Policy gradient is score function gradient estimation. grad_x expectation_p(x) f(x) = 
expectation_p(x) ( f(x) * grad_x log p(x) )
# Deep Recurrent Q-Learning for Partially Observable MDPs
Using RNN on top of the DQN (CNN for modeling Q values). The paper shows experiments where RNN model helps with partial observability of the environment. They
do it but randomly clearing screens given to the model. In DQN paper authors had to give few previous frames from the video game to capute dynamics of the system,
RNN can model it implicitly.



# Bayesian Methods for Adaptive Models
Bayesian model inference of parameters and model comparison. Hasn't finished because was too difficult.

# Tutorial on Variational Autoencoders
Explained VAE and CVAE. very nice

# Variational inference for Monte Carlo objectives
Using multi sample stochastic lower bound for log likelihood. It can deal with discreate and continuous hidden variables.
