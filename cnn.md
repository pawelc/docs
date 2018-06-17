# Temporal Convolutional Networks: A Unified Approach to Action Segmentation
Using one model for temporal segmentation of actions in the video data (TCN) instead of complex models composed of CNN and RNN. According to authors this is easier to train and gives better results.
# Deep Learning Book, chapter about CNN
TDNN (Time Delay Neural Networks) are 1D convolutional NN for TS introduced by Land and Hinton in 1988 (The development of the time-delay neural network architecture for speech recognition.)
# Convolutional networks for images, speech, and time series
Early description of conv nets. TDNN mentioned as CNN which share weights across one temporal dimension. 
Using probabilistic method ass HMM on the output of the replicated CNN (to handle variable size inputs like sentences) to decode the objects on which given CNN centers on (word, letter). Could we apply our VI HMM on this CNN.
# Going deeper with convolutions
Making CNN more efficient and steel improving results compared to the previous bigger CNNs. This could be good for my commercial domain.
# Network in network
Authors use MLP instead of GLM as the convolution filters. At the last layer instead of the fully connected NN authors use global average pooling which creates feature maps for each category. This mean that for each category we create feature map using convolutions and then each map is averaged before all going into softmax layer.I think this idea can also be applied for time series analysis for pattern detection in sequences.
One by One convolution that can be used for dimensionality reduction.
# Improving neural networks by preventing co-adaptation of feature detectors
Original paper about the dropout. Interesting idea about limiting the L2 norm of the vector of the incoming weights to each unit which allows to start with big learning rate (when updating the weights the norm is too large we simply rescale the weights). Compares dropout to Bayesian NN.
Interestingly the authors mention that CNN filters are good for datasets with roughly stationary statistics because the same filter is applied across all examples in the dataset. 
Something new for me here is response normalization units where activations of the neurons at the same positions from different banks are normalized.
Nice discussion about using linear rectifiers units. We have to initialize weights in a way so the input to the linear rectifier are mostly positive because otherwise learning doesn’t take place. It helps to set constant to some positive value (authors use 1) to get learning of the ground. For weight initialization they use normal with 0 and some variance big enough to get positive inputs (chosen experimentally).
