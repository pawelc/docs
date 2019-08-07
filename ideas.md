* NICE: Non-linear independent components estimation - 
The data used in experiments in graphics so we can inpating with our model as the results (actually authors are
doing this).
Here authors also use log-likelihood for multidimensional data so can use in my experiments.
Authors do inpainting using projected gradient ascent on the likelihood which can also be used for our model (iterative 
stochastic gradient procedure that maximizes log likelihood by updated missing values).

* Density estimation using Real NVP
Authors use trick to decrease computational and memory burden - factoring our half of the dimensions at regular intervals.
We could reuse possibly this technique.
They also use batch normalization, weight normalization and residual networks to improve propagation of the signal - we can
also try that out.
They do few data transformations I was not aware of to remove boundary effects (image data).

* Variational lossy autoencoder]
A lot of more heavy examples for density estimation

# Masked autoregressive flow for density estimation
The Figure 1 shows how data distribution can affect estimator which uses autoregressive approach to model it. Can do 
similar experiment with our models.
Our MONDE using MADEs is kind of MAF but with more flexible conditional because we are not using parametrized distribution.
In experiments authors use early stopping of 30 epochs and the same UCI data sets. They use batch normalization and
regularization. Authors use also paired t-test to compare the models.
!!! Actually I can implement flow by stacking one layered MONDEs (using MADEs) on top of each other ?? Then by modeling
density at each level we could shuffle dimensions ? 
They describe batch normalization maybe we should implement it ?

#Efficient backprop
Authors notice that setting labels in classification close to saturation points (by using large weights) can push or 
examples to these values. Therefore the model will not represent uncertainty in the prediction.
Because our model is learned differently it can more naturally represent uncertainty in the output. 
Can we modify the output somehow so the outliers in the model does not cause problems during training?
Check if the inputs in each layer fall into linear parts of the non-linearity so we don't saturate at the beginning.
Too small weight can cause small gradients.
Can we measure Hessian matrix of E wrt parameters to check for example condition number.
Authors present tricks in computing Hessian matrix efficiently in NN (maybe I should do it for our loss function)

# Bounded activation functions for enhanced training stability of deep neural networks on visual pattern recognition problems
Use modified tanh at intermediate layers and capped modified tanh at the output so the gradient is passed
Maybe try different non-linearties referenced by this work.
Use bounded ReLU for F and bounded leaky ReLU for intermediete layyers. 

# Understanding the difficulty of training deep feedforward neural networks
Try softsign activation function, similar to the hyperbolic tangent, but its tails are quadratic polynomials rather
than exponentials. This paper has a lot of ideas to check if we want to understand why training of MONDE is slow.
Monitoring activations and gradients across layers and training iterations is a powerful investigative tool for
understanding training difficulties in deep nets
Keeping the layer-to-layer transformations such that both activations and gradients flow well (i.e. with a Jacobian
around 1) appears helpful, and allows to eliminate a good part of the discrepancy between purely
supervised deep networks and ones pre-trained with unsupervised learning.

#Learning bias in neural networks and an approach to controlling its effect in monotonic classification
Here authors describe bias introduced in training NN (like classification surface which is close to last training
point). Can we also research such bias in our model. Maybe like in the experiment performed in this paper we could
check if our model is used for classification what is decision boundary i.e. is it biased.

#Problems
try bigger learning rate
batch norm every second layer
put batch norm after non linearity
try tanh in intermediente layers
try different way of normalization
use different optimizers (simple momentum, new ones)
adding linear term to tanx to avoid flat spots.

pdf is dF(y|x)/ dy, where F(y|x) is CDF

For small probabilities the model fails because gradient becomes inf.
 d log(pdf) / d pdf is very big negative because = 1/pdf. This gradient can explode ov

Looks like using (1+tanh(x))*0.5 is more stable then sigm

If we apply autoencoder to data to decorelate the dimensions and then apply the copula made, would it achieve better results?

It looks that increasing batch size removes the problem of NaNs in later stages. Maybe we should increase batch size
after some condition.

Check if making dependency on the nodes which do not have monotonic constraint improve performance

#MADE: Masked Autoencoder for Distribution Estimation
Maybe MADE is worse than out model because the first component of the AR decomposition is not encoded my the NN ?
Try adding direct connections between input and output like in MADE.