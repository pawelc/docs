# [Machine Learning as an Experimental Science]()
1988 (2006 revised)

Authors pointed to three components of the standard testing procedure which are problematic: the mea- sures used, 
the reliance on null hypothesis statistical testing and the use of benchmark data sets.

# [Parameterizing correlations: A geometric interpretation]()
2007

How to parametrize the correlation matrix using full or reduced rank unconstrained parametrization. The parametrization
that enforces the correlation matrix to be PSD. Albeit got slightly worse results than our method.

# [Deep Learning: Autodiff, Parameter Tying and Backprop Through Time](http://web4.cs.ucl.ac.uk/staff/D.Barber/publications/ParameterTying.pdf)
2015.02.09

Overview of different ways of differentiation: numerical, symbolic and autodiff. Report present how to efficiently compute
gradient with respect to tied parameters like in RNN. Examples showing autodiff and parameter tying.

# [2016 - A note on the evaluation of generative models]()
They quality of the model in different tests depends on the objective being optimized (KLD, MMD, JSD). The JSD would
be optimal when we want to sample plausible images from the model. But for compression we would like to use KLD when
each possible vector should have sufficient probability.

# [Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767)
2018.02.05

Good survey of AD methods. How they refer to other methods like backpropagation. History. Application in different areas
of machine learning. Summarize in my thesis.

# [From automatic differentiation to message passing](https://www.youtube.com/watch?v=NkJNcEed2NU&feature=youtu.be)
2019.06.19

Presentation about machine learning language that could be build on top of massage passing. Presenter shows issues with applying
blindly algorithmic differentiation which does not support approximation for example. Message passing does.