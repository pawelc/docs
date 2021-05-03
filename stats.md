#[The Metropolis–Hastings algorithm](https://arxiv.org/abs/1504.01896)
2016

Nice overview of the metrolopolis-hastings

#[Unconstrained Cholesky-based parametrization of correlation matrices]()
Overview of existing parameterizations of correlation matrices and propse and new one for lower triangular
matrix.

#[Analyzing within garage fuel economy gaps to support vehicle purchasing decisions - A copula-based modeling & forecasting approach]()
2018

Authors use Archimedean and elliptical copulas to model fuel efficiency data. The data is bivariate where each variable
is the ratio of reported fuel economy by the user to the official value for a car. The 2 variables constitute the 
household with 2 cars.  Authors use exploratory analysis to select correct distributions for marginals and copula
to model dependence. To choose among competing copulas they use some special function to compare goodness of fit
for copulas. They also use BIC and AIC criteria.

#[Dealing with the log of zero in regression models]()
2020

Authors review different methods that practitioners do to deal with non-positive inputs to the log transformation.
Authors show that adding small value to the non positive variable which is transformed by log is not always good idea.
The bias can be large and the best delta is not the smallest one. They modify semi-log regression so dependent variable
can contain 0 values by adding different values to each y.