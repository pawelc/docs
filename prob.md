# Copulas and dependence 
Very nice introduction to copulas and representing dependence using this concept.
Copula: https://github.com/qrmtutorial/qrm/blob/master/slides/qrm_07.pdf and https://www.youtube.com/watch?v=XzfvUaL45cg

#[Approximating the covariance matrix of GMMs with low-rank perturbations]()
Authors compare different ways of approximating covariance: full, diagonal and diagonal plus perturbation.

#[A note on pseudolikelihood constructed from marginal densities]()
2004
TODO
Not completed because to difficult.

#[Tail Dependence]()
2005

Tail dependence explained with Copula. Non parametric estimators given. VaR examples given. Using t-student for marginals
and and copula, when different degrees of freedom the result is not elliptical. We cannot use gaussian copula
for tail dependence (we can plot empirical copula do check if data has tail dependence).


#[An overview of composite likelihood methods]()
2011
Thorough overview of composite likelihood methods/pseudo likelihood where we define the missspecified model as product 
of likelihood functions on subsets of variables. The rationale is often the computational complexity. I skimmed over it because
quite extensive.

# A Tutorial on Fisher Information

“The (unit) Fisher information is a measure for the amount of information that is expected within the prototypical trial X about the parameter of interest θ. 
It is defined as the variance of the so-called score function, i.e., the derivative 20 of the log-likelihood function with respect to the parameter”

#[Copulas in machine learning]()

Tree structured graphical models which has each bivariate dependence encoded as copula. If more
general  structure is needed we can use mixture over such trees.

#[Behaviour of multivariate tail dependence coefficients]()

Description of the tail dependence of the symetric multivariate copulas. Authors ask the question: how will behave 
a random variable of interest if some other variables are larger than a given value; or more generally, 
how will behave a given variable if at least one of other variables is larger than a certain given value.