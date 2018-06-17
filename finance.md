# Adaptive systems for foreign exchange trading
Technical indicators alone are not enough to make strategy profitable. But if we use market maker's inside info of client flow and order book then we can
create profitable strategy. Authors check cointegration tests for flows and market moves and find the some particular flow like hedge fund and funds can conintegrate
nicely with price action. For sterling the most important flows were institutional and corporate.
# Intraday FX Trading: An Evolutionary Reinforcement Learning Approach in "IDEAL 2002 Intelligent Data Engineering and Automated Learning"
Trading agent that buys/sells/keeps neutral using RL with Q lerarning. The state of the system are signals from the technical indicators. This solution
has poor generalization. Using GA algorithm to select which indicators to use helps much out of sample performance.
# Agent Inspired Trading Using Recurrent Reinforcement Learning and LSTM Neural Networks
Value function approach to RL like Q-learning, dynamic programing, TD-learning can have problems in finance because of noise, and nostationary data and the policy 
can change dramarically when value function changes only a bit. Actor/critic is indermediete method when actor is learning policy and critic is learning value function.
The recurrent reinforcment learning  (type of direct reinforcment learning when we learn directly policy functuin) can be better apprach.
Authors setup sharp ratio and system tries to optimise it. Authors try various functions for functionally select decision (what position to have at time t) 
using RNN and LSTM. They use gradient descent to optimize across runs. They find learnt strategies profitable. 
They want to try in the future evolution type optimization and learn multi asset trading agent.
# FX trading via recurrent reinforcement learning
Authors optimize differential sharp ratio. They use simple FF NN with one and two layers. It is found that NN with 1 layer achieves better 
results because possibly works better with noise. It gives hint for the future that when using complex models we need good ways of regularization 
(like dropout or noise in the gradient). Authors also find that strategies are profitable when movement/spread ratios are bigger. 
Authors also train model in the online fashion also during test time.
# Making financial trading by recurrent reinforcement learning
Authors use RRL but not using Sharpe ration but ratio of sum of the positive and negative absolute returns. Authors in the same way compute the ratio 
using moving average. They try to minimize the maximum drawdown.
# Learning to trade via direct reinforcement
Authors argue that direct reinforcement i.e. method learning directly policy is better for problems when agent receives immediate estimates of
incremental performance compared to methods using value function like TD-Learning, Q-Learning. Agent to incorporate trading cost, market impact must have recurrent 
structure. Methods using value function is better when reward is deferred considerably in the future, DR is better when we have signal at each step.
# Machine Learning for Trading
Using Q learning (discrete actions and states) for trading function. Author derives what reward function should be used in case of concave utility function. 
Author models the market as random mean reverting process. He also models the market impact. 
# Equity market impact Almgren R, CThum, E Hauptmann and H Li 2005
Modeling temporary and persistent market impact by power functions

# Almgren R and N Chriss 1999 Value under liquidation
Value of the portfolio taking into consideration the cost of the liquidation

# Multiperiod Portfolio Selection and Bayesian Dynamic Models
Multiperiod portfolio optimization using HMM.
# Market Making via Reinforcement Learning
Using TD/Q learning to build market making agent. Authors discover that techniques like tile encoding, eligibility traces and reward function that
discourages holding of positions can improve performance of the model.

# FX Spot Trading and Risk Management from A Market Maker’s Perspective
Nice review of FX markets. Comprehensive way of simulating the environment of FX market maker.
Comparison of performance of market maker without hedging, with hedging and with VaR (using extreme value apprach)
