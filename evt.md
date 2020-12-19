#[Using neural networks and extreme value distributions to model electricity pool prices: Evidence from the Australian National Electricity Market 1998-2013]()
2014

Predicting electricity spot prices can be tricky because the process combines smooth part and spikes. Authors propose
to model smooth path with the NN and to model spike behaviour with the Generalised Pareto Distribution.
It looks that electricity prices are excellent data for EVM because of large number of large spikes during normally
smooth price process. It is much more spiky than for example FX market.
Authors use very small neural network with 11-5-1 architecture, with skip connections and weight decay. They also transform
the prices via the log function. It presents opportunity if we can devise the model that estimate the entire process
and does not model 2 regimes with 2 different approaches separately.