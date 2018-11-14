# [Who is Afraid of Non-Convex Loss Functions?](http://videolectures.net/eml07_lecun_wia/)
2007 NIPS

Deep nets better than shallow. Mentions auto diff. Use more parameters than data but regularize a lot. Show
approximation to 2nd order optimization function. We are interested more in non-convex models because they give better results.
Sigmoids create better non-convex functions than for example RBFs. Sigmoids are like planks and it is difficult to build a
local minima that gets you stuck but each RBF create its local minima. In models with a lot of parameters (dimension of
loss function) it is easy to go around the hills to the better local minima and most of the local mimima are equivalent.