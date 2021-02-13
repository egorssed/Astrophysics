# VAE_on_HST_galaxies

The trend on all-sky surveys will provide the scientific community with thousands of objects, though it makes scientists look for ways of carrying out 
and validating the analysis of the images obtained with these wide-field surveys. 
One of the problems is that the typical representation of a galaxy relies on a simple parametric fit profile, therefore does not completely describe it. 
The other problem is that the use of Markov Chain Monte Carlo sampling for this parametric fitting takes a lot of time. 
Luckily, machine learning provides a solution to both problems. In this work we present a Deep Neural Network called Variational autoencoder
that can encode a galaxy image as a set of 64 parameters and decode the image back from this representation with minor structural losses, therefore solving the 
problem of usually parametric fits simplicity. Furthermore, the VAE can encode and decode back thousands of images in seconds, making the parametric fitting much 
faster than MCMC sampling. We used the HST COSMOS dataset to train the VAE. The VAE itself was based on convolutional layers. Experiments with loss function lead 
to a conclusion that undamped Kullback-Leibler loss over regularizes the latent space resulting in random VAE output. Constrained Kullback-Leibler loss (Flow-VAE) 
enables the model to have determined output, however, Flow-VAE has poor reconstruction quality along with latent space inconsistent with the prior. Multiplicative 
decrease of Kullback-Leibler loss strength by beta=0.01 (Beta-VAE) leads to good reconstruction quality and plausible latent space. 
A Framework for convenient VAE architecture experiments was developed and published on Github.
