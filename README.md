## Variational Autoencoder Latent Rebalancing with Determinantal Point Process Prior

We proposed to use Determinantal Point Process as a diversity encouraging prior for latent variable models, here Variational Auto-encoderto in particular, to alleviate imbalance learning problem.

### Experiments ###

The training examples include 5000 MNIST ’0’, ’1’ handwritten digits data, where digit ‘1’ is the minor class. In this demo, we use an imbalance ratio 10 to 1.

Run `unbalance_vae_generator.py` to generate synthetic hand-written digits using standard VAE.

Run `unbalance_dppvae_generator.py` to generate synthetic hand-written digits using the proposed VAE with Determinantal Point Process as the prior.

A comparison of results are shown below, where the proposed DPP-VAE generated more minor class '1':

Synthetic data with standard VAE | Synthetic data with DPP VAE
:-------------------------:|:-------------------------:
<img src="https://github.com/tianc01/dpp-vae/blob/master/results/ordered_vae_10to1.jpg" width="400" />  |  <img src="https://github.com/tianc01/dpp-vae/blob/master/results/ordered_dppvae_10to1.jpg" width="400" />
