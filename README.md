# real_time_style_transfer
Real-Time Style Transfer : Improvement on the Gatys approach, which produces high-quality images, but is slow since inference requires solving an optimization problem. Justin Johnson et al. came up with a variant of style transfer that was much faster and produced similar results to the original implementation. Their approach involves training a CNN in a supervised manner, using perceptual loss function to measure the difference between output and ground-truth images.

https://arxiv.org/abs/1603.08155
