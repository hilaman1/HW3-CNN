r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_generation_params():
    start_seq = ""
    temperature = .0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "Prince of Denmark stood"
    temperature = 0.5
    # ========================
    return start_seq, temperature


part1_q1 = r"""
There are some reasons:

1. processing the entire text as a single input sequence can be computationally expensive
2. training on smaller sequences prevents vanishing or exploding gradient problems that can occur when processing long sequences.
3. it encourages the model to learn higher-level patterns and generate creative outputs

"""

part1_q2 = r"""

The hidden states of the model retains information from previously generated text, 
it allows the model to exhibit memory longer than the individual sequence length.

"""

part1_q3 = r"""

In RNN the order of the data is important, each batch relies on the hidden state and information learned from the previous batch.
Also the gradients are propagated through time steps within each batch


"""

part1_q4 = r"""

1. The temperature controls the level of randomness in the generated text, for lower values
it decreases the randomness, hence we get more appropriate words.

2. When the temperature is very high, the softmax emphasises the lower score
which makes the output distribution more uniformly, and then we can more surprised of the network choices.

3. When the temperature is very low, the softmax flattens the distribution,
which makes only the high values to be relevant and therefore the generated word.

"""
# ==============


# ==============
# Part 2 answers


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0,
        h_dim=0, z_dim=0, x_sigma2=0.0,
        learn_rate=0.0, betas=(0.0, 0.0),
    )
    # Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 10  # 10
    hypers['h_dim'] = 256
    hypers['z_dim'] = 128
    hypers['x_sigma2'] = 0.5  # 0.5
    hypers['learn_rate'] = 0.001  # 0.001
    hypers['betas'] = (0.9, 0.999)
    # ========================
    return hypers


part2_q1 = r"""
The hyperparameter x_sigma2 originates in the parametric likelihood distribution: $p_{{\beta}}({X} | {z}) = \mathcal{N}(\Psi_{\beta}({z}) , \sigma^2 {I})$
And so it matches the variance of that distribution.
We can see from the expression of the loss, that the smaller x_sigma2, the bigger the contribution of the data_loss (which coressponds to the reconstruction of the image) to the total loss.
As a result, this parameter will effectively determine how much importance minimizing the data_loss will have compared to minimizing kldiv_loss (the KL divergence regularization term of the loss).
Overall, if x_sigma2 is relatively high, data_loss (the reconstruction loss) takes precedence, and if x_sigma2 is low, kldiv_loss (regularization loss term) takes precedence.
"""

part2_q2 = r"""
1. The purpose of the reconstruction loss is to converge towards successful reconstruction of images. This means that the reconstructed images outputted from
the decoder using the sampled latent variables will be as similar as possible to the original images.
The purpose of the KL divergence loss is to act as a regularization term by increasing the in accordance to the distance between the posterior distribution $q_{\alpha}({Z}|{x})$, approximated 
by the encoder in order to sample latent variables, and the latent space prior distribution $p(\bb{Z})$. The farther apart the 2 distributions are, the larger the penalty will be.

2. In our case, the latent-space prior distribution $p({Z}) = \mathcal{N}({0},{I})$ is fixed. 
The KL loss term is affecting the posterior distribution $q_{\alpha}({Z}|{x})$ (that is approximated by the encoder) by trying to make it close as possible to the latent-space prior 
distribution $p({Z})$. 
In our case, this means that the latent-space data sampled will be more in line with the standard normal distribution $\mathcal{N}({0},{I})$, which will be centered around 0. In particular, it means
that two similar images would be encoded into something that is also relatively similar.

3. The benefit of the KL loss term is to have a regularization term that forces the encoder to make more meaningful encodings that will generalize better and thus give better results on new data.
In particular, we would expect that similar images will have similar representations.
"""

# ==============


# ==============
# Part 3 answers


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0, z_dim=0,
        data_label=0, label_noise=0.0,
        discriminator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
        ),
        generator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
