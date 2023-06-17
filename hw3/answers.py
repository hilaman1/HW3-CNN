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
        h_dim=0, z_dim=0, x_sigma2=0,
        learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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
