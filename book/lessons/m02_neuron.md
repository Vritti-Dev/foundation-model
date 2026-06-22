# Module 2. A Neuron and a Gradient Step

**Runs in your browser.** Nothing to install.

## Why this matters

Every neural network is built from one thing repeated millions of times: a **neuron**. Once you understand exactly what one neuron does and how it learns, the whole field is much smaller than it looks.

In this lesson you build one neuron, by hand, on paper essentially. You make it produce an output. You compute how wrong that output is. You work out, with calculus, how to nudge its weights so the next output is less wrong. You take that one nudge. You watch the error go down. That is the entire learning algorithm of every model in the field, including ChatGPT and Claude. Everything else is engineering.

## What is actually happening

A neuron takes a list of inputs `x = (x1, x2, x3)`, has its own weights `w = (w1, w2, w3)` and a bias `b`, and computes:

```
z = w1*x1 + w2*x2 + w3*x3 + b
y = activation(z)
```

The activation function adds a non-linear bend. Without it, stacking many neurons is mathematically the same as having one neuron, so they would all collapse into a useless linear model. Common activations are ReLU (`max(0, z)`), tanh, and GELU.

To **learn**, you need a notion of "wrong". That is the **loss**: a single number that gets larger when the prediction is worse. The simplest is mean squared error: `L = (y - target)^2`.

Now the trick. Calculus gives you the **gradient**: the derivative of the loss with respect to each weight. The gradient `dL/dw1` answers the question: if I nudge `w1` up by a tiny amount, how much does the loss change?

* If `dL/dw1` is positive, increasing `w1` makes things worse. Decrease it.
* If `dL/dw1` is negative, increasing `w1` makes things better. Increase it.
* If `dL/dw1` is zero, `w1` is already at a sweet spot.

The update rule is: `w1 := w1 - learning_rate * dL/dw1`. Take a small step in the direction that reduces the loss. Do it again. Do it ten thousand times. Your model gets good.

In this lesson you derive `dL/dw` by hand with the chain rule, then verify it numerically with a tiny finite-difference test: nudge `w` by 0.00001, see how much the loss changed, compare to your derivative. They should agree to many decimal places.

## A moment of curiosity

What happens if the learning rate is too big? The update overshoots the minimum and you bounce around or blow up to infinity.

What if it is too small? You crawl. Your loss goes down so slowly that you never converge in any reasonable time.

This trade-off (the **learning rate**) is one of the most important hyperparameters in deep learning. Real training uses schedules: start with a larger rate, gradually decrease it, sometimes warmup at the very beginning. You will see this in Module 8.

## Common confusions

* **Gradient is a number per weight, not a single number.** Each weight has its own partial derivative.
* **The minus sign in the update is essential.** You move *against* the gradient because the gradient points uphill (toward larger loss).
* **A neuron's bias is a weight too.** It learns the same way.

## Going deeper

* 3Blue1Brown, **Gradient descent, how neural networks learn**. Visual proof that gradient descent works, in 20 minutes. <https://www.youtube.com/watch?v=IHZwWFHWa-w>
* Andrej Karpathy, **The spelled-out intro to neural networks and backpropagation** (Zero to Hero #1). The video this module is most closely modelled on. <https://www.youtube.com/watch?v=VMj-3S1tku0>
* Michael Nielsen, **Neural Networks and Deep Learning**, Chapter 1. Free online textbook, exceptionally clear on these foundations. <http://neuralnetworksanddeeplearning.com/chap1.html>

```{jupyterlite} ../../notebooks/m02_neuron.ipynb
:new_tab: False
:width: 100%
:height: 600px
```
