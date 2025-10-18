# On Tuning Learning Rate and Batch Size

There is no concrete fine-tuning technique for learning rate and batch size hyperparameters; however, we can rely on thumb rules and heuristics.

Batch size ($B$) and learning rate ($\eta$) are heuristically co-dependent via the following formula ([source](https://arxiv.org/abs/1706.02677)):

$$
\eta' = \eta \times \frac{B'}{B}
$$

This rule suggests that if you increase the batch size, you should proportionally (linear here) increase the learning rate to maintain the same update dynamics; why this can work? a larger batch size decreases the number of updates per epoch, but it provides a more accurate gradient estimation, so you can afford to take bigger steps; another intuition is that a small batch size introduces more noise on the gradient estimation, therefore, needing a smaller learning rate to stabilize, while the vice versa.

Let's take the batch size as the independent variable and the learning rate as the dependent one. Choose the maximum power-of-two batch size that would yield at least $U^* = 10$ updates per epoch on average (over nodes); then, commit a Leslie N. Smith learning rate range test (see [here](https://arxiv.org/abs/1506.01186)) to find the learning rate that has the steepest gradient.

A take on dropping the last incomplete min-batch (setting `drop_last` in `DataLoader`): an incomplete mini-batch might introduce noisy gradients, potentially harming convergence; we assume that setting it can be beneficial for our use case, because, we are suiting the learning rate for a constant batch size, the data is shuffled at each epoch, usually there are multiple epochs per round, and usually there are multiple rounds per run.