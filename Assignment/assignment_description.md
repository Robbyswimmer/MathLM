CS258/EE227
Intro to RL

Assigned: October 29, 2025
Due: November 10, 2025

Project 3 – Policy Gradients and Actor Critics
Problem Description
In the third programming assignment, you will implement the Vanilla Policy Gradient algorithm
covered in Lecture 10 and the Proximal Policy Optimization algorithm that we will cover next
week. You will have the chance to test both methods on a physically simulated character! The
Colab version of the assignment is available here. You can find the standalone version of the code
here.
Environment

We consider the HopperBulletEnv-v0 where a two-dimensional one-legged agent needs to hop for-
ward as fast as possible. The environment is defined as follows:

Observation: At any given time, the agent’s state consists of the height of the agent’s torso, the
global orientation of the center of mass (COM) in the forward direction, the angular velocity of
COM, the joint angles and linear speeds, roll and pitch information, and a Boolean that denotes
the state of the foot (1 if the foot is in contact with the ground, 0 otherwise).
Actions: There are three continuous actions that denote the torques applied to the three joints of
the agent (thigh, leg, and foot). The expected range of each action is in [-1,1].

Reward: The reward encourages the agent to stay alive and move forward, while penalizing exces-
sive torques.

Termination conditions: A rollout terminates if the agent fails to keep its COM elevated along
the forward direction, or fails to maintain its global orientation upright, or if the agent is acting in
the world for more than a predetermined time limit.

Vanilla Policy Gradient (5 pts)

Starting from the REINFORCE and Actor-Critic codes that we’ve provided, your task is to imple-
ment the Vanilla Policy Gradient algorithm (VPG) and train it on the aforementioned HopperBulletEnv-v0

environment. Your implementation should follow the pseudocode outlined in Lecture 10 and use the

value function as a baseline. In particular, you should collect a batch of (a fixed number of) trajec-
tories using the current policy. While collecting experiences, once a rollout is terminated you should

also update the corresponding advantage estimates of the visited s − a pairs needed for computing
the likelihood ratio policy gradient, as well as the target values of the visited states for training your
value function neural network. As soon as you have a batch of trajectories, you should update your
policy network using a single optimization step and re-fit your value function using a fixed number
of epochs. To re-fit your baseline, you must use reward-to-go returns as your value targets. If you
choose to experiment with other targets (see Implementation Details), briefly report differences but
base your final curves on rewards-to-go.

Implementation Details

1. As the action space is continuous, we will assume a stochastic policy represented as a multi-
variate Gaussian distribution with independent components (you can think of this as each of

the three action dimensions follows a Gaussian distribution). For simplicity, you should only
learn the state-dependent mean vector of the Gaussian (output of your policy network), while
treatinf the standard deviation as a tunable hyperparameter (using torch.nn.Parameter)
that is kept constant during training. To act, you need to sample from the Gaussian, i.e.
at ∼ N (μt
, σ2
I), μ ∈ R
3
, σ ∈ R
3
+. When performing evaluation rollouts, you should employ a
deterministic policy by directly using the output of the policy network (equivalent to having
a Gaussian with a 0 diagonal covariance matrix).
2. The expected range of each action is in [-1,1], so you should make sure that your network
outputs values inside that range, e.g., by using the tanh activation function.

3. While collecting experiences, once a rollout is terminated you should also update the corre-
sponding advantage estimates of the visited s − a pairs. To do so, you need to estimate the

Q-function of each s − a pair. As will be discussed in Lecture 11, there are different ways to
obtain Q-estimates, including using the full trajectory return, the reward-to-go (Monte Carlo
return), the TD return, n-step TD return, and the λ−return that gives rise to the Generalized
Advantage Estimate. In the report, you should briefly justify your return estimator choices
for VPG.
4. You also need to compute the target values of the visited states for training your value function.
Similarly to the Q-function estimate above, you can choose among a number of target options,
such as the reward-to-go (Monte Carlo returns), n-step bootstrapping target, and λ−returns
obtained by adding the value function estimate to the GAE. In your VPG implementation,
you should use the rewards-to-go as the targets for the value function. To train the value
function, you should solve a regression problem on the mean squared error between target
values and predicted value functions using gradient descent (e.g. full-batch gradient descent
for a fixed number of epochs, or mini-batch stochastic gradient descent for a fixed number of
epochs, etc.).

5. It may be easier to create a separate Buffer class to handle the collection of (partial) tra-
jectories as well as the computation of the advantage estimates and the target values when a

rollout terminates.
6. A trick which is known to usually boost empirical performance by lowering variance of the
gradient estimator is to center the computed advantages and normalize them to have a 0 mean
and a standard deviation of 1.
7. You may want to experiment with adding an entropy objective term to your policy loss
function as discussed in Lecture 11. This helps exploration in the early stages of training by
encouraging having evenly distributed actions.
Experiments and Report. Find any setting which results in the agent attaining an average
evaluation score of 650 or more with at most 3M training samples.
• Report the corresponding performance curve of your model as a function of total steps, with
error bars or shaded regions denoting the standard deviation;
• Briefly summarize the architecture of your policy and value networks;

• Document the hyperparameters that you used;
• Answer the following questions briefly:
- Which estimate for the Q-function did you use? Briefly justify your choice, providing
short observations about different estimators that you tried.
- Did advantage centering help?
- Did the entropy objective terms help?
- Did the batch size make an impact?
Keep in mind that training would be slower compared to the simpler environments used in previous
assignments. A good indication that the agent has started learning a walking policy is if the
evaluation performance is above 1,100. However, it is very likely that you may not reach such
performance with VPG.
Along with your report, submit your code and your trained policy network model and value
function network model. Additionally, run an evaluation rollout with your best policy and submit
the corresponding visualization saved as an animation png file using write_apng.
Proximal Policy Optimization (5 pts)
In the second part of the assignment, your task is to implement the proximal policy optimization
algorithm (PPO) using a clipped objective. Your implementation should follow the pseudocode
that we will discuss next week. Similar to VPG, you need to make certain choices about how to
compute the advantages for estimating the policy gradient and the target values for fitting the

value function. Here, you must use Generalized Advantage Estimation (GAE) for computing ad-
vantages. The same λ-returns you compute in GAE should be used as bootstrap targets for the

value function. Going from VPG to PPO involves only a few changes revolving around the defi-
nition of the policy loss function. The beauty of PPO is that you can re-use experience samples.

This means that you can perform multiple gradient steps using the same batch of collected ex-
periences, where at each step (epoch) you update the parameters of the policy network using the

clipped surrogate objective function. In practice, it is common at each epoch to sample without
replacement a minibatch from the entire batch. In this way, training becomes more diverse.
Experiments and Report. Using PPO you should be able to obtain much higher performance
compared to VPG for the same fixed number of training steps. Find any settings which result in
the agent attaining an average evaluation score of 1,200 or more with at most 3M training samples.
• Report the corresponding performance curve of your model as a function of total steps, with
error bars or shaded regions denoting the standard deviation;
• Compare the corresponding performance curve to VPG with same number of training steps,
discussing sample-efficiency and learning stability (variance across returns).
• Briefly summarize the architecture of your policy and value networks;
• Document the hyperparameters that you used, including the λ and γ values for GAE.
Along with your report, submit your code, your trained policy network model and value
function network model, and your corresponding animation png file.

Bonus Credits
• In PPO, to further add diversity to the samples used for training and reduce variance, you
may want to consider an A2C-like architecture, where you collect a batch of experiences by
gathering trajectories from multiple instances of the environment running either in parallel
(using torch.multiprocessing) or sequentially.
• We will rank all submitted VPG and PPO models. Submitted models that significantly stand
out in the leaderboard will receive extra credits.
Submission
Submit your code via Gradescope. As mentioned above, you should also submit your trained
models, as well as corresponding reports for both VPG and PPO. If you are using Colab, download
the python (.py) and notebook (.ipynb) files and submit them along with your trained models, png
files, and report.
Help
If you get stuck, please reach out during office hours. We also encourage