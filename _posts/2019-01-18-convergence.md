---
title: "Convergence of Linear Models"
date: 2019-01-18
tags: [optimization-gradient descent-normal equation-maximum likelihood estimation-regression-classification]
header:
  image: "/images/convergence/header.png"
  caption: "Photo Credit: libretexts.org"
excerpt: "Learning objectives and performance for linear models: a look under the hood"
mathjax: "true"
---

### Introduction

This post will cover the techniques and concepts used for training linear models. While most of the information presented here has already been covered elsewhere with great detail, I hope to present it in such a way that is relatable to anyone interested in data science. In doing so, I will also test the depth of my own knowledge and improve my written communication as a result. For additional information, a list of references will be provided at the end of this post.

### Background

Supervised learning is a classic domain of statistics which involves a set of features and a response variable. A supervised learning model seeks to capture the full relationship between the features and the response for prediction/inference. If this relationship exists, then there must be some function which relates the systematic information of the features to the response. The modeling process amounts to approximating this function and evaluating the results. The general form of the function can be written in the following way:

$$\begin{equation}
Y = f(X) + \epsilon
\end{equation}$$

In this equation, $$f$$ is our model and $$\epsilon$$ represents the variation of $$y$$ not captured by our model. Linear models are named as such because they associate a linear shape for $$f$$. Here is the general form of a linear model:

$$\begin{equation}
f(X) = \beta_{0} + \beta_{1}X_{1} + \beta_{2}X_{2} +...+\beta_{n}X_{n}
\end{equation}$$

$$\beta_{0}$$ is a constant term representing the bias for model training. Every other parameter has an associated feature ($$X_{1:n}$$) representing the measured effect of $$X_{i}$$ on $$Y$$. Given some data, we can use this model to make predictions as a weighted sum of the input features. These predictions are evaluated using a performance measure, typically RMSE (Root Mean Squared Error). Here, RMSE is represented using matrix notation where $$\theta$$ are the model parameters and $$X$$ is our feature set.

$$\begin{equation}
RMSE = ((1/n)\sum_{i=1}^{n}[y_{i}-\theta^{T}X_{i}]^{2})^{1/2}
\end{equation}$$

For each instance, we take the squared difference between our prediction and the label, add them up, and normalize the result. It follows that we need to find the best combination of parameters which minimize model errors ($$y_{i}$$ - f(X)). There are two approaches to accomplish this task.

### The Closed Form Solution

In this section, I will go through the steps to find model parameters for a single feature. Application of multiple features is conceptually the same process but is done using matrix notation for convenience. Either case provides a direct solution of optimal parameters instead of an iterated approach (more on this later). Lets get started!

Our goal is to minimize our performance measure (RMSE) with respect to the model parameters ($$\beta_{0}$$ and $$\beta_{1}$$). We can drop the square root and the (1/m) constant since they have no effect on the end result. This simplifies our function to the squared-sum of model errors (SE):

$$\begin{equation}
SE = \sum_{i=1}^{n}(y_{i}-\beta_{0}-\beta_{1}x_{i})^{2}
\end{equation}$$

First, we'll take the partial derivative of SE for $$\beta_{0}$$.

$$\begin{equation}
\beta_{0}^{'} = -2\sum_{i=1}^{n}(y_{i}-\beta_{0}-\beta_{1}x_{i}) = 0 \\
= \sum_{i=1}^{n}(y_{i}-\beta_{0}-\beta_{1}x_{i}) = 0 \\
\sum_{i=1}^{n}\beta_{0} = \sum_{i=1}^{n}y_{i} -\beta_{1}\sum_{i=1}^{n}x_{i} \\
n\beta_{0} = \sum_{i=1}^{n}y_{i} - \beta_{1}\sum_{i=1}^{n}x_{i} \\
\beta_{0} = (1/n)\sum_{i=1}^{n}y_{i} - \beta_{1}(1/n)\sum_{i=1}^{n}x_{i} \\
\beta_{0} = \bar{y} - \beta_{1}\bar{X}
\end{equation}$$

After dropping the -2 constant, we distribute the summation operator and rewrite in terms of $$\beta_{0}$$. $$\beta_{0}$$ is the model intercept, so a summation over a constant also produces a constant ($$n\beta_{0}$$).

We can now use this result with the partial derivative of SE for $$\beta_{1}$$ to derive the direct solution.

$$\begin{equation}
\beta_{1}^{'} = -2\sum_{i=1}^{n}(y_{i}-\beta_{0}-\beta_{1}x_{i})(x_{i}) = 0 \\
= \sum_{i=1}^{n}x_{i}(y_{i}-\beta_{0}-\beta_{1}x_{i}) = 0 \\
=\sum_{i=1}^{n}x_{i}y_{i}-\beta_{0}\sum_{i=1}^{n}x_{i}-\beta_{1}\sum_{i=1}^{n}(x_{i})^{2} = 0
\end{equation}$$

The result of the partial derivative for $$\beta_{1}$$ is still in terms of $$\beta_{0}$$, so we need to do a bit more work. We can use the partial for $$\beta_{0}$$ as a function of $$\beta_{1}$$ to substitute and solve for $$\beta_{1}$$ using some algebra. The end result is a little messy but is a direct solution nonetheless!

$$\begin{equation}
\beta_{1} = \frac{\sum_{i=1}^{n}x_{i}y_{i}-(1/n)\sum_{i=1}^{n}x_{i}\sum_{i=1}^{n}y_{i}}{\sum_{i=1}^{n}(x_{i})^{2} - (1/n)(\sum_{i=1}^{n}x_{i})^{2}}
\end{equation}$$
$$\begin{equation}
\beta_{1}=cov(x,y)/var(x)
\end{equation}$$

And here is the much cleaner solution in matrix notation for the general case:

$$\begin{equation}
\theta = (X^{T} \cdot{X})^{-1} \cdot{X^{T}} \cdot{y}
\end{equation}$$

### Implementation

Now that we have a conceptual understanding of the linear regression algorithm, let's build it! First, we'll generate some data and plot it. We should expect to see a linear relationship between changes in $$x_{1}$$ and $$y$$.

![png](/images/convergence/linear-data.png?raw=True)

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(11)

X1 = 3*np.random.rand(100,1)
y = 5 + 4*X1 + np.random.randn(100,1)

plt.plot(X1,y,"c.")
plt.xlabel("$x_1$",fontsize=15)
plt.ylabel("$y$",fontsize=15)
plt.axis([0,3,0,20])
plt.show()
```

As expected, the data reflects a positive association of $$x_{1}$$ with $$y$$. $$x_{1}$$ is an array of 100 data points drawn from the uniform distribution ([0,1)) and $$y$$ is also an array of 100 data points dependent on $$x_{1}$$ plus some noise. The noise is randomly generated from the normal distribution, consistent with model assumptions for errors (mean = 0 and unit variance).

Now let's optimize for our parameters. We'll account for our model's intercept by appending a column of one's to the $$x_{1}$$ data. In this way, the intercept is implemented the same as any feature but with an input value of one for every instance. Next, we can implement the direct solution with one line of code and numpy! Here are the results:

```python
X0_X1 = np.c_[np.ones((100,1)),X1]
theta = np.linalg.inv(X0_X1.T.dot(X0_X1)).dot(X0_X1.T).dot(y)
```

![png](/images/convergence/theta.png?raw=True)

We can visualize our linear model as a function of $$x_{1}$$ by connecting the prediction results from its fringe values (constant slope). In this case, our data ranges from [0,3] so we'll compute predictions for these inputs and plot the fitness of our model.

```python
X_fringe = np.array([[0],[3]])
X0_X1_fringe = np.c_[np.ones((2,1)),X_fringe]
y_fringe = X0_X1_fringe.dot(theta)

plt.plot(X_fringe,y_fringe,"b-")
plt.plot(X1,y,"c.")
plt.xlabel("$x_1$",fontsize=15)
plt.ylabel("$y$",fontsize=15)
plt.axis([0,3,0,20])
plt.show()
```

![png](/images/convergence/fitness.png?raw=True)

Looking good! Finally, let's check our implementation vs a packaged version from Scikit-Learn. Our model parameters should be the same as theirs if we didn't make any mistakes.

```python
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X1,y)

print("sklearn B0: ", lr.intercept_)
print("sklearn B1: ", lr.coef_[0])
```

![png](/images/convergence/sklearn.png?raw=True)

### Gradient Descent

Gradient Descent is an iterative approach to finding the optimal set of model parameters for model training. In fact, this method is capable of finding solutions to a range of problems other than linear regression. It starts with a random initialization of parameter values and iteratively computes the change in model errors versus the most recent trial. Model training has converged to the best set of parameter values when the performance difference approximates zero (or a tolerable level of error).

The rate in which the parameters are adjusted for each trail is called the learning rate. If the learning rate is too small then training can take an unnecessarily long amount of time. But if it's too large then it's possible to miss the best set of parameters values, so it's definitely worth the time to tune.