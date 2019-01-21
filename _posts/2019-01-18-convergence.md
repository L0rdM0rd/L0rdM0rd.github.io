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

In this equation, $$f$$ is our model and $$\epsilon$$ represents the inherent variance of $$y$$ relative to our model. Linear models associate a linear shape for $$f$$. Here is the general form of a linear model:

$$\begin{equation}
f(X) = \beta_{0} + \beta_{1}X_{1} + \beta_{2}X_{2} +...+\beta_{n}X_{n}
\end{equation}$$

$$\beta_{0}$$ is a constant term representing the bias for model training. Every other parameter has an associated feature ($$X_{1:n}$$). Given some data, we can use this model to make predictions which are essentially a weighted sum of the input features. These predictions are evaluated using a performance measure, typically RMSE (Root Mean Squared Error). Here, RMSE is represented using matrix notation for convenience.

$$\begin{equation}
RMSE = ((1/m)\sum_{i=1}^{m}[\theta^{T}X_{i}-y_{i}]^{2})^{1/2}
\end{equation}$$

For each instance, we take difference between our prediction and the label, add up all of the differences, and normalize the result. It follows that we need to find the best combination of parameters which produce the lowest RMSE. There are two approaches to accomplish this task.

### The Closed Form Solution
