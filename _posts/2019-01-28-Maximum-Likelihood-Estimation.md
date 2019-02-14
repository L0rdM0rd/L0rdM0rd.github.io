---
title: "Maximum Likelihood Estimation"
date: 2019-01-28
tags: [optimization-gradient ascent-logistic regression]
header:
  image: "/images/mle/nature.png"
  caption: "Photo Credit: youtube.com"
excerpt: "Implementation of Logistic Regression from scratch using the properties of maximum likelihood estimation."
mathjax: "true"
---

### Introduction

This post will build on a previous [one](https://l0rdm0rd.github.io/convergence/) which covered some techniques for convergence of linear models. As a brief review, statistical inference is the process of building a model which reflects the patterns and properties of a given dataset. In the case where the functional form is assumed to be linear, the process conveniently simplifies to computing the set of model parameters which minimize model errors.

For the general case, we need a way to approximate the functional form of the data before computing the model parameters and a common technique used in practice is known as Maximum Likelihood Estimation. We will consider an application of MLE with an implementation of Logistic Regression from scratch.

### Background

The terms probability and likelihood are semantically similar but there is a technical difference in application. First, probability is a measure of outcome where likelihood is a measure of the model. This means that we can compute the probability of an event given the underlying distribution and optimize our model such that the parameter values reflect these probabilities with the highest degree of certainty (accounting for model assumptions). In this way, likelihood quantifies trust in the model given the probabilities of observing the associated events. The formal notation is as follows:

$$\begin{equation}
L(\theta;x^{n}) = P(x^{n};\theta)
\end{equation}$$

L($$\theta$$|O) is known as the likelihood function. It estimates a function of $$\theta$$ which maximizes the chance of observing the data given the underlying distribution. It also has some nice statistical properties which allow for easier computation. Assuming the data is a random sample from some probability distribution, we can represent the joint probability distribution as the product of univariate distributions (which are all the same!).

$$\begin{equation}
L(\theta;x^{n}) = \prod_{i=1}^{n}p(x_{i};\theta)
\end{equation}$$

We can simplify things further by working with the log-likelihood. Since the log function is nondecreasing, the maximum likelihood estimate for the likelihood will be the same for the log-likelihood. This will allow us to avoid working with exponentials and convert the product of univariates to a sum of univariates which helps for derivatives. For information on log properties see this [link](https://www.khanacademy.org/math/algebra2/exponential-and-logarithmic-functions/properties-of-logarithms/a/properties-of-logarithms)

$$\begin{equation}
l(\theta;x_{i}) = logL(\theta;x_{i})
\end{equation}$$

From here, we can use good ol' calculus to partition l($$\theta$$;$$x_{i}$$) for the $$\theta$$ which is most consistent with our data. Alright, this stuff is pretty abstract. Let's put it to the test with an implementation of Logistic Regression.
