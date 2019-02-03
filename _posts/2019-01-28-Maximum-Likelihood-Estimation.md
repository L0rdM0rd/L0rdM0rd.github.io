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

This post will build on a previous [one](https://l0rdm0rd.github.io/convergence/) which covered some techniques for convergence of linear models. As a brief review, statistical inference is the process of building a model that reflects the patterns and properties of a given dataset. In the case where the functional form is assumed to be linear, the process conveniently simplifies to computing the set of model parameters which minimize model errors.

For the general case, we need a way to approximate the functional form of the data before computing the model parameters and a common technique used in practice is known as Maximum Likelihood Estimation. We will consider an application of MLE with an implementation of Logistic Regression from scratch.

### Background

The terms probability and likelihood are semantically similar but there is a technical difference in application. Firstly, probability is a measure of outcome where likelihood is a measure of the model. This means that we can compute the probability of an event given the underlying distribution and optimize our model such that the parameter values reflect these probabilities with the highest degree of certainty (given model assumptions). In this way, likelihood quantifies trust in the model given the probabilities of observing certain events. The formal notation is as follows:

$$\begin{equation}
L(\theta|O) = P(O|\theta)
\end{equation}$$

L(\theta|O) is called the likelihood function 
