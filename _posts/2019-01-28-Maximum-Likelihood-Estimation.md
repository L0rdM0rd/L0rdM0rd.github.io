---
title: "Statistical Properties of Maximum Likelihood Estimation"
date: 2019-01-28
tags: [optimization-gradient ascent-logistic regression]
header:
  image: "/images/mle/nature.png"
  caption: "Photo Credit: youtube.com"
excerpt: "Implementation of Logistic Regression from scratch using the properties of maximum likelihood estimation."
mathjax: "true"
---

### Introduction

This post will build on a previous [one](https://l0rdm0rd.github.io/convergence/) which covered techniques for convergence of linear models. As a brief review, statistical inference is the process of finding a function that best describes the patterns and relationships of a given dataset. In the case where this function is linear, this process conveniently simplifies to computing the set of model parameters which minimize model errors. For the general case, we need a way to estimate this function before computing the model parameters.

### Background
