---
layout: post
title: "Optimizations inside TensorFlow (Draft)"
date: 2021-02-09 01:43:00 +0900
image_url: "/assets/xla/xlalogo.png"
mathjax: true
comments: true
---
# Symbolic Execution vs. Imperative Execution
TensorFlow uses symbolic execution, and PyTorch uses imperative execution strategy. Maybe, both terminologies are not familiar to deep learning practitioners, but in the program analysis world, those are very common words. 

Symbolic execution is a way of expressing a program by using symbols. For example, when you want to express a function that adds up to numbers, you would say `f(x, y) = x + y`. This formula does not has any concrete number, but it somehow expresses adding two numbers with __symbols__. After that, you can use this formula to get a summation by putting the actual numbers in it like `f(1, 2)`. This is exactly how symbolic execution works. In symbolic execution, you define a formula, and get the result by running the formula with inputs. So, we often call it define-and-run as well. In symbolic execution, various optimizations opportunities are available. Because a program is defined statically, optimization can traverse the program graph and simplifies some expressions or remove redundant computations. However, as you might notice, it is hard to code. 

Imperative execution (a.k.a define-by-run) is considered as its counterpart. There is no symbols in this mechanism, but only actual numbers. For example, you can get the addition of 1 and 2 by just adding it, not defining an abstract function that represents the operations. Because its simplicity, deep learning practitioner would love this paradigm, but its performance cannot be fully exploited.

# Tensorflow Design

In the [paper](https://www.usenix.org/system/files/conference/osdi16/osdi16-abadi.pdf) of TensorFlow, some design principles are introduced. Now, I am going to explain each of the design principles and its implementation in the real code. 

## Dataflow graphs of primitive operators
TensorFlow draws a graph 

## Deferred execution


## Common abstraction for heterogeneous accelerators

Tensorflow programs generate a graph which is platform independent intermediate program representation. This graph was represented in `GraphDef` in Tensorflow version 1.x, but it is deprecated in 2.x and after. However, graph representation of data flows is still used in Tensorflow code. 

In `GraphDef`