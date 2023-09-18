---
layout: post
title: "Effective Modern C++ Summary"
date: 2021-01-08 15:39:00 +0900
image_url: ""
mathjax: true
comments: true
---

# Disclaimer
This post is the personal summary and additional information of the book titled "[Effective Modern C++: 42 Specific Ways to Improve Your Use of C++11 and C++14](https://www.amazon.com/Effective-Modern-Specific-Ways-Improve/dp/1491903996)". This post lists the items introduced in the book and summarizes each of the item so that it can be used as cookbook. The post never gaurantees the correctness of the content and may contains my personal opinion. However, for those who already bought this book, it can help. If you want to cite this post, please do not cite this post but [the original book](https://www.amazon.com/Effective-Modern-Specific-Ways-Improve/dp/1491903996). I am planning to add some code examples on each item when I encounter the exact or similar situation.

# Chapter 1: Deducing Types
## Item 1: Understand template type deduction.

The keyword `auto` has pros and cons. Obviously, one of the good aspects is that you don't have to explicitly specify the type. However, the bad news is that the type deduction rules are not easy to intrinsically understand. It sometimes deduces different from your expectation. Thus, it is important to throughly understand the rules before you use. 

## Item 2: Understand `auto` type deduction.

## Item 3: Understand `decltype`

## Item 4: Know how to view deduced types.

# Chapter 2: `auto`

## Item 5: Prefer `auto` to explicit type declarations.

## Item 6: Use the explicitly typed initializer idiom when `auto` deduces undesired types.

# Chapter 3: Moving to Modern C++
## Item 7: Distinguish between `()` and `{}` when creating objects.

## Item 8: Prefer `nullptr` to `0` or `NULL`.

## Item 9: Prefer alias declarations to `typedef`s.

## Item 10: Prefer scoped `enum`s to unscoped `enum`s

## Item 11: Prefer deleted fuctions to private undefined ones.

## Item 12: Declare overriding functions `override`.

## Item 13: Prefer `const_iterator`s to `iterator`s.

## Item 14: Declare functions `noexcept` if they won't emit exceptions.

## Item 15: Use `constexpr` whenever possible.

## Item 16: Make `const` member functions thread safe.

## Item 17: Understand special member function generation.

# Chapter 4: Smart Pointers

## Item 18: Use `std::unique_ptr` for exclusive-ownership resource management.

## Item 19: Use `std::shared_ptr` for shared-ownership resource management.

## Item 20: Use `std::weak_ptr` for `std::shared_ptr`-like pointers that can dangle.

## Item 21: Prefer `std::make_unique` and `std::make_shared` to direct use of new.

## Item 22: WHen using the Pimpl Idiom, define special member functions in the implementation file.

# Chapter 5: Rvalue References, Move Semantices, and Perfect Forwarding

## Item 23: Understand `std::move` and `std::forward`.

## Item 24: Distinguish universal references from rvalue references.

## Item 25: Use `std::move` on rvalue references, `std::forward` on universal references.

## Item 26: Avoid overloading on universal references.

## Item 27: Familiarize yourself with alternamtives to overloading on universal references.

## Item 28: Understand reference collapsing.

## Item 29: Assume that move operations are not present, not cheap, and not used.

## Item 30: Familiarize yourself with perfect forwarding failure cases.

# Chapter 6: Lambda Expressions

## Item 31: Avoid default capture modes.

## Item 32: Use init capture to move objects into closures.

## Item 33: Use `decltype` on `auto&&` parameters to `std::forward` them.

## Item 34: Prefer lambdas to `std::bind`.

# Chapter 7: The Concurrency API

## Item 35: Prefer task-based programming to thread-based.

## Item 36: Specify `std::launch::async` if asynchronicity is essential.

## Item 37: Make `std::threads` unjoinable on all paths.

## Item 38: Be aware of varying thread handle destructor behavior.

## Item 39: Consider `void` futures for one-shot event communication.

## Item 40: Use `std::atomic` for concurrency, `volatile` for special memory.

# Chapter 8: Tweaks

## Item 41: Consider pass by value for copyable parameters that are cheap to move and always copied.

## Item 42: Consider emplacement instead of insertion.