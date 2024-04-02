---
layout: post
title: Count on Steroids
subtitle: 
date: 2023-03-20
comments: true
tags: [algorithem , machine learning, deep learning, papers, benchmark]
---



## Count on Steroids

If you need to count specific items in a list, using the Counter function from the collections module is a simple and effective solution.


```python

from collections import Counter 


items = [1,2,3,1,1,1,3,3,4,5,6]

Counter(items)

# Counter({1: 4, 2: 1, 3: 3, 4: 1, 5: 1, 6: 1})
```

recently i discovered a very interesting extension of this problem where i had to count and see how many times certain 


```python
sentences_1 = ['The quick brown fox jumped over the lazy dog',\
              'I like to eat pizza',\
              'The sun is shining today', \
              'I went to the store to buy some milk',\
              'My favorite color is blue', \
              'I enjoy playing video games',\
              'She plays the piano beautifully', \
              'The cat sat on the mat', \
              'I am going to the beach this weekend']
```
