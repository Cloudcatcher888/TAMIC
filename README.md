

# TAMIC

## Introduction
this is the official implementaion of TAMIC for the SDM paper "Time-aware Multi-interest Capsule Network for Sequential Recommendation". 


## Abstract
In recent years, sequential recommendation has been widely
researched, which aims to predict the next item of interest
based on user’s previously interacted item sequence. Many
works use RNN to model the user interest evolution over
time. However, they typically compute a single vector as the
user representation, which is insufficient to capture the variation of user diverse interests. Some non-RNN models employ the dynamic routing mechanism to automatically vote
out multiple capsules that represent user’s diverse interests,
but they are ignorant of the temporal information of user’s
historical behaviors, thus yielding suboptimal performance.
In this paper, we aim to establish a time-aware dynamic
routing algorithm to effectively extract temporal user multiple interests for sequential recommendation. We observe
that the significance of an item to user interests may change
monotonically over time, and user interests may fluctuate
periodically. Following the intuitive temporal patterns of
user interests, we propose a novel time-aware multi-interest
capsule network named TAMIC that leverages two kinds of
time-aware voting gates, i.e., monotonic gates and periodic
gates, to control the influence of each interacted item on
user’s current interests during the routing procedure. We
further employ an aggregation module to form a temporal
multi-interest user representation which is used for next item
prediction. Extensive experiments on real-world datasets
verify the effectiveness of the time gates and the superior
performance of our TAMIC approach on sequential recommendation, compared with the state-of-the-art methods.

## Requirement

```
pytorch == 1.14
python == 3.7
```

## Instruction
1, You can run the code by: 

```
python code/train.py
```

3, You can change customize the initial interest number K and \delta K in utils.Config.


# Reference

Please cite our paper if you use this code.

```
@inproceedings{wang2022sdm,
  title={Time-aware Multi-interest Capsule Network for Sequential Recommendation},
  author={Zhikai Wang and Yanyan Shen},
  booktitle={SDM},
  year={2022}
}
```
