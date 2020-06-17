# PPO
Proximal Policy Optimization and Generalized Advantage Estimation implementation with Tensorflow2  
This implementation only supports CartPole environment(OpenAI gym).  

このリポジトリは強化学習アルゴリズムProximal Policy Optimization及びGeneralized Advantage EstimationをTensorflow2で実装したものです。学習環境はCartPoleにのみ対応しています。  
アルゴリズムの解説ブログ記事はこちらになります。

## Relevant Papers
 - Proximal Policy Optimization Algorithms, Schulman et al. 2017  
https://arxiv.org/abs/1707.06347
 - High-Dimensional Continuous Control Using Generalized Advantage Estimation, Schulman et al. 2016  
https://arxiv.org/pdf/1506.02438.pdf

## Requirements
 - Python3
 
## Dependencies
 - tensorflow2
 - gym
 - tqdm

## Usage
  - clone this repo
 ```
 $ git clone https://github.com/morikatron/PPO.git
 ```
  - change directory and run 
 ```
 $ cd ppo
 $ python run.py
 ```
