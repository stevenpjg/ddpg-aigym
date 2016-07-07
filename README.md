# ddpg-aigym
This code contains the implementation of Deep Deterministic Policy Gradiet Algorithm (Lillicrap et al.[arXiv:1509.02971](http://arxiv.org/abs/1509.02971).)in openAi gym environments

## Dependencies
- Tensorflow
- OpenAi gym
- Mujoco

## Features
- Network configuration (given in arXiv:1509.02971)
- Grad-inverter to accelerate the learning (given in arXiv: 1511.04143)

## Results
The learning curve after ~1100 episodes for InvertedPendulum-v1 environment. (Trained in Nvidia GTX-960M GPU, took ~10 hours)
![alt tag](http://url/to/img.png)

## How to use
1. Install the required dependencies
2. Run main.py
3. For using grad inverter specify the max and min actions in action_bounds variable in ddpg.py file as per the structure given in comments (default action_bound value is for InvertedPendulum-v1)




