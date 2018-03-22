# ddpg-aigym

## Deep Deterministic Policy Gradient
Implementation of Deep Deterministic Policy Gradiet Algorithm (Lillicrap et al.[arXiv:1509.02971](http://arxiv.org/abs/1509.02971).) in Tensorflow

## How to use
```
git clone https://github.com/stevenpjg/ddpg-aigym.git
cd ddpg-aigym
python main.py
```

## During training
<img src="https://www.stevenspielberg.me/projects/images/ddpg_train.gif" width="507" height="280" />

## Once trained
<img src="https://www.stevenspielberg.me/projects/images/ddpg_test.gif" width="470" height="235" />

## Learning Curve
The learning curve for InvertedPendulum-v1 environment.  
<img src="https://github.com/stevenpjg/ddpg-aigym/blob/master/learning_curve.png" width="800" height="600" />

## Dependencies
- Tensorflow (Developed in tensorflow version 0.11.0rc0 [[CPU version]](https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0rc0-cp27-none-linux_x86_64.whl) [[GPU version]](https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.11.0rc0-cp35-cp35m-linux_x86_64.whl))
- OpenAi gym
- Mujoco

## Features
- Batch Normalization (improvement in learning speed)
- Grad-inverter (given in arXiv: [arXiv:1511.04143](http://arxiv.org/abs/1511.04143))

## Note
To use different environment
```
experiment= 'InvertedPendulum-v1' #specify environments here

```
To use batch normalization
```
is_batch_norm = True #batch normalization switch
```
Let me know if there are any issues and clarifications regarding hyperparameter tuning.








