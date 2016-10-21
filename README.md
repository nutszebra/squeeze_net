# What's this
Implementation of SqueezeNet by chainer  

# Dependencies

    git clone https://github.com/nutszebra/squeeze_net.git
    cd squeeze_net
    git clone https://github.com/nutszebra/trainer.git

# How to run
    python main.py -p ./ -e 300 -b 64 -g 0 -s 1 -trb 1 -teb 1 -lr 0.1

# Details about my implementation
My squeezenet is with simple bypass and most network parameters are same as [[1]][Paper].
However, the implementation slightly differs from orinal implemenatation [[1]][Paper].
* Fire module  
As [[2]][Paper2] is reported, the order of BN_ReLU_Conv works well for residual networks, thus Fire module is composed of three BN_ReLU_Conv layer. 
* Optimization  
The way of optimization and hyperparameters are same as [[3]][Paper3].

# References
SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size [[1]][Paper]  
Identity Mappings in Deep Residual Networks [[2]][Paper2]  
Densely Connected Convolutional Networks [[3]][Paper3]  

[paper]: https://arxiv.org/abs/1602.07360 "Paper"
[paper2]: https://arxiv.org/abs/1603.05027 "Paper2"
[paper3]: https://arxiv.org/abs/1608.06993 "Paper3"
