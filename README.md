# What's this
Implementation of SqueezeNet by chainer  

# Dependencies

    git clone https://github.com/nutszebra/squeeze_net.git
    cd squeeze_net
    git clone https://github.com/nutszebra/trainer.git

# How to run
    python main.py -p ./ -e 300 -b 64 -g 0 -s 1 -trb 1 -teb 1 -lr 0.1

# Details about my implementation
My squeezenet is with simple bypass and most network parameters are same as in [[1]][Paper].
However, the implementation slightly differs from the original implemenatation [[1]][Paper].
* Fire module  
As [[2]][Paper2] is reported, the order of BN_ReLU_Conv works well for residual networks, thus Fire module is composed of three BN_ReLU_Conv layers. 
* Optimization  
Optimization and hyperparameters are same as in [[3]][Paper3].
* Data augmentation  
Train: Pictures are randomly resized in the range of [124, 132], then 122x122 patches are extracted randomly and are normalized locally. Horizontal flipping is applied with 0.5 probability.  
Test: Pictures are randomly resized to 128x128, then they are normalized locally. Single image test is used to calculate total accuracy.  


# Cifar10 result
| network                 | total accuracy (%) |
|:------------------------|-------------------:|
| Alexnet [[4]][url1]     | 92.45              |
| Squeezenet [[1]][Paper] | 92.63               |

<img src="https://github.com/nutszebra/squeeze_net/blob/master/img/loss.jpg" alt="loss" title="loss">
<img src="https://github.com/nutszebra/squeeze_net/blob/master/img/accuracy.jpg" alt="total accuracy" title="total accuracy">

# References
SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size [[1]][Paper]  
Identity Mappings in Deep Residual Networks [[2]][Paper2]  
Densely Connected Convolutional Networks [[3]][Paper3]  
Alexnet implementation by torch [[4]][url1]

[paper]: https://arxiv.org/abs/1602.07360 "Paper"
[paper2]: https://arxiv.org/abs/1603.05027 "Paper2"
[paper3]: https://arxiv.org/abs/1608.06993 "Paper3"
[url1]: http://torch.ch/blog/2015/07/30/cifar.html "url1"
