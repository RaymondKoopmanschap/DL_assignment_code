# Deep learning assignments
This assignments are for the course Deep learning and cover: MLPs and CNNs in the first assignment, LSTMs and GraphNNs in the second assignment and 
VAEs, GANs and generative normalizing flows in the third assignment. 

Below is a guide on how to execute them. 

## Assignment 1 (MLP and CNN)

`python train_mlp_numpy.py` and `python train_mlp_pytorch` in the assignment_1 directory (you need to be in that directory for it to work) 
will train a multi-layer perceptron on the [cifar10 dataset](http://www.cs.toronto.edu/~kriz/cifar.html) with either a numpy or pytorch implementation and test on a testset and plot a final accuracy. 
Additional arguments are possible and see the corresponding python files for options.

The CNN can be trained and tested in a similar way using `python train_convnet_pytorch.py`

## Assignment 2 (RNN, LSTM)
Go to the directory assignment_2/part1 and run `python train.py --model_type RNN` to train and test the RNN and LSTM to train and test the LSTM 
on a simple palindrome dataset.

In the directory part2 the LSTM can be used to be trained on a textbook by using `python train.py --txt_file example.txt` with the path to a text file. More options
can be found in the corresponding python file. This trained model can then be used to complete text using 
`python generation.py --text_completion text to be comple --length_pred 20`
with text that needs to be completed and the model will generate 20 new letters.

## Assignment 3 (VAE, GAN, Normalizing flows)
These models can be trained and tested using
`python a3_vae_template.py` or replace gan by gan or nf. 

This produced cool pictures like the manifolds with VAEs
![](https://github.com/RaymondKoopmanschap/DL_assignment_code/blob/master/assignment_3/code/manifold.png?raw=true)

Or interpolation with GANs

![](https://github.com/RaymondKoopmanschap/DL_assignment_code/blob/master/assignment_3/code/images/interpolation.png?raw=true)






