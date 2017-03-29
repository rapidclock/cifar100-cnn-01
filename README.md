# cifar100-cnn-01
Image Classifier using CNNs on the CIFAR100 Image Dataset

Data from [Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf), 2009.


## Implementation:

- This method uses a rather simple CNN with [ELU](https://arxiv.org/abs/1511.07289)(Exponential Linear Unit) rather than more conventional [RELUs](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)).
- Has about 1.6 Million Parameters for the "Model_B" which is the more accurate Model.
- After Training for 100 epochs we get ~ 44 % Accuracy on the validation(20% of the training data) and Test Set.
- Weights are saved and so is the model, so as to skip training from scratch.
- Has another model(Model_A) which is a much more deeper network, but that yields only ~30% accuracy after 45 epochs of training.

### Confusion Matrix with Model_B after 100 epochs:

![](http://i63.tinypic.com/28rcrrb.jpg)
