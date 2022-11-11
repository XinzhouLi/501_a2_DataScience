## Hyper-parameter
The optimizer is set to be Adam and the learning rate is set to 0.001. Since the learning rate was small, the epochs were increased to 30. the loss function keeps the same to be sparse_categorical_crossentropy. Two Dense layers were add into model, They had 512 and 20 neurons, respectively each of them have relu and softmax as the activated function.
## Model
Since the original data is not to big then I decide to skip the convolution and pooling step. So the first layer is the Flatten layer which make the 2D matrix input data to 1D array, then I add two fully connected layer which use to do the classification task and classify the features. 
## Result
The result is pretty good, for the train set the model can achieve 0.9989 accuracy and 0.0029 loss, for the test set, it reach 0.9827 accuracy and 0.1271 loss. The model is high the requirement 0.95
