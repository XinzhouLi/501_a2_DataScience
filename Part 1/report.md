## Hyper-parameter
The optimizer is set to be Adam and the learning rate is set to 0.001. Since the learning rate was small, the epochs were increased to 30. the loss function keeps the same to be sparse_categorical_crossentropy. Two Dense layers were add into model, They had 512 and 10 units, respectively each of them have relu and softmax as the activated function. For the optimer I decide to use Adam and the learning rate to be 0.001
## Model
Since the original data is not to big then I decide to skip the convolution and pooling step. So the first layer is the Flatten layer which make the 2D matrix input data to 1D array, then I add two fully connected layer which use to do the classification task and classify the features. 
``` python
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation = "relu", use_bias= True))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```
## Result
For the train set the model can achieve 0.9987 accuracy and 0.0044 loss, for the test set, it reach 0.9807 accuracy and 0.1530 loss.
``` python
1875/1875 - 4s - loss: 0.0044 - accuracy: 0.9987 - 4s/epoch - 2ms/step
313/313 - 1s - loss: 0.1530 - accuracy: 0.9807 - 755ms/epoch - 2ms/step
Train / Test Accuracy: 99.9% / 98.1%
```
