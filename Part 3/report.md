## Data splitting 
The training data set is 80% of the whole dataset, and the test dataset is 20% of the whole dataset. 

## Solution for non-numerical data
I convert famhist to int64, if the text is Present then it will be convert to 1, and Absent will be convert to 0.

## Solution for overfitting/underfitting
First I make several Dense layer and a large number of unit which make sure that model will overfitting, then start to decrease the number of layer and number of unit, see if the test accurary increase. Once the accuracy stops increase instead decrease I think that means the model is underfitting so I will increase again, make it reach the peak accuracy. Moreover that I use the random drop and L2 to avoid overfitting
``` python
model.add(tf.keras.layers.Dense(512, activation = "relu", use_bias= True, kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
model.add(tf.keras.layers.Dropout(0.3))
```
reference: https://www.tensorflow.org/tutorials/keras/overfit_and_underfit

## Hyperparameter & Result
Finally, I found that the best result that I can achieve is two dense layer with 500 and 10 unit and relu activation function. The learning rate set to 0.005.
```python
model.add(tf.keras.layers.Dense(500, activation = "relu", use_bias= True))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```
The epochs set to 50
``` python
model.fit(x_train, y_train, epochs=50, verbose=1)
````
The result that we get is 
``` 
12/12 - 0s - loss: 0.5928 - accuracy: 0.7027 - 134ms/epoch - 11ms/step
3/3 - 0s - loss: 0.5426 - accuracy: 0.7717 - 20ms/epoch - 7ms/step
Train / Test Accuracy: 70.3% / 77.2%
```