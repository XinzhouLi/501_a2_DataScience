## Code changed in 3 files
```python
     model = tf.keras.models.load_model(sys.argv[2])
```
The first line I changed the none to load_model function and make it receive the input name as parameter.

```python
    prediction = model.predict(img)
```
I changed the second line to make the prediction of the input image that we input.
```python
    plot(class_names, prediction[0], true_label, predicted_label, img[0])
```
The third place that I change is that I check the result of the prediction var, find it is a 2d list, so I just take the first element as the parameter of the plot function.

## Wrong prediction
88
I found out that my program can not correctly classify the small size letter, I think it may caused by the I did not add any convolution layer 
