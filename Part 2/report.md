## Code changed in 3 files
```python
     model = tf.keras.models.load_model(sys.argv[2])
```
The first line I changed the none to load_model function and make it receive the input name as parameter.

```python
    prediction = model.predict(img)
```
I changed the second line to 