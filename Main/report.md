# Stage1
## Dataset
I will be using the dataset on grade found at kaggle with a license of Public Domain Dedication
[original website](https://www.kaggle.com/datasets/aljarah/xAPI-Edu-Data)
This is an educational data set which is collected from learning management system (LMS) called Kalboard 360. Kalboard 360 is a multi-agent LMS, which has been designed to facilitate learning through the use of leading-edge technology. Such system provides users with a synchronous access to educational resources from any device with Internet connection.
## Cleaning dataset
The dataset is already pretty clean, I delete all the duplicated rows and any row that contains the NULL value.
```python
df.drop_duplicates(inplace = True)
df.dropna(inplace = True)
```
 Moreover, the column headers is proper readable, I just add a name for the index column. 
```python
df.index.name = 'index'
```
I think that is enough for now, the further modification will be remained for later training part.
## Data visualizations 
First graph shows the data the the distribution of the grades that students in the class get. All the student are been separate in three level, 1 means student has the greatest behavior, 3 means the baddest.

![alt 属性文本](figure1.png)

Second graph shows the distribution of the grade that students comes from, all the tested students comes form low middle and high school.

![alt 属性文本](figure2.png)

Third graph shows the the evaluation of the parents to school. From this we can see if parents's attitude will affect students'behavior.

![alt 属性文本](figure3.png)

# Stage2
## Student behavior training
In this part, I need to choose 5 significant data as input of neural network. so I choose the absenceDays, Discussion, AnnouncementsView , VisITedResources, raisedhands. Those five feature will reflect if a student would like to participate in the class. In another word, if they do not like to participate during class, they would not like to get a great grade in this class. The output is final grade that the students get in this class. 
#### Splitting data
For splitting data, I just follow the usual practice, split 80% of whole dataset for training ,then 20% for testing.
``` python
x_train,x_test,y_train,y_test = train_test_split(input_data, true_label.values.ravel(),test_size=0.2,random_state=1)
```
#### model design
I choose the CNN model for training this data. since the first part is also doing a classification task. Firstly I copy the previous network from MNIST part see if can fit this dataset. and i find out with the previous network it can reach 60% accuracy, 
#### Hyper-parameter
On this basis, I did not change any hyper-parameter, all the hyper-parameters stay the same.
#### Accuracy 
After improve the hyper-parameter, the model can reach 77.4% for Train set, and 75% for Test set
```python
13/13 - 0s - loss: 0.5253 - accuracy: 0.7739 - 144ms/epoch - 11ms/step
4/4 - 0s - loss: 0.6290 - accuracy: 0.7500 - 25ms/epoch - 6ms/step
Train / Test Accuracy: 77.4% / 75.0%
```
# Stage3
## Data visualizaion
![alt 属性文本](s3figure1.png)
![alt 属性文本](s3figure2.png)
From those two figures, it is easy to see that most of the student who get high grade in class, are more like to participate in class, their interaction with class is much higher the student who get low grade in class. Although their are some special case that the student are not likely to participate in class but they can still get high grade or vice versa, but those student are minority. 

## Improvement model

Since in last I add one more convolution layer to help extract the feature, but in this model it seams can not make such modification, so i change the some hyper-parameter to improve the accuracy of th model. I increase the learning rate to 0.005 and epochs to 50 to help the mdoel learn the feature of the data set.
```python
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```
then increase the epochs to 50 to better learning the features of the dataset 
```python
model.fit(x_train, y_train, epochs=50, verbose=1)
```
Those two modification helps the model increase accuracy from 75% to 76%.
``` python
--Evaluate model--
13/13 - 0s - loss: 0.5587 - accuracy: 0.7663 - 162ms/epoch - 12ms/step
4/4 - 0s - loss: 0.6733 - accuracy: 0.7600 - 24ms/epoch - 6ms/step
Train / Test Accuracy: 76.6% / 76.0%
```
# Stage4
## Improvement model
First I convert some text data to numerical data, and send them to training. I add 4 new input data, "ParentAnsweringSurvey", "StageID", "ParentschoolSatisfaction" and "gender" which are the potential factor that may affect the behavior of student
``` python 
data['ParentAnsweringSurvey_flaged'] = data['ParentAnsweringSurvey'].apply(convert_ParentAnsweringSurvey)
data['ParentschoolSatisfaction_flaged'] = data['ParentschoolSatisfaction'].apply(convert_ParentschoolSatisfaction)
data['StageID_flaged'] = data['StageID'].apply(convert_StageID)
data['Gender_flaged'] = data['gender'].apply(convert_gender)
input_data = data.loc[:,['absenceDays_flaged','Discussion','AnnouncementsView','VisITedResources','raisedhands','ParentAnsweringSurvey_flaged','ParentschoolSatisfaction_flaged','StageID_flaged','Gender_flaged']]
```
## Improvement for over fitting problem
Then I use the same avoiding overfitting method in heart disease part which use mutil-dense layer and random drop out strategy. All the hyper-parameter stays the same
``` python
model.add(tf.keras.layers.Dense(512, activation = "relu", use_bias= True, kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
model.add(tf.keras.layers.Dropout(0.3))
```
## Result
```python
13/13 - 0s - loss: 0.4670 - accuracy: 0.8116 - 231ms/epoch - 18ms/step
4/4 - 0s - loss: 0.5995 - accuracy: 0.7900 - 30ms/epoch - 7ms/step
Train / Test Accuracy: 81.2% / 79.0%
```