# RNN Model ~ Asset Prediction 

![image](https://github.com/user-attachments/assets/eba097cb-32c2-40db-b139-f65bbe6a1489)


## Impact
The RNN model has the capability to predict the future trends of stocks/assets, by utilizing years of data from the past. This highly accurate model also has advantages in being used in the blockchain.
It has been known since the beginning of it all that timing the market and getting great results does not have a correlation, hence we are told to invest only 20% of the portfolio in single stocks.
Now, this 20% might get a good return or not. Amazingly, this RNN model can help us achieve better outcomes when "risking" our money, it is able to mathematically predict where the trends are headed
and is highly accurate with a Mean Absolute Error of only 0.16%.

<hr>

Imported modules:
<li>Pandas</li>
<li>Numpy</li>
<li>Sklearn</li>
<li>Tensorflow</li>
<li>Matplotlib</li>

<br>

To begin with, I downloaded all the data about the 'AAPL'/APPLE stock from Yahoo Finance and to determine the trends I needed only the closing price. Hence, I made a dataframe that stores only the 'close'
data and their respective dates from the years 2010 to 2023.

Moving forward, we must scale the data from the original to a number ranging from 0 to 1, we use the MinMaxScaler to perform this step. Scaling is important because it improves the model's convergence
speed as neural networks perform better with normalized data.

Now we have to create sequences of "close" values, these sequences are later going to be used to train and validate our model. We have data worth 3,000+ days and the time steps used are 60 days.
<i>(So after every 60 days the 61st day will be considered the target.)</i>. After making small sequences of our data, we convert them into a numpy array.

This array will now be split into training and testing datasets, with an 80:20 split. Once we have split the inputs and targets data into X and y datasets, we must ensure that the X(inputs) are in a 3D shape 
for our LSTM layers. The y should stay in 2D as our output will be just 1 value. 

## Building the RNN model
For building our model, I used 4 LSTM and Dense layers.

```
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])
```

The first parameter in LSTM is for the number of memory cells/units, then we have ```return_sequences=True``` because we want to pass the sequences for our next LSTM layer. More LSTM layers allow us to capture 
complex patterns. The number of LSTM layers depends on the complexity of our data, we can furthermore add more layers if the results weren't as desired. <br>
Input shape should be (time_steps, features).

Next are the Dense layers to connect the neurons and the last Dense layer with the parameter 1 indicates that we need just 1 output.

### Epochs
```
history = model.fit(X_train, y_train, batch_size=32, epochs=25, validation_data=(X_test, y_test))
```
.<br>.<br>.<br>
```
Epoch 25/25
81/81 ━━━━━━━━━━━━━━━━━━━━ 3s 39ms/step - loss: 0.0026 - val_loss: 0.0137
```

Following our model build, I trained it with a small ```batch_size``` and 25 epochs. With this step we have achieved a very minimal ```loss``` and ```val_loss```.

## Prediction
```
predictions = model.predict(X_test)
```
Lastly, we test our model by using our unseen/test data, we have ```X_test``` and ```y_test``` which are our inputs and targets.
However, like we scaled our data from its original form in the beginning, we have to transform it back to visualize properly. To perform this step, the same scaler must be used and we apply the 
```.inverse_transform``` method. This method works with 2D inputs, so if the data is not already 2D we have to convert them.

```
predictions = scaler.inverse_transform(predictions)
y_test_inv = scaler.inverse_transform(y_test)
```

## Visualization
At last, we have reached the fruit of our work! With Matplotlib, we can make a pleasant graph to compare the actual test data with the data our amazingly accurate model predicted. I placed them next to each other
to ensure the visibility of the differences between both the actual and predicted.

```
plt.figure(figsize=(22, 7))
plt.plot(AAPL_data_close.index[2272:], AAPL_data_close[2272:], label='Actual Values')
plt.plot(AAPL_data_close.index[-len(y_test):], predictions, label='Predicted Values', color='lightgreen')
plt.title('Closing price prediction')
plt.xlabel('Date (every 3 months)')
plt.ylabel('Close Price')
plt.legend(loc='upper left')
plt.show()
```

![image](https://github.com/user-attachments/assets/39af6d90-344f-496f-ad7b-a7f6cbba4944)






