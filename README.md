# Python Plays Super Mario

## Used Deep Learning to Play super mario game with python.


## Version 1

In version 1, I have used Convolutional Neural Nets for game automation

- Dataset is created manually by playing the game for couple of hours and recording every key stroke with every frame and storing it in a numpy array.
- Dataset is cleaned and data augmentation is used to get couple of more training examples
- Used a mini version of vgg model with 3 layers and a 4th fully connected layer.
- Trained Model is used to predict the correct key stroke which is either jump or forward or fire


I was able to only get 15k samples as playing game and collecting samples is a very painful task.
Actually i got 1 Lakh samples but after data cleaning it got reduced to 15k.

### Actual Idea
    Idea is to grab every frame of the game and binding the correct key stroke with each frame and collecting samples.
    Training the Neural Net on this data and then again passing a frame to get the predicted key stroke for the particular frame.


Accuracy was about 62% and mario was making mistakes but can be further improved with more training samples.



## Version 2 comming...

In version 2, I'll be using Reinforcement Learning's Actor Critic Technique as CNN's are only good with image processing.
Actor Critic model is a technique in which we give rewards to the actor which is mario in our case, if he make's a good move in the enviroment.
Actor learns to play in the enviroment and a Q-Table is used to store these reward value's
