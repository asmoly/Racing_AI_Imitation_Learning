# Racing AI Using Imitation Learning
This is a neural network I made that uses immitation learning to drive around a track.

### Running the Program
To test the neural network run this command `python main.py -mtest -t1` in the code directory. If you want to run the test on the second track you can use the command `python main.py -mtest -t2`.

To train the neural network using the data in the code directory run the command `python main.py -mtrain`

If you want to collect your own data to train the network then you can uncomment this line `threading.Thread(target=collect_data).start()` in the main.py file and run the command `python main.py -mtest -t1`. After this you can drive around the track with the green car until the program prints "Finished collecting data". You can also adjust the number of batches to collect and the batchsize in the "collect_data" function. (Make sure to train the neural network after collecting your data)
