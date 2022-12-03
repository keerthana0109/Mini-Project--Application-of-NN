# Mini-Project--Application-of-NN


(Expected the following details in the report )
## Project Title:
SPEECH EMOTION RECOGNITION USING MLPCLASSIFIER

## Project Description 

Speech Emotion Recognition, abbreviated as SER, is the act of 
attempting to recognize human emotion and the associated affective 
states from speech. This is capitalizing on the fact that voice often 
reflects underlying emotion through tone and pitch. Emotion recognition 
is a rapidly growing research domain in recent years. Unlike humans, 
machines lack the abilities to perceive and show emotions. But human-
computer interaction can be improved by implementing automated 
emotion recognition, thereby reducing the need of human intervention
. In this project, basic emotions like calm, happy, fearful, disgust etc. are 
analyzed from emotional speech signals. We use machine learning 
techniques like Multilayer perceptron Classifier (MLP Classifier) which 
is used to categorize the given data into respective groups which are non 
linearly separated. Mel-frequency cepstrum coefficients (MFCC), 
chroma and mel features are extracted from the speech signals and used
 to train the MLP classifier. For achieving this objective, we use python 
libraries like Librosa, sklearn, pyaudio, numpy and soundfile to analyze 
the speech modulations and recognize the emotion.


## Algorithm:

Our SER system consists of four main steps. First is the voice sample collection. 

The second features vector that is formed by extracting the features.

As the next step, we tried to determine which features are most relevant to differentiate each emotion. 

These features are introduced to machine learning classifier for recognition. 

## Program:
```
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score
from utils import load_data

import os
import pickle

# load RAVDESS dataset
X_train, X_test, y_train, y_test = load_data(test_size=0.25)
# print some details
# number of samples in training data
print("[+] Number of training samples:", X_train.shape[0])
# number of samples in testing data
print("[+] Number of testing samples:", X_test.shape[0])
# number of features used
# this is a vector of features extracted 
# using utils.extract_features() method
print("[+] Number of features:", X_train.shape[1])
# best model, determined by a grid search
model_params = {
    'alpha': 0.01,
    'batch_size': 256,
    'epsilon': 1e-08, 
    'hidden_layer_sizes': (300,), 
    'learning_rate': 'adaptive', 
    'max_iter': 500, 
}
# initialize Multi Layer Perceptron classifier
# with best parameters ( so far )
model = MLPClassifier(**model_params)

# train the model
print("[*] Training the model...")
model.fit(X_train, y_train)

# predict 25% of data to measure how good we are
y_pred = model.predict(X_test)

# calculate the accuracy
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

print("Accuracy: {:.2f}%".format(accuracy*100))

# now we save the model
# make result directory if doesn't exist yet
if not os.path.isdir("result"):
    os.mkdir("result")

pickle.dump(model, open("result/mlp_classifier.model", "wb"))
```


## Sample Output:
![image](https://user-images.githubusercontent.com/114254543/205448260-671c4ef1-eb85-4349-8b78-1ddea44bd7c5.png)
![image](https://user-images.githubusercontent.com/114254543/205448361-d7d6ef4b-abe3-4bdf-a34e-b65219eba127.png)

WAVEFORM:
![image](https://user-images.githubusercontent.com/114254543/205448311-35364222-fdde-4175-937d-ab2846de4eed.png)

FEATURE EXTRACTION USING MFCC:
![image](https://user-images.githubusercontent.com/114254543/205448374-e92d87e9-053f-406c-b391-b97bfc526a45.png)

MFCC’S PREDICTED EMOTION:
![image](https://user-images.githubusercontent.com/114254543/205448385-d5d7ea84-5ace-4577-b540-8ae4df51ea3d.png)

MFCC’S PREDICTED GENDER:
![image](https://user-images.githubusercontent.com/114254543/205448405-52a1b0e4-6093-41b0-82d7-f809ee2c27cf.png)


## Advantage :

➨It helps employees and HR (Human Resource) team of any company to manage stress levels. This will create healthy work environment and increase productivity. HR and managers will recongize positive and negative moods of the employees and customers which help businesses to grow.

➨The technology does not require any additional expensive hardware to adopt. AI recognition software will help in accomplishing the task of emotion sensing.

➨Real time voice based emotion analysis creates opportunities for automated customer service agents to recognize emotional state of the callers. This helps it adapt accordingly.

➨This helps companies establish deep emotional connections with their consumers through virtual assistant devices. Moreover emotion sensing wearable helps in monitoring state of mind of the users in terms of mental and other health parameters.

➨This technology help children and elderly people by providing timely medical care and assistance by alerting to their caregivers or other family members.

➨Analysis of comments on social media is helpful for the country and the world.

## Result:
Thus the speech emotion recognition using MLP classifier is implemented successfully.

