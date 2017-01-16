# Anomaly Detection using Deep Learning Network on Intrusion Detection Dataset
- In this project, a multi-layer feed forward neural network has been used to claasify anomalies (attacks) on a dataset released by Cyber Range Lab of the Australian Centre for Cyber Security (ACCS). 
- The dataset can be accessed here: https://www.unsw.adfa.edu.au/australian-centre-for-cyber-security/cybersecurity/ADFA-NB15-Datasets/
- Furthermore, DeepLearning4j framework has been used to implement the MLP using Java. First, you set up the whole maven project in Java and try running this basec example from Deeplearning4J framework: https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/feedforward/ classification/MLPClassifierSaturn.java
- Modified code has been provided above. Point ot note is that you have to do preprocessing input data to make it suitable for deep learning architecture. Some crucial steps are:
--- Normalizing data using mean and variance.
--- Using one hot encoding to convert categorical variables into sparse input vectors. This will certainly increase the input vectors in the input data.


