## BIO-INF-project--Pseudo-coma-detection

An algorithm to detect a steady and stable alpha rhythm in an EEG recording. First, descriptors are computed for fixed-time (2s) windows of the EEG signals, then a classifier determines whether these windows contain the searched for alpha rhythm.

For confidentiality reasons, EEG files are not provided. You are, however, welcome to test your own filess on it.

# Visualiser.py
Displays the EEG files in multi-channel or single channel mode.

# CreateData-pt1.py
Computes descriptors for EEGs. EEG recordings are needed to run it.

# CreateData-pt2.py
Creates labeled data points from descriptors and annotations.

# SupervisedModel.py and unsupervisedModel.py
Train and export a model.
These are two different approaches to perform the task. Supervised classifier is an SVM with RBF kernel, and unsupervised classifier is a k-means classifier.

# Pipeline.py
Complete pipeline to compute descriptors and determine portions of an EEG recording with a stable and strong alpha signal (needs a new EEG file). 



