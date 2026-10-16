# Neuroimaging Classification for Alzheimer’s Disease using ConvNeXt
## Overview
This project aims to create and train a ConvNeXt model to classify brain MRI scans as either having Alzheimer’s Disease (AD) or being Cognitively Normal (CN). ConvNeXt is a modern convolutional neural network that is adapted from a standard ResNet architecture.
## Feasibility Review
### User Need, Scope & Acceptance Criteria

The intended user for this model would be those in the medical field (e.g. doctors for the use of Alzheimer’s screening). The purpose of the model would be to classify brain MRI images as either Alzheimer’s Disease (AD) or Cognitively Normal (CN).

The main acceptance criteria for the model would be to obtain at least 0.8 test accuracy. Another constraint will be to limit the training runtime to 2 hours or less to avoid wait times being too long. Another important criteria would be to keep the false-negative rate to a minimum since false-negatives would correspond to someone with AD being classified as CN. For a specific threshold, false-negative rate should be kept below 0.05 (out of all the test scans with AD, at most 0.05 of them should be classified as CN).

### Model Choice & Course Concepts

The chosen model is the ConvNeXt (Hard difficulty). The reason why this kind of model would be effective is because it is based on a CNN architecture which is effective for image processing.

For dataset configuration, we can make sure we have approximately equal amounts of AD as there are CN images in both the train and test sets to avoid any class imbalance. For the test/train split, we can have 70% for training and 30% for testing since this is just a standard choice for the split (and because the dataset has already been configured like this on Rangpur).

### Preliminary Feasibility Evidence

The dataset has already been acquired and preprocessed so there are no issues there. The resources to be used would be the resources available on Rangpur which should be sufficient for model training.

A basic CNN classifier was made so that data loading and model training could be tested. When training this on Rangpur, data loaded successfully in a minute or less, and one epoch took 55 seconds. Based on this experiment it can be confirmed that the ConvNeXt will be able to be trained on Rangpur. 1 epoch per minute is a very rough estimate for training time, so 50 - 100 epochs should be reasonable to keep training under 2 hours.

### Risks, Budget & Fallback

The main risks for this would be training taking too long or wait times on the cluster being too long. The main way to mitigate this would just be to start training the model sooner. For computing budget, the model will be trained on Rangpur so the resources used will be whatever is available on the cluster. In terms of a fallback, if the requirements/acceptance criteria are unable to be met, then some of the criteria can be relaxed (e.g. accepting a lower test accuracy). Or if things fail, any previous models that were trained during experimentation can be used instead, even if their performance is poor or less than ideal.
