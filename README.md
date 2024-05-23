
# PNEUMONIA DETECTION USING NON LINEAR SVM WITH RBF KERNEL

This project aims to create a machine learning model to detect pneumonia disease from x-ray chest images


## Library requirements
The library requirements for the code are the following:

- Pandas
- Numpy
- Sklearn
- Pillow
- cv2
- Matplotlib



## DATASET

The dataset is composed of 1430 chest x-ray images taken from Kaggle website . 80 % of the dataset is used as training set while the remaining 20 % is used as training set. Differnt image preprocessing methods are applied and evaluated.


## HOW TO RUN
In order to run the files , changing paths inside the code is needed. The code part consists of only one file named svm_pneumonia_detection.ipynb

## RESULTS
The SVM model is able to correctly classifies almost 93% of the examples in the test set. Overall performance are :
- Accuracy : 0.93;
- Precision : 0.91;
- Recall :  0.8;
- F-1 score  : 0.85.
Similar results are achieved preforming creoss validation on the entire daatset with cv = 10.
## Authors

- [@PasqualePipiciello](https://github.com/PasqualePipiciello)
- [@GianmarcoBorrata](https://github.com/GianmarcoBorrata)

