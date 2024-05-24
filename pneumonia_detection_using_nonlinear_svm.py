#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import math
from PIL import Image as PImage
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from PIL import ImageOps, ImageFilter
import cv2 as cv
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import cross_validate


# In[2]:


path = "D:\\PROGETTO_NUMERICAL_MODELS\\DATA\\"


# In[3]:


def load_data(path, image_size=(200,200), resolution_filter=(200,200), split=False):
    ''' Function to load data from directories. It takes as input:
    1) path : principal path of the data;
    2) image_size : tuple of the desired shape of the image (input of img.resize), default = (200,200);
    3) resolution_filter : tuple of the resolution to filter images. If resolution is less than it than the image is discarded;
    4) split : boolean , if true then splitting data according to the directories (test, train, validation).
    It returns:
    1) data: dictionary["train"/"test"/"val"] : list of PilImage. If split = False then "data" is the only key of the dictionary;
    2 ) labels : dictionary["train"/"test"/"val"] : np.array of the labels for each element.
    '''
    directories = ["train", "val", "test"]
    label_dir = ["NORMAL", "PNEUMONIA"]
    if split:
        data = {x:[] for x in directories}
        labels = {x:[] for x in directories}
        for folder in data:
            for label in label_dir:
                path_tp = path + folder + "\\" + label
                files = os.listdir(path_tp)
                for file in files:
                    img = PImage.open(path_tp + "\\" + file)
                    if np.array(img).shape[0] >= resolution_filter[0] and np.array(img).shape[1] >= resolution_filter[1] and len(np.array(img).shape) == 2:
                        img = img.resize(image_size)
                        data[folder].append(img)
                        labels[folder].append(label)
    else:
        data = {"data": []}
        labels = {"data": []}
        for folder in directories:
            for label in label_dir:
                path_tp = path + folder + "\\" + label
                files = os.listdir(path_tp)
                for file in files:
                    img = PImage.open(path_tp + "\\" + file)
                    if np.array(img).shape[0] >= resolution_filter[0] and np.array(img).shape[1] >= resolution_filter[1] and len(np.array(img).shape) == 2:
                        img = img.resize(image_size)
                        data["data"].append(img)
                        labels["data"].append(label)
    labels_arr = {x: 0 for x in labels}
    for folder in labels_arr:
        labels_arr[folder] = np.array(labels[folder])
    return data, labels_arr


# In[4]:


data, labels = load_data(path,image_size = (200,200),  resolution_filter = (200,200), split = False)


# In[5]:


# To show an image
data["data"][0].show()


# In[6]:


def from_image_to_number(data):
    ''' Function that trasforms Pil image in np.array. It takes as input:
    1) data: in the format of load_data output.
    It returns:
    1) data_numeric:dictionary of the same type of data but with numpy matrices in place of Pilimages. 
    '''
    data_numeric = {x:[] for x in data}
    for folder in data:
        for img in data[folder]:    
            data_numeric[folder].append(np.array(img))
    return data_numeric


# In[7]:


def from_number_to_image(data_numeric):
    '''
    Function that does the reverse of from_image_to_number function
    '''
    data_image  = {x:[] for x in data_numeric}
    for folder in data_numeric:
        for img in data_numeric[folder]:
            data_image[folder].append(PImage.fromarray(img))
    return data_image     


# In[8]:


def equalize_data(data):
    ''' Function to apply histogram equalizer on data. It takes as input:
    1) data: in the format of load_data output;
    It returns :
    1) equalized_data: dictionary of the same type of data but with equalized Pilimages in place of Pilimages.
    '''
    equalized_data = {x:[] for x in data}
    for folder in data:
        for img in data[folder]:
            equalized_data[folder].append(ImageOps.equalize(img, mask = None))
    return equalized_data


# In[9]:


def blur_gaussian(data, radius=2):
    ''' Function to apply Gaussian Blur on data. It takes as input:
    1) data: in the format of load_data output;
    2) radius: radius of Gaussian Blur kernel.
    It returns :
    1) blurred_data: dictionary of the same type of data but with blurred Pilimages in place of Pilimages.
    '''
    blurred_data = {x: [] for x in data}
    for folder in data:
        for img in data[folder]:
            blurred_data[folder].append(img.filter(ImageFilter.GaussianBlur(radius=radius)))
    return blurred_data


# In[10]:


def save_filtered_images(path_image, label = "Normal"):
    ''' Function to save filtered images
    '''
    img = PImage.open(path_image)
    img.save("Example "+ label+".jpeg")
    img = img.resize((200,200))
    img.save("Example resized (200,200)  "+ label+".jpeg")
    ImageOps.equalize(img,mask = None).save("Example "+ label+" histogram equalizer"+".jpeg")
    img.filter(ImageFilter.GaussianBlur(radius = 2)).save("Example "+ label+" Gaussian_Bluer diameter =5"+".jpeg")
    ImageOps.equalize(img,mask = None).filter(ImageFilter.GaussianBlur(radius = 2)).save("Example "+ label+" Gaussian_Blur diameter =5 + Equalizer"+".jpeg")   


# In[11]:


# To run one tome to save images
'''
path_normal = "D:\\PROGETTO_NUMERICAL_MODELS\\DATA\\train\\NORMAL\\IM-0115-0001.jpeg"
path_pneumonia = "D:\\PROGETTO_NUMERICAL_MODELS\\DATA\\train\\PNEUMONIA\\person21_bacteria_73.jpeg"
save_filtered_images(path_normal)
save_filtered_images(path_pneumonia, label = "PNEUMONIA")
'''


# In[12]:


def flatten_data(data_numeric):
    ''' Function to flatten data. It takes as input:
    1) data_numeric: as the output of from_image_to_number function.
    It returns:
    1) flattened_data_1 = dictionary of the same type of data but with matrices in place of Pilimages.
    '''
    flattened_data = {x: [] for x in data_numeric}
    for folder in data_numeric:
        for img in data_numeric[folder]:
            flattened_data[folder].append(np.reshape(img, (img.shape[0] * img.shape[1])))
    flattened_data_1 = {x: 0 for x in data_numeric}
    for folder in flattened_data_1:
        flattened_data_1[folder] =  np.vstack(flattened_data[folder])
    return flattened_data_1


# In[ ]:


############################################################# TRIAL-MODEL ON EQUALIZED DATA ####################################


# In[13]:


data_equalized_flattened = flatten_data(from_image_to_number(equalize_data(data)))
data_equalized_flattened["data"].shape


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(data_equalized_flattened["data"], labels["data"],test_size=0.20, random_state=42,
                                                    shuffle = True)
print("Number of training examples is :",X_train.shape[0])
print("Number of test examples is :",X_test.shape[0])
unique_train, counts_train = np.unique(y_train,return_counts=True)
print("-------------------TRAINING SET CLASS SPLITTING----------------------")
print(unique_train)
print(counts_train)
unique_test, counts_test = np.unique(y_test,return_counts=True)
print("-------------------TESTING SET CLASS SPLITTING----------------------")
print(unique_test)
print(counts_test)


# In[16]:


# BARPLOT DATASET SPLITTING
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(unique_train, counts_train, color=['blue', 'green'])
axes[0].set_title('Training Set Class Distribution')
axes[0].set_xlabel('Class')
axes[0].set_ylabel('Count')
axes[0].set_xticks(unique_train)
axes[0].set_xticklabels(unique_train)

axes[1].bar(unique_test, counts_test, color=['blue', 'green'])
axes[1].set_title('Testing Set Class Distribution')
axes[1].set_xlabel('Class')
axes[1].set_ylabel('Count')
axes[1].set_xticks(unique_test)
axes[1].set_xticklabels(unique_test)

fig.suptitle('Class Distribution in Training and Testing Sets', fontsize=16)
plt.tight_layout()
plt.show()

fig.savefig("class_distribution.jpeg")


# In[17]:


# MODEL FITTING
model = SVC(kernel = "rbf", C = 20, gamma = "auto")
X_train_scaled = (1/255)* X_train #scaling in [0,1]
X_test_scaled = (1/255) * X_test #scaling in [0,1]
model.fit(X_train_scaled,y_train)


# In[18]:


# MODEL EVALUATION (Accuracy)
print( "Accuracy on test set : ",model.score(X_test_scaled,y_test))
#print( "Accuracy on train set : ", model.score(X_train_scaled,y_train))


# In[19]:


def evaluate_model( data,labels, C_vector, gamma_vector, 
                   filters = ["histogram_equalizer", "Gaussian_Blur", "Bilateral_Filter","equalizer + Gaussian Blur"],
                   radius_Gaussian_Blur = 2,  
                   diameter = 5, sigma_color = 75, sigma_space = 75, save_path = False):
    ''' Function to detect best C and gamma parameter for Gaussian kernel. It takes as input:
    1) data: dictionary as the output of load_data function;
    2) labels : dictionary as the output of load_data_function;
    3) filters: list of fliter to take into account (only filter in default:filters are allowed);
    4) radius_Gaussian_Blur: radius for Gaussian blur (default = 2);
    5) diameter, sigma_color, sigma_space: float, parameters of the bilateral filter (default = 5,75,75);
    6) save_path : file name to save the numpy array of the surfaces.
    It returns:
    1) accuracy_on_test : dictionary of surfaces values where accuray_on_test[filter] = matrix where matrix_ij = accuracy(C = C_vector[i], 
    gamma = gama_vector[j]) for the correspondig filter;
    2) surface : numpy array in which surface[i,j,z] = accuracy_on_test(C_vector[i], gamma_vector[j], filters[z]).
    '''
    accuracy_on_test = {x:0 for x in filters}
    for filter in filters:
        print("Applying filter :",filter)
        if filter == "histogram_equalizer":
            data_filtered = flatten_data(from_image_to_number(equalize_data(data)))
            X_train, X_test, y_train, y_test = train_test_split(data_filtered["data"], labels["data"],test_size=0.20, random_state=42,
                                                    shuffle = True)
            surface = np.zeros((len(C_vector),len(gamma_vector)))
            X_train_scaled = (1/255)* X_train
            X_test_scaled = (1/255) * X_test
            for i in range(len(C_vector)):
                for j in range(len(gamma_vector)):
                    print("iteration C = ",C_vector[i]," gamma = ", gamma_vector[j])
                    model = SVC(kernel = "rbf", C = C_vector[i], gamma = gamma_vector[j])
                    model.fit(X_train_scaled, y_train)
                    surface[i,j] = model.score(X_test_scaled,y_test)
            accuracy_on_test[filter] = surface
        elif filter == "Gaussian_Blur":
            data_filtered = flatten_data(from_image_to_number(blur_gaussian(data, radius = radius_Gaussian_Blur )))
            X_train, X_test, y_train, y_test = train_test_split(data_filtered["data"], labels["data"],test_size=0.20, random_state=42,
                                                    shuffle = True)
            surface = np.zeros((len(C_vector),len(gamma_vector)))
            X_train_scaled = (1/255)* X_train
            X_test_scaled = (1/255) * X_test
            for i in range(len(C_vector)):
                for j in range(len(gamma_vector)):
                    print("iteration C = ",C_vector[i]," gamma = ", gamma_vector[j])
                    model = SVC(kernel = "rbf", C = C_vector[i], gamma = gamma_vector[j])
                    model.fit(X_train_scaled, y_train)
                    surface[i,j] = model.score(X_test_scaled,y_test)
            accuracy_on_test[filter] = surface
        elif filter == "Bilateral_Filter":
            data_filtered = flatten_data(from_image_to_number(bilateral_filter(data, diameter = diameter, 
                                                                                sigma_color = sigma_color, sigma_space = sigma_space)))
            X_train, X_test, y_train, y_test = train_test_split(data_filtered["data"], labels["data"],test_size=0.20, random_state=42,
                                                    shuffle = True)
            surface = np.zeros((len(C_vector),len(gamma_vector)))
            X_train_scaled = (1/255)* X_train
            X_test_scaled = (1/255) * X_test
            for i in range(len(C_vector)):
                for j in range(len(gamma_vector)):
                    print("iteration C = ",C_vector[i]," gamma = ", gamma_vector[j])
                    model = SVC(kernel = "rbf", C = C_vector[i], gamma = gamma_vector[j])
                    model.fit(X_train_scaled, y_train)
                    surface[i,j] = model.score(X_test_scaled,y_test)
            accuracy_on_test[filter] = surface
        elif filter == "equalizer + Gaussian Blur":
            data_filtered = flatten_data(from_image_to_number(blur_gaussian(equalize_data(data), radius = radius_Gaussian_Blur)))
            X_train, X_test, y_train, y_test = train_test_split(data_filtered["data"], labels["data"],test_size=0.20, random_state=42,
                                                    shuffle = True)
            surface = np.zeros((len(C_vector),len(gamma_vector)))
            X_train_scaled = (1/255)* X_train
            X_test_scaled = (1/255) * X_test
            for i in range(len(C_vector)):
                for j in range(len(gamma_vector)):
                    print("iteration C = ",C_vector[i]," gamma = ", gamma_vector[j])
                    model = SVC(kernel = "rbf", C = C_vector[i], gamma = gamma_vector[j])
                    model.fit(X_train_scaled, y_train)
                    surface[i,j] = model.score(X_test_scaled,y_test)
            accuracy_on_test[filter] = surface
    surface = np.zeros((len(C_vector),len(gamma_vector),len(filters)))
    for i in range(len(filters)):
        surface[:,:,i] = accuracy_on_test[filters[i]]  
    if save_path:
        np.save(save_path, surface)
    return accuracy_on_test, surface                        


# In[20]:


# NOT TO RUN , TOO MUCH TIME!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''
accuracy_on_test, surface = evaluate_model( data,labels, C_vector, gamma_vector, 
                   filters = ["histogram_equalizer", "Gaussian_Blur", "Bilateral_Filter","equalizer + Gaussian Blur"],
                   radius_Gaussian_Blur = 2,  
                   diameter = 5, sigma_color = 75, sigma_space = 75, save_path = "surfaces1")
'''


# In[21]:


# Loading previous results on smaller datasets
filters = ["histogram_equalizer", "Gaussian_Blur", "Bilateral_Filter","equalizer + Gaussian Blur"]
surface = np.load("surfaces.npy")
C_vector = np.linspace(1,30, num = 10)
gamma_vector = np.linspace(1/80000, 1/20000, num = 10)


# In[22]:


def plot_results_models(C_vector, gamma_vector, surface ,
                        filters = ["histogram_equalizer", "Gaussian_Blur", "Bilateral_Filter","equalizer + Gaussian Blur"],
                       save_path = False):
    ''' Function to plot surfaces. It takes as input:
    1) C_vector: numpy array of values of parameter C;
    2) gamma_vector: numpy array of values of parameter gamma;
    3) surface : numpy array where surfaces[:,:,i] is the correspondig sufacce for the corresponding filter = filters[i] 
    4) filters : list of filters applied ( it should be len(filters) = surfaces.shape[2];
    5) save_path : file_name  to save the plot. Dafault = False, it does not save the plot.
    It plots the surfaces on the same plot
    '''
    fig, axes = plt.subplots(2, len(filters) // 2, subplot_kw={'projection': '3d'}, figsize=(15, 12))
    X, Y = np.meshgrid(C_vector, gamma_vector)
    for i in range(len(filters)):
        row = i // (len(filters) // 2)
        col = i % (len(filters) // 2)
        ax = axes[row, col]
        #ax = axes[row]
        ax.plot_surface(X, Y, surface[:, :, i], cmap='viridis')
        ax.set_title(f"Applied {filters[i]}")
        ax.set_xlabel('C')
        ax.set_ylabel('Gamma')
        ax.set_zlabel('Accuracy on test set')
    fig.suptitle('Surface Plots for Different Filters', fontsize=16)
    plt.tight_layout()
    plt.show()
    if save_path:
        fig.savefig(save_path)  


# In[23]:


plot_results_models(C_vector = C_vector, gamma_vector = gamma_vector, surface = surface,
                   save_path = "surfaces.jpeg")


# In[24]:


def plot_confusion_matrix(model, X_test, y_test, image_path = False):
    '''Function to plot the confusion matrix. It takes as input:
    1) model : fitted model (sklearn model);
    2) X_test : testing data (matrix of testing data);
    3) y_test : numpy array of the labels for X_test;
    4) image_path : file_name  to save the plot. Dafault = False, it does not save the plot.
    '''
    fig, ax = plt.subplots(figsize=(10, 8))
    cm_display = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax,
                                                       colorbar=False, cmap="viridis_r")
    ax.set_title("Confusion Matrix", fontsize=18)
    ax.set_xlabel("Predicted Label", fontsize=14)
    ax.set_ylabel("True Label", fontsize=14)
    if image_path:
        plt.savefig(image_path)
    plt.show()   


# In[25]:


# Positive = PNEUMONIA
# Negative = NORMAL
def compute_performance_metrics(model,X_test, y_test):
    ''' Function to compute classical performance measures. It takes as input:
    1) model : sklearn fitted model;
    2) X_test : testing data (matrix of testing data);
    3) y_test : numpy array of the labels for X_test.
    It returns the following performance measures:
    1) accuracy;
    2) precision;
    3) recall;
    4) F1-score.
    '''
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test,y_test)
    precision, recall,f_beta_score, support = precision_recall_fscore_support(y_test, y_pred)
    return accuracy,precision[1], recall[1], f_beta_score[1]


# In[ ]:


############################################################## BEST MODEL #######################################################


# In[26]:


filter_best = "equalizer + Gaussian Blur"
data_def = flatten_data(from_image_to_number(blur_gaussian(equalize_data(data), radius = 2)))
X_train_def, X_test_def, y_train_def, y_testdef = train_test_split(data_def["data"], labels["data"],test_size=0.20, random_state=73,
                                                    shuffle = True)
X_train_def = (1/255) * X_train_def
X_test_def = (1/255) * X_test_def


# In[27]:


C_best = 47
gamma_best = 0.001
best_model =  SVC(kernel = "rbf", C = C_best, gamma = gamma_best)
best_model.fit(X_train_def, y_train_def)


# In[30]:


plot_confusion_matrix(best_model, X_test_def, y_testdef, image_path = "confusion.jpeg")


# In[29]:


accuracy,precision, recall, f_beta_score = compute_performance_metrics(best_model, X_test_def, y_testdef)
print(f"Accuracy on test set = {accuracy}") 
print("-----------------------------------")
print(f"Precision on test set = {precision}") 
print("-----------------------------------")
print(f"Recall on test set = {recall}")
print("-----------------------------------")
print(f"F-1 score on test set = {f_beta_score}")


# In[ ]:


################################################################### CROSS VALIDATION #########################################


# In[45]:


X = (1/255) * data_def["data"]
y = np.where(labels["data"] == "NORMAL", 0,1)
res_cross_val =  cross_validate(best_model, X,  y, cv=5,
                        scoring=('accuracy','precision','recall' ,'f1'),
                        return_train_score=False,verbose=1)


# In[ ]:


cross_val = pd.DataFrame(res_cross_val)
cross_val.index = [f"{i+1} fold" for i in range(len(cross_val1))]
mean_res = cross_val1.apply(np.mean)
cross_val.loc['Mean results'] = mean_res
cross_val

