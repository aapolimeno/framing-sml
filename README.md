# Automatic Detection of Media Frames in Dutch COVID-19-related News Articles 

## Description 
This project investigates the automatic detection of media framing in news articles that are related to COVID-19 by means of machine learning. 

## Data
### `annotated_sample.csv` 
Contains a sample of the annotated data, which consists of news paper texts accompanied by annotations labels. 

## Code 
### `main.py`
This Python file calls all other scripts in the pipeline. Start the experiment by running it in your terminal or command prompt as follows: 
```bash
python main.py
```
Before you do so, make sure to check if the following settings are set to your wishes: 
- The `path` variable, which can be found at the beginning of the script. It should point to where the dataset is saved on your disk. By default, a relative path is used to the sample specified above. 
- The `framing` variable, which can contain all the media framing types described in the coding book;
- The `text_reps` variable, which specifies the text representation methods (Bag of Words with TF-IDF weighting, pre-trained word embeddings, custom word embeddings, and Sentence-BERT sentence embeddings);
- The `algorithms` variable, which specifies the classification algorithms that are used in this project (logistic regression, SVM, passive aggressive classifier); 
- The `input_type` variable, which specifies the format of the input. The default setting is texts. 

By default, all framing types, text representation methods and algorithms are enabled. You only have to adapt the variables if you want to exclude certain variables. 

### `text_representation.py`
This script transforms the textual data into machine-readable format with the methods that are specified in the main script above. Please check the first few lines of the script to verify whether the necessary models are installed on your machine. If not, you can uncomment the corresponding lines. 

### `preprocessing.py`
This script transforms the raw data into the correct format. It selects relevant columns, transforms label encodings to binary representations, and splits the data into a training set and a test set. 

### `classification.py`
In this script, models are trained with the selected algorithms, and the resulting predictions are saved. 

### `evaluation.py`
This script performs the evaluation of the pipeline by means of a classification report with the Precision, Recall and F1-Score metrics (displayed in your terminal) and a confusion matrix (automatically saved in a folder called `eval`)

## `Results`
The place where the restuls (classification reports as .txt file, confusion matrices as .png) are stored
