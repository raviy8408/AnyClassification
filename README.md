# AnyClassification
A fully automated framework for building binary classification model. It enables users to quickly test performances of various 
classification algorithms, just by feeding in cleaned & normalized tabular data set. User can select one or multiple 
models from the current list of models LR, SVM Linear, SVM Kernel, RF, XgBoost and ANN. 

The framework has several advantages like:
- It frees up the users from several redundant tasks like splitting training & testing data, transforming one hot 
encoding of the categorical variables, performing cross validation etc.
- It generates data insights by describing the data and saving various distributions plots.
- It summarises the result for quickly assessing the model performance and feature importance.
    
## User need to provide following input:
- Raw Data(should not have Nas)
- Numeric, Integer and categorical variable lists
- Model list of available models 
- Cross validation parameters for the models selected 
- Train and test data split 
- K from K-fold cross-validation
- Number of cross-validation iterations 
- Many more non-mandatory inputs

## How does the code works?
- Assign all the variables their respective types 
- Generates data insights and saves variable distribution plots
- Splits data into test and train 
- Performs the one hot encoding for all the categorical variables 
- Generates the parameter set for the models selected  
- Performs defined number of cross validation iterations for all the models selected 
- Best model is evaluated on test data
- Summarises the results using all the classification accuracy matrices and displays/saves it for all the iterations

## How to use it?
- Provide all the inputs in _user_input.py file
- Run the run.py file

### Note:  
- Users are advised to provide transformed numerical variables in the input data, there is no continuous variable 
transformation done in the code as the there could be different kind of data transformation requirement depending on the data
