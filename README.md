# any_classification
Build binary classifier just by feeding in raw data.

## User need to provide following input:
- Raw Data(should not have Nas)
- Numeric, Integer and categorical variable lists
- Random Forest parameters 
- Cross validation paramters

## How does the code works?
- Assign all the variables their respective classes 
- Splits data into test and train 
- Performs the one hot encoding for all the categorical variables 
- Generates the parameter set for the RF classifier 
- Performs cross validation
- Best model is evaluated on test data

Note: Currently only RF classifier is being used, other models would be added in future 
