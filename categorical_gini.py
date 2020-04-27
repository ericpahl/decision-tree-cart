"""Implementation of the mathematical formulation of the Gini impurity Index 
that computes the Gini impurity index for categorical variables."""
import numpy as np

def categorical_gini(X, y, varIdx):
    """Calculate the gini impurity index of categorical variables with respect to
    the outcome variable. Only appropriate for binary outcomes"""
    n_classes_ = len(set(y))  # classes are assumed to go from 0 to n-1
    n_features_ = X.shape[1]

    """Convert outcome vector to desired format"""
    # change class type to integers 0 to n-1
    distinct_classes = np.unique(y)
    for i in range(n_classes_):
        y = np.where(y == distinct_classes[i], i,y)

    """ Change categories to one-hot encoding values, create dummy variables
    only works for categories with type str and two levels """
    for i in range(n_features_):
        var = X[:,i]
        if type(var[1]) == str: # cat must be type str
            cats = np.unique(var)
            if len(cats) == 2: # cat must be two levels
                X[:,i] = np.where(var == cats[1],0,1)
    
    """Calculate gini"""
    gini = 0
    for i in range(n_classes_):
        gini += np.mean(y[X[:,varIdx]==i]) * (1-np.mean(y[X[:,varIdx]==i]))

    return gini

# default run on cmd
if __name__ == "__main__": 
    import numpy as np
    import pandas as pd
    df = pd.read_csv('~/ExampleData.csv')
    arr = df.to_numpy()
    
    # predictors
    X = arr[:,1:]
    
    # outcome
    y = arr[:,0]
    
    # index of variable to calculate
    sexIdx = 1       # idx is 1 for Sex
    censoredIdx = 4  # idx is 4 for Censored
    
    print("Sex Gini: " + str(categorical_gini(X,y,sexIdx)))
    print("Censored Gini: " + str(categorical_gini(X,y,censoredIdx)))