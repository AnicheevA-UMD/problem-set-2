'''
PART 4: Decision Trees
- Read in the dataframe(s) from PART 3
- Create a parameter grid called `param_grid_dt` containing three values for tree depth. (Note C has to be greater than zero) 
- Initialize the Decision Tree model. Assign this to a variable called `dt_model`. 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv_dt`. 
- Run the model 
- What was the optimal value for max_depth?  Did it have the most or least regularization? Or in the middle? 
- Now predict for the test set. Name this column `pred_dt` 
- Save dataframe(s) save as .csv('s) in `data/`
'''

# Import any further packages you may need for PART 4
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.tree import DecisionTreeClassifier as DTC

# -----------------------------
# Decision Tree
# -----------------------------


def decision_tree(df_arrests_train, df_arrests_test) -> tuple:
    """
    Runs decision tree with GridSearchCV on the arrests data.

    Args:
        df_arrests_train: Training dataframe from Part 3.
        df_arrests_test: Test dataframe from Part 3.

    Returns:
        tuple: (df_arrests_train, df_arrests_test) with predictions added.
    """

    # Setup
    features = ['current_charge_felony', 'num_fel_arrests_last_year']

    param_grid_dt = {'max_depth': [2, 5, 10]}

    # Training model & validation
    dt_model = DTC()
    gs_cv_dt = GridSearchCV(dt_model, param_grid_dt, cv=5)
    gs_cv_dt.fit(df_arrests_train[features], df_arrests_train['y'])

    # Print optimal depth and interpretation
    print(f"Optimal value for max_depth: {gs_cv_dt.best_params_['max_depth']}")
    print(f"For these results, 2 has the most regularization, 5 is in the middle and 10 has the least regularization. Best score ends up being {gs_cv_dt.best_score_:.4f}. But, like before, the gs_cv_dt is equal for all three depths which makes the code pick the first value I put in - which in this case is 2. The data's lack of features (only two) makes these results not useful in a real scenario.")

    # Predict on test set
    df_arrests_test = df_arrests_test.copy()
    df_arrests_test['pred_dt'] = gs_cv_dt.predict(df_arrests_test[features])

    # Save to CSV
    df_arrests_train.to_csv('../data/df_arrests_train.csv', index=False)
    df_arrests_test.to_csv('../data/df_arrests_test.csv', index=False)

    return df_arrests_train, df_arrests_test, gs_cv_dt