'''
PART 5: Calibration-light
- Read in data from `data/`
- Use `calibration_plot` function to create a calibration curve for the logistic regression model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
- Use `calibration_plot` function to create a calibration curve for the decision tree model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
- Which model is more calibrated? Print this question and your answer. 

Extra Credit
- Compute  PPV for the logistic regression model for arrestees ranked in the top 50 for predicted risk
- Compute  PPV for the decision tree model for arrestees ranked in the top 50 for predicted risk
- Compute AUC for the logistic regression model
- Compute AUC for the decision tree model
- Do both metrics agree that one model is more accurate than the other? Print this question and your answer. 
'''

# Import any further packages you may need for PART 5
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

# Calibration plot function 
def calibration_plot(y_true, y_prob, n_bins=10):
    """
    Create a calibration plot with a 45-degree dashed line.

    Parameters:
        y_true (array-like): True binary labels (0 or 1).
        y_prob (array-like): Predicted probabilities for the positive class.
        n_bins (int): Number of bins to divide the data for calibration.

    Returns:
        None
    """
    #Calculate calibration values
    bin_means, prob_true = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    #Create the Seaborn plot
    sns.set(style="whitegrid")
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(prob_true, bin_means, marker='o', label="Model")
    
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Plot")
    plt.legend(loc="best")
    plt.show()

# -----------------------------
# Calibration Plot
# -----------------------------

def calibration_plots(df_arrests_test, gs_cv, gs_cv_dt):
    """
    Creates calibration plots for both models.

    Args:
        df_arrests_test: Test dataframe.
        gs_cv: Fitted logistic regression GridSearchCV object.
        gs_cv_dt: Fitted decision tree GridSearchCV object.
    """
    features = ['current_charge_felony', 'num_fel_arrests_last_year']

    # Get predicted probabilities (second column = probability of class 1)
    lr_probs = gs_cv.predict_proba(df_arrests_test[features])[:, 1]
    dt_probs = gs_cv_dt.predict_proba(df_arrests_test[features])[:, 1]

    # Calibration plot for logistic regression
    calibration_plot(df_arrests_test['y'], lr_probs, n_bins=5)

    # Calibration plot for decision tree
    calibration_plot(df_arrests_test['y'], dt_probs, n_bins=5)

    print("Which model is more calibrated?")
    print("The linear regression model is better for this specific case since the second point in the decision plot line is slightly not on the line (sometimes more visibly than other times) and the linear regression plot does not have that problem.")

# -----------------------------
# Extra Credit
# -----------------------------

    #PPV
    df_arrests_test['lr_probs'] = lr_probs
    top_50 = df_arrests_test.nlargest(50, 'lr_probs')
    ppv_log = top_50['y'].mean()

    df_arrests_test['dt_probs'] = dt_probs
    top_50 = df_arrests_test.nlargest(50, 'dt_probs')
    ppv_dt = top_50['y'].mean()
    
    print(f"PPV for the logistic regression model is {ppv_log}, while the PPV for the decision tree is {ppv_dt}")

    #AUC
    auc_log = roc_auc_score(df_arrests_test['y'], lr_probs)
    auc_dt = roc_auc_score(df_arrests_test['y'], dt_probs)
    print(f"AUC for the logistic regression model is {auc_log}, while the AUC for the decision tree is {auc_dt}")

    print(f"Because the above metrics are generated on the spot AND because we use only two columns for this calculation, these metrics will be spat out diferently on each run. They might agree or disagree on which is more accurate, but, either way, the datas make them not very useful.")
