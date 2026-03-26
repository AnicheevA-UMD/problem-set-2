'''
You will run this problem set from main.py, so set things up accordingly
'''

import pandas as pd
import part1_etl
import part2_preprocessing
import part3_logistic_regression
import part4_decision_tree
import part5_calibration_plot

def main():
    # PART 1: Instanciate etl, saving the two datasets in `./data/`
    print(f"First CSV saved to {part1_etl.felony_data()}")
    print(f"Second CSV saved to {part1_etl.arrest_data()}")


    # PART 2: Call functions/instanciate objects from preprocessing
    df_arrests = part2_preprocessing.wrapper()

    # PART 3: Call functions/instanciate objects from logistic_regression
    df_arrests_train, df_arrests_test, gs_cv = part3_logistic_regression.logistic_regression(df_arrests)

    # PART 4: Call functions/instanciate objects from decision_tree
    df_arrests_train, df_arrests_test, gs_cv_dt = part4_decision_tree.decision_tree(df_arrests_train, df_arrests_test)
    # PART 5: Call functions/instanciate objects from calibration_plot
    part5_calibration_plot.calibration_plots(df_arrests_test, gs_cv, gs_cv_dt)
    
if __name__ == "__main__":
    main()
