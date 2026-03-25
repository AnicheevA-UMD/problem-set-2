'''
You will run this problem set from main.py, so set things up accordingly
'''

import pandas as pd
import part1_etl
import part2_preprocessing
import part3_logistic_regression



def main():
    # PART 1: Instanciate etl, saving the two datasets in `./data/`
    print(f"First CSV saved to {part1_etl.felony_data()}")
    print(f"Second CSV saved to {part1_etl.arrest_data()}")


    # PART 2: Call functions/instanciate objects from preprocessing
    df_arrests = part2_preprocessing.wrapper()

    # PART 3: Call functions/instanciate objects from logistic_regression
    df_arrests_train, df_arrests_test = part3_logistic_regression.logistic_regression(df_arrests)

    # PART 4: Call functions/instanciate objects from decision_tree

    # PART 5: Call functions/instanciate objects from calibration_plot


if __name__ == "__main__":
    main()
