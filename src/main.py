'''
You will run this problem set from main.py, so set things up accordingly
'''

import pandas as pd
import part1_etl


# Call functions / instanciate objects from the .py files
def main():
    print(f"First CSV saved to {part1_etl.felony_data()}")
    print(f"Second CSV saved to {part1_etl.arrest_data()}")
    # PART 1: Instanciate etl, saving the two datasets in `./data/`

    # PART 2: Call functions/instanciate objects from preprocessing

    # PART 3: Call functions/instanciate objects from logistic_regression

    # PART 4: Call functions/instanciate objects from decision_tree

    # PART 5: Call functions/instanciate objects from calibration_plot


if __name__ == "__main__":
    main()