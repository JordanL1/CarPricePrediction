from Car_Price_Prediction import Car_Price_Prediction
import math
import time

FILE_NAME = "data/train.csv"
RUN_TIME = 10

class Random_Search:
    
    def __init__(self, file_name=FILE_NAME):
        self.cpp = Car_Price_Prediction(file_name)
        self.bounds = self.cpp.bounds() 

    def timed_random_search(self, run_time=RUN_TIME):
        lowest_mse = math.inf
        best_sols = []

        end_time = time.time() + run_time

        while time.time() < end_time:
            solution = self.cpp.generate_random_parameters(self.bounds)
            mse = self.cpp.evaluate(solution)

            if mse < lowest_mse:
                lowest_mse = mse
                best_sols = [solution]
            elif mse == lowest_mse:
                best_sols.append(solution)

        return [lowest_mse, best_sols]