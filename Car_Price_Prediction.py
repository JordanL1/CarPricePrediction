import math
import random

class Car_Price_Prediction:
    """
    Models an instance of the Car_Price_Prediction problem.

    Loads, processes and stores a dataset to be used when testing/training algorithms. Provides
    methods for 
    """

    N_INPUTS = 21
    HIDDEN_LAYER_SIZE = 2
    N_WEIGHTS = N_INPUTS * HIDDEN_LAYER_SIZE + HIDDEN_LAYER_SIZE * 1
    N_BIASES = HIDDEN_LAYER_SIZE + 1
    N_PARAMETERS = N_WEIGHTS + N_BIASES

    def __init__(self, file_name="data/test.csv"):
        car_data = self.load_dataset(file_name)
        
        self.cars_x = car_data[0]
        self.cars_y = car_data[1]


    def load_dataset(self, file_name):
        csv = open(file_name, "r")

        cars_x = []
        cars_y = []

        for line in csv:
            car = line.rstrip().split(",")
            x = []

            for i in range(self.N_INPUTS):
                x.append(float(car[i]))

            cars_x.append(x)
            cars_y.append(float(car[self.N_INPUTS]))

        return [cars_x, cars_y]


    def predict(self, input, parameters):
        weight_pos = 0
        bias_pos = self.N_WEIGHTS

        hidden_layer_vals = []

        for i in range(self.HIDDEN_LAYER_SIZE):
            weighted_sum = parameters[bias_pos]
            bias_pos += 1

            for j in range(self.N_INPUTS):
                weighted_sum += input[j] * parameters[weight_pos]
                weight_pos += 1
            
            hidden_layer_vals.insert(i, self.relu(weighted_sum))

        output = parameters[bias_pos]
        for i in range(self.HIDDEN_LAYER_SIZE):
            output += hidden_layer_vals[i] * parameters[weight_pos]
            weight_pos += 1

        return output


    def evaluate(self, parameters):
        mse = 0.0

        for i in range(len(self.cars_x)):
            y_pred = self.predict(self.cars_x[i], parameters)
            mse += math.pow(self.cars_y[i] - y_pred, 2.0)
        
        mse /= len(self.cars_x)
        return mse


    def relu(self, v):
        if v < 0:
            return 0
        else:
            return v


    def bounds(self, low=-10.0, high=10.0) -> list[list[float]]:
        """
        Rectangular bounds on the search space.

        Returns: Vector b such that b[i][0] is the minimum permissible value of the
        ith solution component and b[i][1] is the maximum.
        """
        b = []
        dim = [low, high]

        for i in range(self.N_PARAMETERS):
            b.append(dim)

        return b

    
    def is_valid(self, parameters):
        """
        Check whether the ANN parameters (biases/weights) lie within the
        problem's feasible region.

        * There should be the correct number of biases and weights for the network structure.
        * Each bias/weight should lie within the range specified by the bounds.

        .. parameters ::
            :parameters: hello world
            :type parameters: int

        parameters : list
        
        """
        if len(parameters) != self.N_PARAMETERS:
            return False

        b = self.bounds()
        for i in range(len(parameters)):
            if parameters[i] < b[i][0] or parameters[i] > b[i][1]:
                return False
        
        return True

    def generate_random_parameters(self, bounds):
        params = []
        
        for i in range(len(bounds)):
            params.append(random.uniform(bounds[i][0], bounds[i][1]))

        return params
