import random

from Car_Price_Prediction import Car_Price_Prediction


class Particle:
    """
    Represents a single particle in the PSO algorithm.
    """

    INERTIAL_COEFFICIENT = 0.721 
    COGNITIVE_COEFFICIENT = 1.1193
    SOCIAL_COEFFICIENT = 1.1193

    def __init__(self, cpp : Car_Price_Prediction, initial_solution : list[float], bounds : list[list[float]]):
        """[summary]

        Args:
            cpp (Car_Price_Prediction): [description]
            initial_solution ([type]): [description]
        """
        
        self.cpp : Car_Price_Prediction = cpp
        self.position : list[float] = initial_solution
        self.bounds : list[list[float]] = bounds
        self.pbest : list[float] = self.position
        self.pbest_cost : float = self.cpp.evaluate(self.pbest)
        self.velocity = self.initialize_velocity()


    def initialize_velocity(self):
        rand_pos : list[float] = self.cpp.generate_random_parameters(self.bounds)

        return [((p1 - p2) / 2) for (p1, p2) in zip(rand_pos, self.position)]


    def calculate_new_velocity(self, gbest):
        """
        Calculate the new velocity vector for the particle.

        Arguments:
        :param gbest: the global best position found by the swarm

        Returns: the new velocity vector
        """
        new_velocity = [0.0 for x in range(len(gbest))]
        
        for i in range(len(self.velocity)):
            new_velocity[i] = (self.INERTIAL_COEFFICIENT * self.velocity[i]) + self.COGNITIVE_COEFFICIENT * random.random() * (self.pbest[i] - self.position[i]) + self.SOCIAL_COEFFICIENT * random.random() * (gbest[i] - self.position[i])

        return new_velocity


    def calculate_new_position(self):
        """
        Use the particle's velocity to calculate its new position.

        Returns: the new position
        """
        new_position = [0.0 for x in range(len(self.position))]

        for i in range(len(self.position)):
            new_position[i] = self.position[i] + self.velocity[i]

        return new_position


    def update_particle(self, gbest):
        """
        Update the particle's velocity and position and, if the new position is better, update pbest.

        Parameters:
        gbest: the global best position found by the swarm

        Returns: the new position of the particle
        """
        self.velocity = self.calculate_new_velocity(gbest)
        self.position = self.calculate_new_position()
        new_cost = self.cpp.evaluate(self.position)

        if new_cost < self.pbest_cost:
            self.pbest = self.position
            self.pbest_cost = new_cost

        return self.position

