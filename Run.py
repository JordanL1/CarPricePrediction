from Swarm import Swarm
from Car_Price_Prediction import Car_Price_Prediction

cpp = Car_Price_Prediction()
swarm = Swarm(cpp)

data = swarm.run()