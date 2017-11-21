import pickle
from Aquarium import aquarium
import matplotlib.pyplot as plt
import matplotlib

with open('TrainingData.p', 'rb') as f:
    pso = pickle.load(f)

plt.plot(pso.list_of_swarm_best_value, 'b.')
plt.xlabel('iterations')
plt.ylabel('fitness score')
plt.show()

aquarium_1 = pso.list_of_aquarium[0]
aquarium_1.pred_brain.update_brain(pso.swarm_best_position)
aquarium_1.set_videoutput('trained_.mp4')

print(aquarium_1.run_simulation())
