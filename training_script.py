import pickle
from PSO_class import PSO
from Aquarium import aquarium

pso = PSO(train_prey=True)

pso.run_pso()

with open('TrainingData.p', 'wb') as f:
    pickle.dump(pso, f)

