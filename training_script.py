import pickle
from PSO_class import PSO

pso = PSO(train_prey=False)

pso.run_pso()

with open('TrainingData.p', 'wb') as f:
    pickle.dump(pso, f)