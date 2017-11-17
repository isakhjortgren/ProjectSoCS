import pickle
from PSO_class import PSO

pso = PSO()

pso.run_pso()

with open('TrainingData.p', 'wb') as f:
    pickle.dump(pso, f)