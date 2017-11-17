import pickle
from PSO_class import PSO
from Aquarium import aquarium

pso = PSO(train_prey=False)

pso.run_pso()

aquarium_1 = aquarium(pso.aquarium_parameters)
aquarium_1.pred_brain.update_brain(pso.swarm_best_position)
aquarium_1.record_video = True



with open('TrainingData.p', 'wb') as f:
    pickle.dump(pso, f)

