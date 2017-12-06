import numpy as np
import numpy.matlib as npml
from Aquarium import aquarium
import multiprocessing
from joblib import Parallel, delayed
nrb_of_cores = multiprocessing.cpu_count()

import time


class PSO(object):

    def __init__(self, aquarium_parameters, train_prey=False):
        # aquarium parameters
        self.nbr_of_hidden_neurons = aquarium_parameters['nbr_of_hidden_neurons']
        self.nbr_of_inputs = len(aquarium_parameters['input_set'])*2
        self.nbr_of_outputs = aquarium_parameters['nbr_of_outputs']
        self.weight_range = aquarium_parameters['weight_range']
        self.aquarium_parameters = aquarium_parameters

        self.train_prey = train_prey
        # PSO parameters
        self.nbr_of_validation_aquariums = 4

        self.nbr_of_aquariums = 4
        self.nbr_of_particles = 1
        self.nbr_of_iterations = 10

        self.maximum_velocity = self.weight_range
        self.c1 = 2
        self.c2 = 2
        self.inertia_weight = 1.4
        self.inertia_weight_lower_bound = 0.3
        self.beta = 0.99

        self.vector_length = (self.nbr_of_inputs + 1) * self.nbr_of_hidden_neurons + (self.nbr_of_hidden_neurons + 1) * self.nbr_of_outputs
        self.positions_matrix = self.weight_range * (2 * np.random.rand(self.nbr_of_particles, self.vector_length) - 1)
        self.velocity_matrix = self.maximum_velocity * (2 * np.random.rand(self.nbr_of_particles,
                                                                           self.vector_length) - 1)
        self.swarm_best_value = -np.inf
        self.swarm_best_position = None
        self.list_of_swarm_best_positions = list()
        self.particle_best_value = np.zeros(self.nbr_of_particles)
        self.particle_best_position = np.copy(self.positions_matrix)
        self.list_of_swarm_best_value = list()
        self.list_of_validation_results = list()
        self.list_of_aquarium = list()
        self.list_of_validation_aquarium = list()
        self.create_aquariums()

    def create_aquariums(self):
        self.list_of_aquarium = list()
        for i in range(self.nbr_of_aquariums):
            self.list_of_aquarium.append(aquarium(**self.aquarium_parameters))

        self.list_of_validation_aquarium = list()
        for i in range(self.nbr_of_validation_aquariums):
            self.list_of_validation_aquarium.append(aquarium(**self.aquarium_parameters))

    def update_brain(self, pred_or_prey, array):
        if pred_or_prey == 'prey':
            for i_aquarium in self.list_of_aquarium:
                i_aquarium.prey_brain.update_brain(array)
        elif pred_or_prey == 'pred':
            for i_aquarium in self.list_of_aquarium:
                i_aquarium.pred_brain.update_brain(array)

    def run_one_aquarium(self, i_aquarium, array):
        if self.train_prey:
            i_aquarium.prey_brain.update_brain(array)
        else:
            i_aquarium.pred_brain.update_brain(array)

        result_prey, result_pred = i_aquarium.run_simulation()

        if self.train_prey:
            return result_prey
        else:
            return result_pred

    def get_particle_position_with_best_val_fitness(self):
        index_with_best_val_fitness = np.argmax(self.list_of_validation_results)
        return self.list_of_swarm_best_positions[index_with_best_val_fitness]

    def run_all_particles(self,training_type):
        particle_values = np.zeros(self.nbr_of_particles)
        
        if not (training_type=="pred" or training_type=="prey"):
            raise ValueError("'"+ str(training_type)+"' is not a recognized training_type")

        for i_particle in range(self.nbr_of_particles):
            array = self.positions_matrix[i_particle, :]

            #check brain
            tmp_brain = self.list_of_aquarium[0].prey_brain
            tmp_brain.update_brain(array)
            defined_inputs = np.zeros((4, self.nbr_of_inputs))
            enemy_pso_start_index = 0
            if 'friend_vel' in self.aquarium_parameters['input_set']:
                enemy_pso_start_index += 2
            if 'friend_pos' in self.aquarium_parameters['input_set']:
                enemy_pso_start_index += 2
            defined_inputs[:,enemy_pso_start_index]      = np.array([0,     -0.5,   0,      0.5])
            defined_inputs[:,enemy_pso_start_index+1]      = np.array([-0.5,   0,      0.5,    0])

            passed_test = True
            for i in range(4):
                tmp_input = defined_inputs[i,:]
                tmp_decicion = tmp_brain.make_decision(tmp_input)
                tmp_enemy_vector = tmp_input[enemy_pso_start_index:enemy_pso_start_index+2]

                dot_prod = np.dot(tmp_enemy_vector, tmp_decicion) / \
                    (np.linalg.norm(tmp_enemy_vector) * np.linalg.norm(tmp_decicion))
                angle = np.arccos(np.clip(dot_prod, -1, 1))*180/3.1415

                #raise NotImplementedError("Class is not finished, see comment below")
                """" 
                Below the score is set to zero if angle<70. This is bad for sharks and good for fishes. 
                This test is not designed for sharks yet as sharks _should_ have angle<70. 
                How can we in this function tell if it's a pred brain or prey brain we're testing? 
                """
                if training_type == "prey" and angle < 45:                        
                    passed_test = False
                    break
       
                elif training_type == "pred" and  angle > 90:
                    passed_test = False
                    break
            #END FOR LOOP             
            
            if passed_test:
                print(training_type,"Brain passed test",sep="")
                list_of_result = Parallel(n_jobs=nrb_of_cores)(delayed(self.run_one_aquarium)(i_aquarium, array)
                                                               for i_aquarium in self.list_of_aquarium)    
            else:
                list_of_result = -1000   

            particle_values[i_particle] = np.mean(list_of_result)
        return particle_values

    def run_pso(self,training_type):
        
        start_time = time.time()

        for i_iteration in range(self.nbr_of_iterations):
            
            elapsed_sec = time.time()-start_time
            if elapsed_sec<1:
                time_string = ""
            else:
                time_per_iteration = elapsed_sec/i_iteration
                iterations_left = self.nbr_of_iterations - i_iteration
                ETA_sec_tot = time_per_iteration*iterations_left
                ETA_h = int(ETA_sec_tot//3600)
                ETA_min = int((ETA_sec_tot-3600*ETA_h) // 60)
                ETA_sec = int(ETA_sec_tot-3600*ETA_h-60*ETA_min)
                time_string = "ETA: " +str(ETA_h) +"h " +str(ETA_min) + "m " + str(ETA_sec) +"s"

            #print("Epoch number", i_iteration+1,"out of", self.nbr_of_iterations, time_string)

            particle_values = self.run_all_particles(training_type)

            iteration_best = np.max(particle_values)
            if iteration_best > self.swarm_best_value:
                self.swarm_best_value = iteration_best
                self.swarm_best_position = self.positions_matrix[np.argmax(particle_values), :]
                validation_scores = Parallel(n_jobs=nrb_of_cores)(delayed(self.run_one_aquarium)(i_aquarium, self.swarm_best_position) for i_aquarium in self.list_of_validation_aquarium)
                validation_score = np.mean(validation_scores)
            self.list_of_swarm_best_value.append(self.swarm_best_value)
            self.list_of_swarm_best_positions.append(self.swarm_best_position)

            self.list_of_validation_results.append(validation_score)


            temp = particle_values > self.particle_best_value
            self.particle_best_value[temp] = particle_values[temp]
            self.particle_best_position[temp] = self.positions_matrix[temp]

            q = np.random.rand(self.nbr_of_particles, 1)
            cognitive_component = self.c1 * q * (self.particle_best_position - self.positions_matrix)

            best_global_position_matrix = npml.repmat(self.swarm_best_position, self.nbr_of_particles, 1)
            r = np.random.rand(self.nbr_of_particles, 1)
            social_component = self.c2 * r * (best_global_position_matrix - self.positions_matrix)

            new_particle_velocities = self.inertia_weight*self.velocity_matrix + cognitive_component + social_component

            self.positions_matrix = self.positions_matrix + new_particle_velocities
            self.velocity_matrix = new_particle_velocities
            self.velocity_matrix[self.velocity_matrix > self.maximum_velocity] = self.maximum_velocity
            self.velocity_matrix[self.velocity_matrix < -self.maximum_velocity] = -self.maximum_velocity

            if self.inertia_weight > self.inertia_weight_lower_bound:
                self.inertia_weight = self.inertia_weight * self.beta




