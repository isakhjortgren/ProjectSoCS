from Aquarium import aquarium
import pickle
import numpy as np

class respawnAquarium(aquarium):
	def __init__(self, **aq_par):

		self.outfile = aq_par.pop("outfile")
		self.sim_time = aq_par.pop("sim_time")  #Fattar ni? Sim_time!  Haha! 

		#init super class
		super().__init__(**aq_par)

		#Run initiations
		self.fish_xy = np.copy(self.fish_xy_start )
		self.fish_vel = np.zeros(self.fish_xy_start.shape)
		self.acc_fish = np.zeros(self.fish_xy_start.shape)
		self.eaten = 0    
		self.pred_score = 0
		self.prey_score = 0
		self.max_vels = np.concatenate((\
                            np.ones(self.nbr_of_pred)*self.max_vel_pred, \
                            np.ones(self.nbr_of_prey)*self.max_vel_prey))

		self.max_acc = np.concatenate((\
                            np.ones(self.nbr_of_pred)*self.max_acc_pred, \
                            np.ones(self.nbr_of_prey)*self.max_acc_prey))       
		self.interval_pred = list(range(self.nbr_of_pred))
		self.interval_prey = list(range(self.nbr_of_pred, self.nbr_of_prey+self.nbr_of_pred))

		#TODO: test so we don't overwrite old data file by misstake 


		np.seterr(all="raise") 
	

	def remove_fishes(self,indices):
		indices = list(set(indices))
		N = len(indices)

		self.fish_xy[indices]  = self.size_X * np.random.random( (N,2) )
		self.fish_vel[indices] = np.zeros( (N,2) )
        
		self.eaten += N
		self.log_eaten_times.extend([self.i]*len(indices))

	def run_simulation(self):

		dt = 0.25*  min(self.eat_radius, self.collision_len) / max(self.max_vel_prey, self.max_vel_pred)

		iterations = int(self.sim_time // dt )
		self.time = 0

		self.MAX_TIME = self.sim_time

		self.log_eaten_times = []
		log_pos = np.empty( (iterations, self.fish_xy_start.shape[0], 2) )
		log_vel = np.empty( (iterations, self.fish_xy_start.shape[0], 2) )
		log_t = np.empty(iterations)


		next_print = 1
		self.i = 0
 		
		try:
			while self.i < iterations:
				self.timestep(dt, self.time)
	            
	            #Log stuff
				log_t[self.i] = self.time
				log_pos[self.i] = self.fish_xy 
				log_vel[self.i] = self.fish_vel

				self.time += dt
				self.i+=1 
				if self.time>next_print:
					next_print += 1
					print(self.time,"/", self.sim_time, "eaten =",self.eaten)
		except KeyboardInterrupt:
			print("KeyboardInterrupt!")             
		finally:
			with open(self.outfile, 'wb') as f:
				fish_data = {'vel_over_time': log_vel,
		                     'pos_over_time': log_pos,
		                     'nbr_pred': self.nbr_of_pred,
		                     'nbr_prey': self.nbr_of_prey, 
		                     'fishes_eaten': np.array(self.log_eaten_times),
		                     'score': (self.prey_score, self.pred_score) }
				pickle.dump(fish_data, f)
				print('data saved!')
        


if __name__ == '__main__':
	with open('TrainingData.p', 'rb') as f:
		pso_data = pickle.load(f)

	list_of_pso_pred = pso_data['list_of_pso_pred']
	list_of_pso_prey = pso_data['list_of_pso_prey']
	pso_prey = list_of_pso_prey[-1]
	pso_pred = list_of_pso_pred[-1]

	aq_par = pso_prey.aquarium_parameters

	aq_par["outfile"] = "respawn_data.p"
	aq_par["sim_time"] = 300


	aq = respawnAquarium(**aq_par)
	aq.run_simulation()