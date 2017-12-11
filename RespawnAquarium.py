from Aquarium import aquarium
import pickle
import numpy as np

import matplotlib.pyplot as plt
try:
    import matplotlib.animation as manimation
except ImportError:
    pass


class respawnAquarium(aquarium):
	def __init__(self, **aq_par):

		self.outfile = aq_par.pop("outfile")
		self.sim_time = aq_par.pop("sim_time")  #Fattar ni? Sim_time!  Haha! 

		#init super class
		super().__init__(**aq_par)
		print(aq_par)

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
		self.fish_vel[indices] = np.zeros((N, 2))
		self.acc_fish[indices] = np.zeros((N, 2))
        
		self.eaten += N
		self.log_eaten_times.extend([self.i]*len(indices))

	def initialize_simulation(self):
		self.dt = 0.25 * min(self.eat_radius, self.collision_len) / max(self.max_vel_prey, self.max_vel_pred)

		self.iterations = int(self.sim_time // self.dt)
		self.time = 0

		self.MAX_TIME = self.sim_time

		self.log_eaten_times = []
		self.log_pos = np.empty((self.iterations, self.fish_xy_start.shape[0], 2))
		self.log_vel = np.empty((self.iterations, self.fish_xy_start.shape[0], 2))
		self.log_t = np.empty(self.iterations)

		self.next_print = 1
		self.i = 0

	def run_simulation(self):

		self.initialize_simulation()

		try:
			while self.i < self.iterations:
				self.timestep(self.dt, self.time)

				#Log stuff
				self.log_t[self.i] = self.time
				self.log_pos[self.i] = self.fish_xy
				self.log_vel[self.i] = self.fish_vel

				self.time += self.dt
				self.i+=1
				if self.time>self.next_print:
					self.next_print += 5
					print(self.time,"/", self.sim_time, "eaten =",self.eaten)
		except KeyboardInterrupt:
			print("KeyboardInterrupt!")
		finally:
			with open(self.outfile, 'wb') as f:
				fish_data = {'vel_over_time': self.log_vel,
		                     'pos_over_time': self.log_pos,
		                     'nbr_pred': self.nbr_of_pred,
		                     'nbr_prey': self.nbr_of_prey,
		                     'fishes_eaten': np.array(self.log_eaten_times),
		                     'score': (self.prey_score, self.pred_score) }
				pickle.dump(fish_data, f)
				print('data saved!')

	def run_simulation_video(self, video_filename):

		self.set_videoutput(video_filename)
		self.initialize_simulation()

		try:
			with self.video_writer.saving(self.fig, self.video_filename, self.video_dpi):
				while self.i < self.iterations:
					self.timestep(self.dt, self.time)
					self.grab_Frame_()

					#Log stuff

					self.log_t[self.i] = self.time
					self.log_pos[self.i] = self.fish_xy
					self.log_vel[self.i] = self.fish_vel

					self.time += self.dt
					self.i+=1
					if self.time>self.next_print:
						self.next_print += 5
						print(self.time,"/", self.sim_time, "eaten =",self.eaten)
		except KeyboardInterrupt:
			print("KeyboardInterrupt!")             
		finally:
			with open(self.outfile, 'wb') as f:
				fish_data = {'vel_over_time': self.log_vel,
		                     'pos_over_time': self.log_pos,
		                     'nbr_pred': self.nbr_of_pred,
		                     'nbr_prey': self.nbr_of_prey, 
		                     'fishes_eaten': np.array(self.log_eaten_times),
		                     'score': (self.prey_score, self.pred_score) }
				pickle.dump(fish_data, f)
				print('data saved!')



if __name__ == '__main__':
	with open('TrainingData7.p', 'rb') as f:
		pso_data = pickle.load(f)

	list_of_pso_pred = pso_data['list_of_pso_pred']
	list_of_pso_prey = pso_data['list_of_pso_prey']
	pso_prey = list_of_pso_prey[-1]
	pso_pred = list_of_pso_pred[-1]

	aq_par = pso_prey.aquarium_parameters

	aq_par["outfile"] = "respawn_data.p"
	aq_par["sim_time"] = 150

	aq_par["size_X"] = 10
	aq_par["size_X"] = 10


	aq = respawnAquarium(**aq_par)
	aq.pred_brain.update_brain( pso_pred.get_particle_position_with_best_val_fitness() )
	aq.prey_brain.update_brain( pso_prey.get_particle_position_with_best_val_fitness() )




	#aq.run_simulation()
	aq.run_simulation_video("video_filename.mp4")