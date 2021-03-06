import numpy as np


class Brain(object):
    def __init__(self, nbr_of_hidden_neurons=None, nbr_of_inputs=10, nbr_of_outputs=2,
                 weight_range=0.5):
        if not nbr_of_hidden_neurons:
            nbr_of_hidden_neurons = int((nbr_of_inputs + nbr_of_outputs)/2)

        self.nbr_of_hidden_neurons=nbr_of_hidden_neurons
        self.nbr_of_inputs=nbr_of_inputs
        self.nbr_of_outputs=nbr_of_outputs
        self.weights1 = weight_range*(2*np.random.rand(nbr_of_hidden_neurons, nbr_of_inputs + 1) - 1)
        self.weights2 = weight_range*(2*np.random.rand(nbr_of_outputs, nbr_of_hidden_neurons + 1) - 1)

    def make_decision(self, total_inputs):
        total_inputs = np.concatenate((total_inputs, [1]))
        hidden_state = np.tanh(np.dot(self.weights1, total_inputs))
        hidden_state = np.concatenate((hidden_state, [1]))
        output_state = (np.dot(self.weights2, hidden_state))

        norm = np.linalg.norm(output_state) + 10**(-9)
        return output_state/norm

    def update_brain(self, weight_array):
        number_weights1 = self.nbr_of_hidden_neurons*(self.nbr_of_inputs + 1)
        self.weights1 = weight_array[0: number_weights1].reshape(self.nbr_of_hidden_neurons, self.nbr_of_inputs + 1)
        self.weights2 = weight_array[number_weights1:].reshape(self.nbr_of_outputs, self.nbr_of_hidden_neurons + 1)
#ToDo Add function that sets weight with given input weights vector


class randomBrain(Brain):
    def make_decision(self, total_inputs):
        return (np.random.random(2)-0.5)*2

    def update_brain(self, weight_array):
        raise NotImplementedError('Do not fucking train random walk sharks!!!')

class attackBrain(Brain):
    
    def make_decision(self, total_inputs):
        enemies = total_inputs[2:4]
        
        n = np.linalg.norm(enemies) 
        if n>0:
            enemies /= n       
        friend_pos = total_inputs[0:2]
        output_state = enemies - 0.2*friend_pos

       

        if len(total_inputs)==6:
            enemy_vel = total_inputs[4:6]
            enemy_vel_norm = np.linalg.norm(enemy_vel)
            if enemy_vel_norm > 0.01:
                enemy_vel /= enemy_vel_norm


            output_state += 0.3*enemy_vel

        return output_state / np.linalg.norm(output_state)

    def update_brain(self, weight_array):
        raise NotImplementedError('Do not fucking train random walk sharks!!!')

class dodgeBrain(Brain):
    def make_decision(self, total_inputs):
        perpendicular_to_shark = np.array([total_inputs[3],-total_inputs[2]])  
        position = total_inputs[-2:]   
        output_state = perpendicular_to_shark  
        output_state /=  np.linalg.norm(output_state) 

        if np.sum(output_state*position) > 0.9: 
            output_state *= -1 
        return output_state  

    def update_brain(self, weight_array):
        raise NotImplementedError('Do not fucking train random walk sharks!!!')


class NoneBrain(Brain):
    def make_decision(self, total_inputs):
        return np.zeros(2)

    def update_brain(self, weight_array):
        raise NotImplementedError('Do not fucking train None walk sharks!!!')

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    nbr_input = 6
    hn = 10
    wr = 0.1
    runs = 100000
    nbr_out = 2
    data = np.zeros((runs, 2))
    try:
        for i in range(runs):
            rand_input = 2*np.random.rand(nbr_input)-1
            tmp_brain = Brain(nbr_of_hidden_neurons=hn, nbr_of_inputs=nbr_input, weight_range=wr, nbr_of_outputs=nbr_out)
            tmp_out = tmp_brain.make_decision(rand_input)
            if nbr_out == 2:
                data[i, :] = tmp_out
            elif nbr_out == 4:
                data[i, :] = [tmp_out[0]+tmp_out[1], tmp_out[2]+tmp_out[3]]
    except:
        print('ouch')
    plt.hist2d(data[:,0], data[:,1],bins=50)
    plt.colorbar()
    plt.show()




