def neighbourhood_weighted(self, distances):
    return np.exp(-distances ** 2 / (2 * self.visibility_range ** 2)) / self.visibility_range


def calculate_inputs_weighted(self):
    next_col = 0

    n_preds = len(self.interval_pred)
    n_preys = len(self.interval_prey)

    N = len(self.fish_xy)  # also equal to (n_preds + n_preys)

    return_matrix = np.zeros((N, self.pred_brain.nbr_of_inputs))

    ## Differences ##
    self.x_diff = np.column_stack([self.fish_xy[:, 0]] * N) - np.row_stack([self.fish_xy[:, 0]] * N)
    self.y_diff = np.column_stack([self.fish_xy[:, 1]] * N) - np.row_stack([self.fish_xy[:, 1]] * N)

    x_diff = self.x_diff
    y_diff = self.y_diff

    ## Derived matricis ##
    distances = np.sqrt(x_diff ** 2 + y_diff ** 2)
    inv_distances = 1 / (distances + 0.000000001)
    neighbr_mat = self.neighbourhood(distances)

    if "friend_vel" in self.inputs or "enemy_vel" in self.inputs:
        v_x_diff = np.column_stack([self.fish_vel[:, 0]] * N) - np.row_stack([self.fish_vel[:, 0]] * N)
        v_y_diff = np.column_stack([self.fish_vel[:, 1]] * N) - np.row_stack([self.fish_vel[:, 1]] * N)

        vel_distances = np.sqrt(v_x_diff ** 2 + v_y_diff ** 2)
        inv_vel_distances = 1 / (vel_distances + 0.000000001)

    ## PREYS: ##
    if "friend_pos" in self.inputs:
        # Prey to Prey: X & Y center of mass
        temp_matrix = neighbr_mat[n_preds:, n_preds:] * inv_distances[n_preds:, n_preds:]
        return_matrix[n_preds:, next_col] = 1 / (n_preys - 1) * np.sum(temp_matrix * x_diff[n_preds:, n_preds:], axis=0)
        return_matrix[n_preds:, next_col + 1] = 1 / (n_preys - 1) * np.sum(temp_matrix * y_diff[n_preds:, n_preds:],
                                                                           axis=0)

        # Pred-Pred: X & Y center of mass
        temp_matrix = neighbr_mat[:n_preds, :n_preds] * inv_distances[:n_preds, :n_preds]
        return_matrix[:n_preds, next_col] = 1 / (n_preds - 1) * np.sum(temp_matrix * x_diff[:n_preds, :n_preds], axis=0)
        return_matrix[:n_preds, next_col + 1] = 1 / (n_preds - 1) * np.sum(temp_matrix * y_diff[:n_preds, :n_preds],
                                                                           axis=0)

        next_col += 2

    if "friend_vel" in self.inputs:
        # Prey to prey: X & Y velocity:
        temp_matrix = neighbr_mat[n_preds:, n_preds:] * inv_vel_distances[n_preds:, n_preds:]
        return_matrix[n_preds:, next_col] = 1 / (n_preys - 1) * np.sum(temp_matrix * v_x_diff[n_preds:, n_preds:],
                                                                       axis=0)
        return_matrix[n_preds:, next_col + 1] = 1 / (n_preys - 1) * np.sum(temp_matrix * v_y_diff[n_preds:, n_preds:],
                                                                           axis=0)

        # Pred-Pred: X & Y velocity
        temp_matrix = neighbr_mat[:n_preds, :n_preds] * inv_vel_distances[:n_preds, :n_preds]
        return_matrix[:n_preds, next_col] = 1 / (n_preds - 1) * np.sum(temp_matrix * v_x_diff[:n_preds, :n_preds],
                                                                       axis=0)
        return_matrix[:n_preds, next_col + 1] = 1 / (n_preds - 1) * np.sum(temp_matrix * v_y_diff[:n_preds, :n_preds],
                                                                           axis=0)

        next_col += 2

    if "enemy_pos" in self.inputs:
        # Prey-Pred: X & Y. center of mass
        temp_matrix = neighbr_mat[:n_preds, n_preds:] * inv_distances[:n_preds, n_preds:]
        return_matrix[n_preds:, next_col] = (1 / n_preds) * np.sum(temp_matrix * x_diff[:n_preds, n_preds:], axis=0)
        return_matrix[n_preds:, next_col + 1] = (1 / n_preds) * np.sum(temp_matrix * y_diff[:n_preds, n_preds:], axis=0)

        # Pred-Prey: X & Y. center of mass
        temp_matrix = neighbr_mat[n_preds:, :n_preds] * inv_distances[n_preds:, :n_preds]
        return_matrix[:n_preds, next_col] = 1 / n_preys * np.sum(temp_matrix * x_diff[n_preds:, :n_preds], axis=0)
        return_matrix[:n_preds, next_col + 1] = 1 / n_preys * np.sum(temp_matrix * y_diff[n_preds:, :n_preds], axis=0)

        next_col += 2

    ## PREDETORS ##
    if "enemy_vel" in self.inputs:
        # Pred-Prey: X & Y. velocity
        temp_matrix = neighbr_mat[n_preds:, :n_preds] * inv_vel_distances[n_preds:, :n_preds]
        return_matrix[:n_preds, next_col] = 1 / n_preys * np.sum(temp_matrix * v_x_diff[n_preds:, :n_preds], axis=0)
        return_matrix[:n_preds, next_col + 1] = 1 / n_preys * np.sum(temp_matrix * v_y_diff[n_preds:, :n_preds], axis=0)

        # Prey-Pred: X & Y. velocity
        temp_matrix = neighbr_mat[:n_preds, n_preds:] * inv_vel_distances[:n_preds, n_preds:]
        return_matrix[n_preds:, next_col] = (1 / n_preds) * np.sum(temp_matrix * v_x_diff[:n_preds, n_preds:], axis=0)
        return_matrix[n_preds:, next_col + 1] = (1 / n_preds) * np.sum(temp_matrix * v_y_diff[:n_preds, n_preds:],
                                                                       axis=0)

        next_col += 2

    if "wall" in self.inputs:
        # Relative position to wall. X & Y. [-1, 1]
        return_matrix[:, next_col] = 2 * self.fish_xy[:, 0] / self.size_X - 1
        return_matrix[:, next_col + 1] = 2 * self.fish_xy[:, 1] / self.size_Y - 1

    return return_matrix

