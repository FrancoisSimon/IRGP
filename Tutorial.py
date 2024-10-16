

nb_states = 1
nb_obs_vars = 1
nb_hidden_vars = 3
nb_gaussians = nb_obs_vars + nb_hidden_vars

dtype = 'float64'
pi = tf.constant(np.pi, dtype = 'float64')

hidden_vars = np.repeat(np.concatenate((np.tri(nb_obs_vars + nb_hidden_vars).T, np.ones((nb_obs_vars + nb_hidden_vars, nb_hidden_vars-nb_obs_vars))), 1)[:,None,None], nb_states, 2)

hidden_vars = hidden_vars * (np.random.rand(*hidden_vars.shape)*2-1)

obs_vars = np.ones((nb_obs_vars + nb_hidden_vars, 1,nb_states, nb_obs_vars))

obs_vars = obs_vars * (np.random.rand(*obs_vars.shape)*2-1)

indices_hidden_vars = np.array(np.where(hidden_vars != 0)).T

indices_obs_vars = np.array(np.where(obs_vars != 0)).T

obs_var_coef_values = obs_vars[np.where(obs_vars != 0)]

hidden_var_coef_values = hidden_vars[np.where(hidden_vars != 0)]

Gaussian_stds = np.ones((nb_obs_vars + nb_hidden_vars, 1, nb_states, 1))

biases = np.zeros((nb_obs_vars + nb_hidden_vars, 1, nb_states))

initial_hidden_vars = np.repeat(np.tri(nb_hidden_vars)[:,None,None,::-1], nb_states, 2)

initial_hidden_vars = initial_hidden_vars * (np.random.rand(*initial_hidden_vars.shape)*2-1)

initial_obs_vars = np.zeros((nb_hidden_vars, 1, nb_states, nb_obs_vars))

indices_initial_hidden_vars = np.array(np.where(initial_hidden_vars != 0)).T
indices_initial_obs_vars = np.array(np.where(initial_obs_vars != 0)).T

initial_obs_var_coef_values = initial_obs_vars[np.where(initial_obs_vars != 0)]
initial_hidden_var_coef_values = initial_hidden_vars[np.where(initial_hidden_vars != 0)]

initial_Gaussian_stds = np.ones((nb_hidden_vars, 1, nb_states, 1))

initial_biases =  np.zeros((nb_hidden_vars, 1, nb_states))
# get oscilatory tracks

track_len = 40
nb_tracks = 1000
batch_size = nb_tracks
nb_dims = nb_obs_vars

dtype = 'float64'
pi = tf.constant(np.pi, dtype = dtype)

final_tracks, all_states = anomalous_diff_mixture(track_len=track_len,
                                                  nb_tracks = nb_tracks,
                                                  LocErr=0.02, # localization error in x, y and z (even if not used)
                                                  Fs = np.array([1.]),
                                                  Ds = np.array([0.0]),
                                                  nb_dims = 2,
                                                  velocities = np.array([0.1]),
                                                  angular_Ds = np.array([0.0]),
                                                  conf_forces = np.array([0.0]),
                                                  conf_Ds = np.array([0.0]),
                                                  conf_dists = np.array([0.0]),
                                                  LocErr_std = 0,
                                                  dt = 0.02,
                                                  nb_sub_steps = 10,
                                                  field_of_view = np.array([10,10]))

final_tracks = np.array(final_tracks)

tracks = tf.constant(final_tracks[:,None, :, None, :nb_obs_vars], dtype)

#inputs = tracks[:,:,:track_len]#tf.keras.Input(batch_shape=(batch_size, 1, track_len,1,1), dtype = dtype)
inputs = tf.keras.Input(batch_shape=(batch_size, 1, track_len,1,nb_dims), dtype = dtype)
transposed_inputs = transpose_layer(dtype = dtype)(inputs, perm = [2, 1, 0, 3, 4])

Init_layer = Initial_layer(obs_var_coef_values,
                           indices_obs_vars,
                           hidden_var_coef_values,
                           indices_hidden_vars,
                           Gaussian_stds,
                           biases,
                           nb_states,
                           nb_gaussians,
                           nb_obs_vars,
                           nb_hidden_vars,
                           initial_obs_var_coef_values,
                           indices_initial_obs_vars,
                           initial_hidden_var_coef_values,
                           indices_initial_hidden_vars,
                           initial_Gaussian_stds,
                           initial_biases,
                           trainable_params = {'obs': True, 'hidden': True, 'stds': False, 'biases': True},
                           trainable_initial_params = {'obs': False, 'hidden': True, 'stds': False, 'biases': True},
                           dtype = dtype)

tensor1, initial_states = Init_layer(transposed_inputs)

Prev_coefs, Prev_biases, LP, Log_factors, reccurent_obs_var_coefs, reccurent_hidden_var_coefs, reccurent_next_hidden_var_coefs, reccurent_biases = initial_states

sliced_inputs = tf.keras.layers.Lambda(lambda x: x[1:], dtype = dtype)(transposed_inputs)

layer = Custom_RNN_layer(Init_layer.recurrent_sequence_phase_1, Init_layer.recurrent_sequence_phase_2, dtype = dtype)
states = layer(sliced_inputs, Prev_coefs, Prev_biases, LP, Log_factors, reccurent_obs_var_coefs, reccurent_hidden_var_coefs, reccurent_next_hidden_var_coefs, reccurent_biases)

F_layer = Final_layer(Init_layer.final_sequence_phase_1, dtype = dtype)
outputs = F_layer(states)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Diffusion_model")

preds = model.predict(tracks, batch_size = batch_size)

np.mean(preds)

def MLE_loss(y_true, y_pred): # y_pred = log likelihood of the tracks shape (None, 1)
    #print(y_pred)
    
    max_LP = tf.math.reduce_max(y_pred, 1, keepdims = True)
    reduced_LP = y_pred - max_LP
    pred = tf.math.log(tf.math.reduce_sum(tf.math.exp(reduced_LP), 1, keepdims = True)) + max_LP
    
    return - tf.math.reduce_mean(pred) # sum over the spatial dimensions axis

nb_epochs = 2000
nb_data_points = batch_size*(track_len-1)

adam = tf.keras.optimizers.Adam(learning_rate=1/20, beta_1=0.9, beta_2=0.99) # after the first learning step, the parameter estimates are not too bad and we can use more classical beta parameters
model.compile(loss=MLE_loss, optimizer=adam, jit_compile = False)

with tf.device('/CPU:0'):
    history = model.fit(tracks, tracks, epochs = 1000000, batch_size = batch_size, shuffle=False, verbose = 1) #, callbacks  = [l_callback])

est_obs_var_coef_values, est_hidden_var_coef_values, est_biases, est_initial_hidden_var_coefs_values, est_initial_biases, est_fractions, est_stds, est_initial_obs_var_coefs_values, est_initial_stds = model.layers[3].weights

tf.math.softmax(est_fractions)

sparse_obs_var_coefs = tf.SparseTensor(indices = indices_obs_vars,
                                       values = est_obs_var_coef_values,
                                       dense_shape = (nb_gaussians, 1, nb_states, nb_obs_vars))

est_obs_var_coefs = tf.sparse.to_dense(sparse_obs_var_coefs)

sparse_hidden_var_coefs = tf.SparseTensor(indices = indices_hidden_vars,
                                       values = est_hidden_var_coef_values,
                                       dense_shape = (nb_gaussians, 1,nb_states, 2*nb_hidden_vars))

est_hidden_var_coefs = tf.sparse.to_dense(sparse_hidden_var_coefs)

#norm_vals = tf.math.reduce_max(tf.math.abs(hidden_var_coefs), -1, keepdims = True)
#LF = - tf.math.reduce_sum(tf.math.log(Gaussian_stds), 0)[:,:,0] + tf.math.log(tf.math.abs(tf.linalg.det(Mat)))#+ tf.math.reduce_sum(tf.math.log(tf.abs(self.hidden_var_coef_values)))
#Log_factors = - tf.math.reduce_sum(tf.math.log(Gaussian_stds), 0)[:,:,0] #+ tf.math.log(tf.math.abs(tf.linalg.det(Mat)))#+ tf.math.reduce_sum(tf.math.log(tf.abs(self.hidden_var_coef_values)))
#print('Simple log factor', Log_factors)



#print('hidden_var_coefs', hidden_var_coefs)
# change of variables to deal with gaussians of variance 1
est_hidden_var_coefs = est_hidden_var_coefs/est_stds
est_obs_var_coefs = est_obs_var_coefs/est_stds
est_biases = est_biases/est_stds[:,:,:,0]

sparse_initial_obs_var_coefs = tf.SparseTensor(indices = indices_initial_obs_vars,
                                       values = est_initial_obs_var_coefs_values,
                                       dense_shape = (nb_hidden_vars, 1, nb_states, nb_obs_vars))

est_initial_obs_var_coefs = tf.sparse.to_dense(sparse_initial_obs_var_coefs)

sparse_initial_hidden_var_coefs = tf.SparseTensor(indices = indices_initial_hidden_vars,
                                       values = est_initial_hidden_var_coefs_values,
                                       dense_shape = (nb_hidden_vars, 1, nb_states, 2*nb_hidden_vars))

est_initial_hidden_var_coefs = tf.sparse.to_dense(sparse_initial_hidden_var_coefs)


#print('hidden_var_coefs', hidden_var_coefs)
# change of variables to deal with gaussians of variance 1
est_initial_hidden_var_coefs = est_initial_hidden_var_coefs/est_initial_stds
est_initial_obs_var_coefs = est_initial_obs_var_coefs/est_initial_stds
est_initial_biases = est_initial_biases/est_initial_stds[:,:,:,0]
#self.nb_gaussians += initial_hidden_var_coefs.shape[0]

def sample_multivariate_Gaussian(est_initial_obs_var_coefs, est_initial_hidden_var_coefs, est_initial_biases, est_obs_var_coefs, est_hidden_var_coefs, nb_hidden_vars, nb_obs_vars, nb_steps, nb_samples, state = 0):
    
    #total_nb_vars = N*(nb_hidden_vars + nb_obs_vars) + nb_hidden_vars
    Log_factors = []
    initial_Log_factors = []
    
    init_obs_vars = est_initial_obs_var_coefs[:,0,state]
    init_hidden_vars = est_initial_hidden_var_coefs[:,0,state]
    init_biases = est_initial_biases[:,0,state]
    
    obs_vars = est_obs_var_coefs[:,0,state]
    hidden_vars = est_hidden_var_coefs[:,0,state]
    biases = est_biases[:,0,state]
    
    init_obs_zeros = tf.zeros(init_obs_vars.shape, dtype = dtype)
    init_hidden_zeros = tf.zeros((init_hidden_vars.shape[0], nb_hidden_vars), dtype = dtype)

    N = nb_steps
    A0 = [init_obs_vars] + [init_obs_zeros]*(N-1) + [init_hidden_vars] + [init_hidden_zeros]*(N-1)
    A0 = tf.concat(tuple(A0), axis = 1)
    
    biases_Y = [init_biases]
    A = [A0]
    for i in range(0, N):
        obs_zeros = tf.zeros(obs_vars.shape, dtype = dtype)
        hidden_zeros = tf.zeros((hidden_vars.shape[0], nb_hidden_vars), dtype = dtype)
        
        var_list = [obs_zeros]*i + [obs_vars] + [obs_zeros]*(N-i-1) + [hidden_zeros]*i + [hidden_vars] + [hidden_zeros]*(N-i-1) 
        Ai = tf.concat(tuple(var_list), axis = 1)
        
        A.append(Ai)
        biases_Y.append(biases)
    
    A = tf.concat(A, axis = 0)
    biases_Y = tf.concat(biases_Y, axis = 0)[:,None]

    inv_A = tf.linalg.inv(A)
    #print('A', A)
    #A[0,2] =0
    
    Sigma_XY = tf.matmul(inv_A, tf.transpose(inv_A))
    
    Sigma_X = Sigma_XY[:nb_steps*nb_obs_vars, :nb_steps*nb_obs_vars]
    
    est_bias_XY = tf.matmul(inv_A, biases_Y)
    est_bias_X = est_bias_XY[:nb_steps*nb_obs_vars, 0]
    
    est_samples = np.random.multivariate_normal(est_bias_X, Sigma_X, nb_samples)
    est_samples = est_samples.reshape((nb_samples, nb_steps, nb_obs_vars))
    
    return est_samples

nb_steps=50
nb_samples = 200
state = 0

est_samples_x = sample_multivariate_Gaussian(est_initial_obs_var_coefs, est_initial_hidden_var_coefs, est_initial_biases, est_obs_var_coefs, est_hidden_var_coefs, nb_hidden_vars, nb_obs_vars, nb_steps, nb_samples, state = state)
est_samples_y = sample_multivariate_Gaussian(est_initial_obs_var_coefs, est_initial_hidden_var_coefs, est_initial_biases, est_obs_var_coefs, est_hidden_var_coefs, nb_hidden_vars, nb_obs_vars, nb_steps, nb_samples, state = state)
est_samples = np.concatenate((est_samples_x, est_samples_y), axis = -1)

nb_rows = 10

est_samples_y.shape

lim = 2

plt.figure(figsize = (25, 25))

for i in range(nb_rows):
    for j in range(nb_rows):
        track = est_samples[i*nb_rows+j]
        track = track - np.mean(track,0, keepdims = True) + [[lim*i, lim*j]]
        plt.plot(track[:,0], track[:,1], alpha = 1)
plt.gca().set_aspect('equal', adjustable='box')



plt.figure(figsize = (25, 25))
nb_rows = 10
for i in range(nb_rows):
    for j in range(nb_rows):
        track = final_tracks[i*nb_rows+j]
        track = track - np.mean(track,0, keepdims = True) + [[lim*i, lim*j]]
        plt.plot(track[:,0], track[:,1], alpha = 1)
plt.gca().set_aspect('equal', adjustable='box')
