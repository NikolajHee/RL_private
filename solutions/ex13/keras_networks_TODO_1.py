    adv_dense = layers.Dense(hidden_size, activation='relu', kernel_initializer=init())(dense2)
    adv_out = layers.Dense(num_actions, kernel_initializer=init())(adv_dense)
    v_dense = layers.Dense(hidden_size, activation='relu', kernel_initializer=init())(dense2)
    v_out = layers.Dense(1, kernel_initializer=init())(v_dense)
    norm_adv = layers.Lambda(lambda x: x - tf.reduce_mean(x))(adv_out)
    combine = layers.add([v_out, norm_adv]) 