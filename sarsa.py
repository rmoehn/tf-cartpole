import itertools

import gym
import numpy as np
import tensorflow as tf


#### Helper functions

def true_with_prob(p):
    return np.rand(1)[0] < p


#### Parameters of the algorithm and training

alpha           = 0.001
lmbda           = 0.9
epsi            = 0.1
fourier_order   = 3
N_episodes      = 400
N_max_steps     = 500


#### Setup for the environment

env = gym.make('CartPole-v1')

high = np.array([2.5, 4.4, 0.28, 3.9])
o_ranges = np.array([-high, high])


#### Derived values

N_acts          = env.action_space.n    # Assumes discrete action space
N_dims          = o_ranges.shape[1]
N_weights_per_a = (fourier_order + 1)**N_dims


#### Fourier function approximation

C = np.array(
        list( itertools.product(range(fourier_order+1), repeat=N_dims) ),
        dtype=np.int32 )

tC              = tf.constant(C, dtype=tf.float64)
tlow            = tf.constant(-high)
to_ranges_diff  = tf.constant( np.diff(o_ranges, axis=0) )
tpi             = tf.constant(np.pi, dtype=tf.float64)

def phi(local_to):
    tnc_o = tf.div( tf.subtract(local_to, tlow), to_ranges_diff)
        # normalized, centered
    return tf.cos( tf.mul(tpi, tf.matmul(tC, tf.transpose(tnc_o))) )


#### Set up variables for the algorithm

vtheta  = tf.Variable(tf.zeros([N_acts, high.shape[0]]))
velig   = tf.Variable(tf.float64, [N_acts, N_weights_per_a])


#### Set up placeholders for the algorithm

to      = tf.placeholder(tf.float64, shape=high.shape)
tpo     = tf.placeholder(tf.float64, shape=high.shape)

tr      = tf.placeholder(tf.float32, shape=[1])

ta      = tf.placeholder(tf.int32, shape=[1])
tga     = tf.placeholder(tf.int32, shape=[1])
tpa     = tf.placeholder(tf.int32, shape=[1])


#### Assemble the graph

tphio   = phi(to)
tphipo  = phi(tpo)

tQall       = tf.matmul(vtheta, tf.transpose(tphio))

vthetapa    = tf.slice(vtheta, [tpa, 0], [1, N_weights_per_a])
tpQpopa     = tf.matmul(vthetapa, tphipo)

vthetaa     = tf.slice(vtheta, [ta, 0], [1, N_weights_per_a])
tpQoa       = tf.matmul(vthetaa, tphio)

velig_a     = tf.slice(velig, [ta, 0], [1, N_weights_per_a])
update_elig = velig_a.assign( tf.add(tf.mul(lmbda, velig_a), tphio) )

loss            = tf.mul(tf.sub(tpQpopa, tf.add(tr, tpQoa)), velig_a)
optimizer       = tf.train.GradientDescentOptimizer(learning_rate=alpha)
update_model    = optimizer.minimze(loss)


#### Core algorithm

for n_episode in xrange(N_episodes):
    po      = None
    phipo   = None
    pa      = None
    o       = env.reset()
    r       = 0
    done    = False

    for n_step in xrange(N_max_steps):
        phio = tf.run([tphio], feed_dict={to: o})
        if not done:
            if true_with_prob(epsi):
                a, pQall = tf.run([tga, tQall], feed_dict={tphio: phio})
            else:
                a = env.action_space.sample()

            pQoa = pQall[a]


        if po is not None:
            _ = tf.run([update_model], feed_dict={tphipo: phipo,
                                                  tpa: pa,
                                                  tpQoa: pQoa})

            _ = tf.run([update_elig])

        po    = o
        pa    = a
        phipo = phio

        o, r, d, _ = env.step(a)
