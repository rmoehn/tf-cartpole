import itertools

import gym
import numpy as np
import tensorflow as tf

#### Helper functions

def true_with_prob(p):
    return np.random.rand(1)[0] < p


#### Parameters of the algorithm and training

alpha           = tf.constant(0.001, dtype=tf.float64)
lmbda           = tf.constant(0.9, dtype=tf.float64)
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

def phi(local_to, name=None):
    tnc_o = tf.div( tf.subtract(local_to, tlow), to_ranges_diff)
        # normalized, centered
    return tf.cos( tf.mul(tpi, tf.matmul(tC, tf.transpose(tnc_o))), name=name )


#### Set up variables for the algorithm

vtheta  = tf.Variable(tf.zeros([N_acts, N_weights_per_a], dtype=tf.float64))
velig   = tf.Variable(tf.zeros([N_acts, N_weights_per_a], dtype=tf.float64))


#### Set up placeholders for the algorithm

to      = tf.placeholder(tf.float64, shape=high.shape, name="to")
tpo     = tf.placeholder(tf.float64, shape=high.shape, name="tpo")

tr      = tf.placeholder(tf.float64, shape=[])

ta      = tf.placeholder(tf.int32, shape=[])
tpa     = tf.placeholder(tf.int32, shape=[])


#### Assemble the graph

tphio   = phi(to, name="to")
tphipo  = phi(tpo, name="tpo")

tQall       = tf.matmul(vtheta, tphio)
tga         = tf.squeeze( tf.argmax(tQall, axis=0) )

vthetaa     = tf.slice(vtheta, [tf.squeeze(ta), 0], [1, N_weights_per_a])
tpQoa       = tf.matmul(vthetaa, tphio, name='tpQoa')

vthetapa    = tf.slice(vtheta, [tf.squeeze(tpa), 0], [1, N_weights_per_a])
tpQpopa     = tf.matmul(vthetapa, tphipo, name='tpQpopa')

velig_a     = tf.slice(velig, [tf.squeeze(ta), 0], [1, N_weights_per_a])
update_elig = tf.scatter_update(velig, [ta],
                    tf.add(tf.mul(lmbda, velig_a), tf.squeeze(tphio)))

ptpQoa = tf.placeholder(tf.float64, shape=[1])
loss            = tf.mul(tf.sub(tpQpopa, tf.add(tr, ptpQoa)), velig_a)
optimizer       = tf.train.GradientDescentOptimizer(learning_rate=alpha)
update_model    = optimizer.minimize(loss)

init    = tf.global_variables_initializer()

#### Core algorithm

with tf.Session() as sess:
    file_writer = tf.summary.FileWriter("tf-logs", sess.graph)

    sess.run(init)

    for n_episode in xrange(N_episodes):
        po      = None
        phipo   = None
        pa      = None
        o       = env.reset()
        r       = 0
        done    = False

        n_step = 0
        for n_step in xrange(N_max_steps):
            phio = sess.run(tphio, feed_dict={to: o})
            if not done:
                ga, pQall = sess.run([tga, tQall], feed_dict={tphio: phio})
                if true_with_prob(epsi):
                    a = ga
                else:
                    a = env.action_space.sample()

                pQoa = pQall[a]
            else:
                a    = None
                phio = None
                pQoa = 0

            if po is not None:
                sess.run([update_model], feed_dict={tphipo: phipo,
                                                    ta: a,
                                                    tpa: pa,
                                                    ptpQoa: pQoa,
                                                    tr: r})

                sess.run([update_elig], feed_dict={ta: a, tphio: phio})

            po    = o
            pa    = a
            phipo = phio

            o, r, done, _ = env.step(a)

            if done:
                break

        print n_step
