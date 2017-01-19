import itertools

import gym
import matplotlib
matplotlib.use('GTK3Agg')
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import pyrsistent

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

vtheta  = tf.Variable(tf.zeros([N_acts, N_weights_per_a], dtype=tf.float64),
                        name="theta")
#tf.summary.histogram("vtheta", vtheta)
velig   = tf.Variable(tf.zeros([N_acts, N_weights_per_a], dtype=tf.float64))
tf.summary.histogram("velig", velig)


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

ptpQoa = tf.placeholder(tf.float64, shape=[], name='ptpQoa')
offset          = tf.mul(tf.sub(tpQpopa, tf.add(tr, ptpQoa)), velig)
update_theta    = tf.assign_sub(vtheta, tf.mul(alpha, offset))


#### Core algorithm

Timestep = pyrsistent.immutable('o, a, phio')

def think(prev, o, r, done):
    phio = sess.run(tphio, feed_dict={to: o})
    if not done:
        ga, pQall = sess.run([tga, tQall], feed_dict={tphio: phio})
        if true_with_prob(epsi):
            a = ga
        else:
            a = env.action_space.sample()

        pQoa = pQall[a][0]
    else:
        a    = None
        phio = None
        pQoa = 0

    if prev is not None:
        sess.run([update_theta], feed_dict={tphipo: prev.phio,
                                            tpa: prev.a,
                                            ptpQoa: pQoa,
                                            tr: r})

        if not done:
            sess.run([update_elig], feed_dict={ta: a, tphio: phio})

    return a, Timestep(o, a, phio)


def wrapup(prev, o, r, done=False):
    if done:
        think(prev, o, r, done=True)

    return None


with tf.Session() as sess:
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    summary_writer  = tf.summary.FileWriter("tf-logs", sess.graph)
    merged_summary  = tf.summary.merge_all()
    init            = tf.global_variables_initializer()
    sess.run(init)

    for n_episode in xrange(N_episodes):
        previous    = None
        observation = env.reset()
        reward      = 0
        is_done     = False

        n_step = 0
        for n_step in xrange(N_max_steps):
            action, previous = think(previous, observation, reward, is_done)

            observation, reward, is_done, _ = env.step(action)

            if is_done:
                break

        wrapup(previous, observation, reward,
                done=(is_done and (n_step != N_max_steps - 1)))
        previous = None

        if n_episode % 10 == 0:
            summary = sess.run(merged_summary)
            summary_writer.add_summary(summary, n_episode)

        print n_step

    #theta = sess.run(vtheta)
    #plt.plot(np.hstack(theta))
    #plt.show()

    #summary = sess.run(merged_summary)
    #summary_writer.add_summary(merged_summary, 0)

