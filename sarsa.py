import itertools
import sys

sys.path.append("../cartpole")

import gym
import matplotlib
matplotlib.use('GTK3Agg')
# pylint: disable=unused-import
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
# pylint: disable=unused-import
from tensorflow.python import debug as tf_debug
import pyrsistent

from hiora_cartpole import fourier_fa
from hiora_cartpole import linfa

#### Helper functions

def true_with_prob(p):
    return np.random.rand(1)[0] < p


#### Parameters of the algorithm and training

alpha           = tf.constant(0.001, dtype=tf.float64)
lmbda           = tf.constant(0.9, dtype=tf.float64)
epsi            = 0
fourier_order   = 3
N_episodes      = 10
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
    return tf.cos( tf.mul(tpi, tf.matmul(tC, tnc_o, transpose_b=True)),
                name=name )


#### Set up non-TF algorithm

four_n_weights, four_feature_vec = fourier_fa.make_feature_vec(o_ranges,
                                        n_acts=2, order=fourier_order)
experience = linfa.init(lmbda=0.9,
                    init_alpha=0.001,
                    is_use_alpha_bounds=False,
                    epsi=epsi,
                    feature_vec=four_feature_vec,
                    n_weights=four_n_weights,
                    act_space=env.action_space,
                    theta=None)

Celig = []
Ctheta = []

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
tpQoa       = tf.squeeze( tf.matmul(vthetaa, tphio, name='tpQoa') )

vthetapa    = tf.slice(vtheta, [tf.squeeze(tpa), 0], [1, N_weights_per_a])
tpQpopa     = tf.squeeze( tf.matmul(vthetapa, tphipo, name='tpQpopa') )

velig_a     = tf.slice(velig, [tf.squeeze(ta), 0], [1, N_weights_per_a])
#update_elig = tf.scatter_update(velig, [ta],
#                    lmbda * velig_a + tf.squeeze(tphio))
add_to_elig = tf.scatter_add(velig, [tpa], tf.transpose(tphipo))
degrade_elig = velig.assign(lmbda * velig)

update          = alpha * (tpQpopa - (tr + tpQoa)) * velig
update_theta    = tf.assign_sub(vtheta, update)


#### Core algorithm

Timestep = pyrsistent.immutable('o, a, phio')

Ttheta = []
Telig = []

def think(prev, o, r, done):
    elig, theta = sess.run([velig, vtheta])
    Ttheta.append(np.hstack(theta))
    Telig.append(np.hstack(elig))

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
        sess.run(add_to_elig, {tpa: prev.a, tphipo: prev.phio})
        pQpopa = sess.run(tpQpopa, {tpa: prev.a, tphipo: prev.phio})
        print "T", pQpopa, r, pQoa
        sess.run(update_theta, feed_dict={tphipo: prev.phio,
                                          tpa: prev.a,
                                          tpQoa: pQoa,
                                          tr: r})

        sess.run(degrade_elig)
        #if not done:
            #sess.run([update_elig], feed_dict={ta: a, tphio: phio})

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
            Celig.append(np.copy(experience.E))
            Ctheta.append(np.copy(experience.theta))
            action, previous = think(previous, observation, reward, is_done)

            experience, linfa_action = linfa.think(experience, observation,
                    reward, is_done)
            #if action != linfa_action:
                #print "different", n_step

            observation, reward, is_done, _ = env.step(action)

            if is_done:
                break

        wrapup(previous, observation, reward,
                done=(is_done and (n_step != N_max_steps - 1)))
        Celig.append(experience.E)
        Ctheta.append(experience.theta)
        experience = linfa.wrapup(experience, observation, reward,
                            done=(is_done and (n_step != N_max_steps - 1)))
        previous = None

        if n_episode % 10 == 0:
            summary = sess.run(merged_summary)
            summary_writer.add_summary(summary, n_episode)

        #print n_step

    #theta = sess.run(vtheta)
    #plt.plot(np.hstack(theta))
    #plt.show()

    #summary = sess.run(merged_summary)
    #summary_writer.add_summary(merged_summary, 0)

nCelig = np.array(Celig)
nTelig = np.array(Telig)
nCtheta = np.array(Ctheta)
nTtheta = np.array(Ttheta)
