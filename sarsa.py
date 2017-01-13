# pylint: disable=all

update_elig = elig.assign_add( tf.mul(lmbda, elig) )

for n_episode in xrange(N_episodes):
    o    = env.reset()
    r    = 0
    done = False

    for n_step in xrange(N_max_steps):
        phio = tf.run([tphio], feed_dict={to: o})
        if not done:
            if true_with_prob(epsi):
                a, pQall = tf.run([tga, tQall], feed_dict={tphio: phio})
            else:
                a = env.action_space.sample()

            pQoa = pQall[a]



        _ = tf.run([update_model], feed_dict={tpphio: pphio,
                                              tpa: pa,
                                              tpQoa: pQoa})

        _ = tf.run([update_elig])


        po    = o
        pa    = a
        pphio = phio


        o, r, d, _ = env.step(a)

