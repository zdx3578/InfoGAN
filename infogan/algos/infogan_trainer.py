from infogan.models.regularized_gan import RegularizedGAN
import prettytensor as pt
import tensorflow as tf
import numpy as np
from progressbar import ETA, Bar, Percentage, ProgressBar
from infogan.misc.distributions import Bernoulli, Gaussian, Categorical
import sys
from infogan.misc.utils import *
from infogan.misc.utilsdcgan import *

from glob import glob
#import MA
import dateutil
import dateutil.tz
import datetime



TINY = 1e-8


class InfoGANTrainer(object):
    def __init__(self,
                 model,
                 batch_size,
                 dataset=None,
                 exp_name="experiment",
                 log_dir="logs",
                 checkpoint_dir="ckt",
                 restore_checkpoint_file="tmp",
                 isrestore=False,
                 max_epoch=100,
                 updates_per_epoch=100,
                 snapshot_interval=4000,
                 ganlp=1,
                 info_reg_coeff=1.0,
                 discriminator_learning_rate=2e-4,
                 generator_learning_rate=2e-4,
                 ):
        """
        :type model: RegularizedGAN
        """
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.exp_name = exp_name
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.restore_checkpoint_file = restore_checkpoint_file
        self.isrestore = isrestore

        self.snapshot_interval = snapshot_interval
        self.updates_per_epoch = updates_per_epoch
        self.generator_learning_rate = generator_learning_rate
        self.discriminator_learning_rate = discriminator_learning_rate
        self.info_reg_coeff = info_reg_coeff
        self.discriminator_trainer = None
        self.generator_trainer = None
        self.input_tensor = None
        self.log_vars = []
        self.ganlp = ganlp




    def init_opt(self):
        #self.input_tensor = input_tensor = tf.placeholder(tf.float32, [self.batch_size, self.dataset.image_dim])
        #self.images = tf.placeholder(tf.float32, [self.batch_size, self.image_size,self.image_size,1])

        self.images = tf.placeholder(tf.float32, [self.batch_size, self.dataset.image_size, self.dataset.image_size, self.dataset.c_dim ])


        pstr('self.input_tensor',self.input_tensor)
        
        with pt.defaults_scope(phase=pt.Phase.train):
            z_var = self.model.latent_dist.sample_prior(self.batch_size)
            pstr('0 batch_size',self.batch_size)
            pstr('1 z_var',z_var)
            #print("1 %d | " % z_var )
            fake_x, _ = self.model.generate(z_var)
            pstr('1.1 fake_x',fake_x)
            pstr('1.1 self.images',self.images)

            real_d, _, _, _, real_d_log = self.model.discriminate(self.images)
            fake_d, sample, fake_reg_z_dist_info, reg_dist_flat, fake_d_log  = self.model.discriminate(fake_x)
            
            #d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(real_d_log, tf.ones_like(real_d)))
            #d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fake_d_log, tf.zeros_like(fake_d)))
            #discriminator_loss = d_loss_real + d_loss_fake
            #generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fake_d_log, tf.ones_like(fake_d)))


            pstr('1.1.2 sample',sample)
            pstr('1.1.2 fake_reg_z_dist_info',fake_reg_z_dist_info)
            pstr('1.1.2 reg_dist_flat',reg_dist_flat)


            pstr('1.1 fake_d',fake_d)
            pstr('1.5 fake_reg_z_dist_info',fake_reg_z_dist_info)

            reg_z = self.model.reg_z(z_var)
            pstr('2 reg_z',reg_z)

            discriminator_loss = - tf.reduce_mean(tf.log(real_d + TINY) + tf.log(1. - fake_d + TINY))
            generator_loss = - tf.reduce_mean(tf.log(fake_d + TINY))

            self.log_vars.append(("discriminator_loss", discriminator_loss))
            self.log_vars.append(("generator_loss", generator_loss))

            mi_est = tf.constant(0.)
            cross_ent = tf.constant(0.)



            # compute for discrete and continuous codes separately
            # discrete:
            if len(self.model.reg_disc_latent_dist.dists) > 0:
                disc_reg_z = self.model.disc_reg_z(reg_z)
                pstr('3 disc_reg_z',disc_reg_z)
                disc_reg_dist_info = self.model.disc_reg_dist_info(fake_reg_z_dist_info)
                pstr('4 disc_reg_dist_info',disc_reg_dist_info)
                disc_log_q_c_given_x = self.model.reg_disc_latent_dist.logli(disc_reg_z, disc_reg_dist_info)
                pstr('5 disc_log_q_c_given_x',disc_log_q_c_given_x)
                disc_log_q_c = self.model.reg_disc_latent_dist.logli_prior(disc_reg_z)
                pstr('6 disc_log_q_c',disc_log_q_c)
                disc_cross_ent = tf.reduce_mean(-disc_log_q_c_given_x)
                pstr('7 disc_cross_ent',disc_cross_ent)
                disc_ent = tf.reduce_mean(-disc_log_q_c)
                disc_mi_est = disc_ent - disc_cross_ent
                mi_est += disc_mi_est
                cross_ent += disc_cross_ent
                self.log_vars.append(("MI_disc", disc_mi_est))
                self.log_vars.append(("CrossEnt_disc", disc_cross_ent))
                discriminator_loss -= 10 * disc_mi_est
                generator_loss -= self.info_reg_coeff * disc_mi_est

            if len(self.model.reg_cont_latent_dist.dists) > 0:
                cont_reg_z = self.model.cont_reg_z(reg_z)
                pstr('8 cont_reg_z',cont_reg_z)
                cont_reg_dist_info = self.model.cont_reg_dist_info(fake_reg_z_dist_info)
                pstr('9 cont_reg_dist_info', cont_reg_dist_info)
                cont_log_q_c_given_x = self.model.reg_cont_latent_dist.logli(cont_reg_z, cont_reg_dist_info)
                pstr('10 cont_log_q_c_given_x', cont_log_q_c_given_x)
                cont_log_q_c = self.model.reg_cont_latent_dist.logli_prior(cont_reg_z)
                pstr('11 cont_log_q_c',cont_log_q_c)
                cont_cross_ent = tf.reduce_mean(-cont_log_q_c_given_x)
                cont_ent = tf.reduce_mean(-cont_log_q_c)
                cont_mi_est = cont_ent - cont_cross_ent
                mi_est += cont_mi_est
                cross_ent += cont_cross_ent
                self.log_vars.append(("MI_cont", cont_mi_est))
                self.log_vars.append(("CrossEnt_cont", cont_cross_ent))
                discriminator_loss -= 10 * cont_mi_est
                generator_loss -= self.info_reg_coeff * cont_mi_est

            pstr('1.1 generator_loss',generator_loss)

            for idx, dist_info in enumerate(self.model.reg_latent_dist.split_dist_info(fake_reg_z_dist_info)):
                if "stddev" in dist_info:
                    self.log_vars.append(("max_std_%d" % idx, tf.reduce_max(dist_info["stddev"])))
                    self.log_vars.append(("min_std_%d" % idx, tf.reduce_min(dist_info["stddev"])))

            self.log_vars.append(("MI", mi_est))
            self.log_vars.append(("CrossEnt", cross_ent))

            all_vars = tf.trainable_variables()
            d_vars = [var for var in all_vars if var.name.startswith('d_')]
            g_vars = [var for var in all_vars if var.name.startswith('g_')]

            pstr('1.1 g_vars',g_vars)


            self.log_vars.append(("max_real_d", tf.reduce_max(real_d)))
            self.log_vars.append(("min_real_d", tf.reduce_min(real_d)))
            self.log_vars.append(("max_fake_d", tf.reduce_max(fake_d)))
            self.log_vars.append(("min_fake_d", tf.reduce_min(fake_d)))

            discriminator_optimizer = tf.train.AdamOptimizer(self.discriminator_learning_rate, beta1=0.1)
            self.discriminator_trainer = pt.apply_optimizer(discriminator_optimizer, losses=[discriminator_loss],
                                                            var_list=d_vars)

            generator_optimizer = tf.train.AdamOptimizer(self.generator_learning_rate, beta1=0.1,epsilon=1e-1024)
            self.generator_trainer = pt.apply_optimizer(generator_optimizer, losses=[generator_loss], var_list=g_vars)

            for k, v in self.log_vars:
                tf.scalar_summary(k, v)

        with pt.defaults_scope(phase=pt.Phase.test):
            with tf.variable_scope("model", reuse=True) as scope:
                self.visualize_all_factors()

    def visualize_all_factors(self):
        with tf.Session():
            fixed_noncat = np.concatenate([
                np.tile(
                    self.model.nonreg_latent_dist.sample_prior(10).eval(),
                    [10, 1]
                ),
                self.model.nonreg_latent_dist.sample_prior(self.batch_size - 100).eval(),
            ], axis=0)
            fixed_cat = np.concatenate([
                np.tile(
                    self.model.reg_latent_dist.sample_prior(10).eval(),
                    [10, 1]
                ),
                self.model.reg_latent_dist.sample_prior(self.batch_size - 100).eval(),
            ], axis=0)
            pshape('fixed_cat',fixed_cat)
            pshape('fixed_noncat',fixed_noncat)

        offset = 0
        for dist_idx, dist in enumerate(self.model.reg_latent_dist.dists):
            if isinstance(dist, Gaussian):
                assert dist.dim == 1, "Only dim=1 is currently supported"
                c_vals = []
                for idx in xrange(10):
                    c_vals.extend([-1.0 + idx * 2.0 / 9] * 10)
                #pshape('c_vals',c_vals)
                c_vals.extend([0.] * (self.batch_size - 100))
                #pshape('c_vals',c_vals)
                vary_cat = np.asarray(c_vals, dtype=np.float32).reshape((-1, 1))
                #pshape('vary_cat',vary_cat)
                cur_cat = np.copy(fixed_cat)
                cur_cat[:, offset:offset+1] = vary_cat
                #pshape('cur_cat  Gaussian',cur_cat)
                offset += 1
            elif isinstance(dist, Categorical):
                lookup = np.eye(dist.dim, dtype=np.float32)
                cat_ids = []
                for idx in xrange(10):
                    cat_ids.extend([idx] * 10)

                cat_ids.extend([0] * (self.batch_size - 100))
                #pshape('cat_ids',cat_ids)
                cur_cat = np.copy(fixed_cat)
                cur_cat[:, offset:offset+dist.dim] = lookup[cat_ids]
                #pshape('cur_cat  Categorical',cur_cat)
                offset += dist.dim
            elif isinstance(dist, Bernoulli):
                assert dist.dim == 1, "Only dim=1 is currently supported"
                lookup = np.eye(dist.dim, dtype=np.float32)
                cat_ids = []
                for idx in xrange(10):
                    cat_ids.extend([int(idx / 5)] * 10)
                cat_ids.extend([0] * (self.batch_size - 100))
                cur_cat = np.copy(fixed_cat)
                cur_cat[:, offset:offset+dist.dim] = np.expand_dims(np.array(cat_ids), axis=-1)
                # import ipdb; ipdb.set_trace()
                offset += dist.dim
            else:
                raise NotImplementedError
            z_var = tf.constant(np.concatenate([fixed_noncat, cur_cat], axis=1))

            _, x_dist_info = self.model.generate(z_var)

            # just take the mean image
            if isinstance(self.model.output_dist, Bernoulli):
                img_var = x_dist_info["p"]
            elif isinstance(self.model.output_dist, Gaussian):
                img_var = x_dist_info["mean"]
            else:
                raise NotImplementedError
            img_var = self.dataset.inverse_transform(img_var)
            rows = 10
            img_var = tf.reshape(img_var, [self.batch_size] + list(self.dataset.image_shape))
            img_var = img_var[:rows * rows, :, :, :]
            imgs = tf.reshape(img_var, [rows, rows] + list(self.dataset.image_shape))
            stacked_img = []
            for row in xrange(rows):
                row_img = []
                for col in xrange(rows):
                    row_img.append(imgs[row, col, :, :, :])
                stacked_img.append(tf.concat(1, row_img))
            imgs = tf.concat(0, stacked_img)
            imgs = tf.expand_dims(imgs, 0)
            tf.image_summary("image_%d_%s" % (dist_idx, dist.__class__.__name__), imgs)


    def train(self):

        self.init_opt()

        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)

            summary_op = tf.merge_all_summaries()
            summary_writer = tf.train.SummaryWriter(self.log_dir, sess.graph)

            saver = tf.train.Saver()

            if  self.isrestore  and  self.restore_checkpoint_file:
                saver.restore(sess, self.restore_checkpoint_file)
                print("Model restored.")

            counter = 0
            #epochc = 0

            log_vars = [x for _, x in self.log_vars]
            log_keys = [x for x, _ in self.log_vars]

            for epoch in range(self.max_epoch):
                widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
                pbar = ProgressBar(maxval=self.updates_per_epoch, widgets=widgets)
                pbar.start()
                all_log_vals = []

                #pstr('ganlp',self.ganlp)
                for i in range(self.updates_per_epoch):
                    pbar.update(i)
                    batch_images  = self.dataset.next_batch(self.batch_size)
                    feed_dict={ self.images: batch_images}
                    log_vals = sess.run([self.discriminator_trainer] + log_vars, feed_dict)[1:]
                    print 
                    print '  '
                    print '  '
                    pstr("log_vals-d-run",log_vals)
                    #gencount=0
                    #for j in range(self.ganlp):
                    ganlpw=1
                    all_log_vals_G = []
                    while ganlpw :
                        log_vals_G = sess.run([self.generator_trainer] + log_vars, feed_dict)
                        ganlpw += 1
                        #pstr("log_vals_G",log_vals_G)
                        log_vals_G = log_vals_G[1:]
                        #pstr("log_vals_G",log_vals_G)
                        all_log_vals_G.append(log_vals_G)
                        #pstr("all_log_vals_G",all_log_vals_G)
                        avg_log_vals_G = np.mean(np.array(all_log_vals_G), axis=0)
                        log_dict_G = dict(zip(log_keys, avg_log_vals_G))
                        if not ganlpw%3 :
                            print ' . '
                            pstr('max_fake_d',log_dict_G['max_fake_d'])
                            log_line2 = "; ".join("%s: %s" % (str(k), str(v)) for k, v in zip(log_keys, avg_log_vals_G))
                            print("While_G  epoch -%d ganlpw -%d |  " % (epoch , ganlpw )) + log_line2
                            now = datetime.datetime.now(dateutil.tz.tzlocal())
                            timestamp = now.strftime('%Y_%m_%d : %H_%M_%S')
                            print timestamp
                        
                        ganlpw2 = ganlpw
                        if log_dict_G['max_fake_d'] > 0.5 or ganlpw > 300 :
                        #if ganlpw > 2 :
                            ganlpw = 0


                    print ' . '
                    pstr('max_fake_d',log_dict_G['max_fake_d'])
                    log_line2 = "; ".join("%s: %s" % (str(k), str(v)) for k, v in zip(log_keys, avg_log_vals_G))
                    print("While_G  ganlpw2 %d |  " % ( ganlpw2 )) + log_line2

                    now = datetime.datetime.now(dateutil.tz.tzlocal())
                    timestamp = now.strftime('%Y_%m_%d : %H_%M_%S')
                    pstr("ganlpw",ganlpw2)
                    print timestamp


                    all_log_vals.append(log_vals)
                    counter += 1
                    if counter % self.snapshot_interval == 0:
                        snapshot_name = "%s_%s" % (self.exp_name, str(counter))
                        fn = saver.save(sess, "%s/%s.ckpt" % (self.checkpoint_dir, snapshot_name))
                        print("Model saved in file: %s" % fn)


                batch_images  = self.dataset.next_batch(self.batch_size)

                summary_str = sess.run(summary_op, { self.images: batch_images})
                summary_writer.add_summary(summary_str, counter)

                avg_log_vals = np.mean(np.array(all_log_vals), axis=0)
                log_dict = dict(zip(log_keys, avg_log_vals))

                log_line = "; ".join("%s: %s" % (str(k), str(v)) for k, v in zip(log_keys, avg_log_vals))
                print("Epoch %d | " % (epoch) + log_line)
                sys.stdout.flush()
                #pstr2(log_dict)
                #pstr2(log_dict['max_fake_d'])
                if np.any(np.isnan(avg_log_vals)):
                    raise ValueError("NaN detected!")
