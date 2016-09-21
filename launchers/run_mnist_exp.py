from __future__ import print_function
from __future__ import absolute_import
from infogan.misc.distributions import Uniform, Categorical, Gaussian, MeanBernoulli

import tensorflow as tf
import os
from infogan.misc.datasets import *
from infogan.models.regularized_gan import RegularizedGAN
from infogan.algos.infogan_trainer import InfoGANTrainer
from infogan.misc.utils import mkdir_p
import dateutil
import dateutil.tz
import datetime

if __name__ == "__main__":

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    root_log_dir = "logs/mnist"
    root_checkpoint_dir = "ckt/mnist"
    
    #!!!!!
    #batch_size = 128
    batch_size = 128
    #updates_per_epoch = 50
    updates_per_epoch = 3
    #max_epoch = 122
    max_epoch = 2
    #snapshot_interval = 5000
    snapshot_interval = 5
    ganlp=2
    #ganlp=4


    exp_name = "mnist_%s" % timestamp

    log_dir = os.path.join(root_log_dir, exp_name)
    checkpoint_dir = os.path.join(root_checkpoint_dir, exp_name)
    isrestore=False
    restore_checkpoint_file = os.path.join(root_checkpoint_dir, "mnist_2016_09_16_15_29_24/mnist_2016_09_16_15_29_24_1455.ckpt")

    mkdir_p(log_dir)
    mkdir_p(checkpoint_dir)

    dataset = CelebADataset()

    latent_spec = [
        (Uniform(62), False),
        (Categorical(10), True),
        (Uniform(1, fix_std=True), True),
        (Uniform(1, fix_std=True), True),
        (Uniform(1, fix_std=True), True),
        (Uniform(1, fix_std=True), True),
        (Uniform(1, fix_std=True), True),        
        (Uniform(1, fix_std=True), True),
        (Uniform(1, fix_std=True), True),
        (Uniform(1, fix_std=True), True),
#        (Uniform(1, fix_std=True), True),
#        (Uniform(1, fix_std=True), True),
#        (Uniform(1, fix_std=True), True),
#        (Uniform(1, fix_std=True), True),
#        (Uniform(1, fix_std=True), True),
#        (Uniform(1, fix_std=True), True),
#        (Uniform(1, fix_std=True), True),
#        (Uniform(1, fix_std=True), True),
#        (Uniform(1, fix_std=True), True),
#        (Uniform(1, fix_std=True), True),
    ]

    model = RegularizedGAN(
        output_dist=MeanBernoulli(dataset.image_dim),
        latent_spec=latent_spec,
        batch_size=batch_size,
        image_shape=dataset.image_shape,
        network_type="mnist",
    )

    algo = InfoGANTrainer(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        exp_name=exp_name,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        isrestore=isrestore,
        restore_checkpoint_file=restore_checkpoint_file,
        snapshot_interval=snapshot_interval,
        ganlp=ganlp,
        max_epoch=max_epoch,
        updates_per_epoch=updates_per_epoch,
        info_reg_coeff=1.0,
        generator_learning_rate=1e-3,
        discriminator_learning_rate=2e-4,
    )

    algo.train()
