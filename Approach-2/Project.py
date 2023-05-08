import tensorflow as tf
import numpy as np

import sys
import os
import argparse
import json
# from scipy.misc import imsave
from matplotlib.pyplot import imsave
# from keras.preprocessing.image import save_img

from tf_dataset_hw import *
from source.tf_models import VRNNGMM
from tf_models_hw import HandwritingVRNNGmmModel, HandwritingVRNNModel
from source.utils_visualization import plot_latent_variables, plot_latent_categorical_variables, plot_matrix_and_get_image, plot_and_get_image
import visualize_hw as visualize

# Sampling options
run_original_sample = True  # Save an image of reference samples (see reference_sample_ids).
run_reconstruction = False  # Reconstruct reference samples and save reconstruction results.
run_biased_sampling = True  # Use a real reference sample to infer style (see reference_sample_ids) and synthesize the given text (see conditional_texts).

# Sampling hyper-parameters
eoc_threshold = 0.05
cursive_threshold = 0.51
seq_len = 1600  # Maximum number of steps.

# Text to be written by the model.
conditional_texts = ["Spider-Man, does whatever a spider can."]
# Indices of reference style samples from validation split.
reference_sample_ids = [95,32,124]
# Concatenate reference sample with synthetic sample to make a direct comparison.
concat_ref_and_synthetic_samples = True

# Sampling output options
plot_eoc = False  # Plot end-of-character probabilities.
plot_latent_vars = False  # Plot a matrix of approximate posterior and prior mu values.
save_plots = True  # Save plots as image.
show_plots = True  # Show plots in a window.


def plot_eval_details(data_dict, sample, save_dir, save_name):
    visualize.draw_stroke_svg(sample, factor=0.001, svg_filename=os.path.join(save_dir, save_name + '.svg'))

    plot_data = {}

    if plot_latent_vars and 'p_mu' in data_dict:
        plot_data['p_mu'] = np.transpose(data_dict['p_mu'][0], [1, 0])
        plot_data['q_mu'] = np.transpose(data_dict['q_mu'][0], [1, 0])
        plot_data['q_sigma'] = np.transpose(data_dict['q_sigma'][0], [1, 0])
        plot_data['p_sigma'] = np.transpose(data_dict['p_sigma'][0], [1, 0])

        plot_img = plot_latent_variables(plot_data, show_plot=show_plots)
        if save_plots:
            imsave(os.path.join(save_dir, save_name + '_normal.png'), plot_img)

    if plot_latent_vars and 'p_pi' in data_dict:
        plot_data['p_pi'] = np.transpose(data_dict['p_pi'][0], [1, 0])
        plot_data['q_pi'] = np.transpose(data_dict['q_pi'][0], [1, 0])
        plot_img = plot_latent_categorical_variables(plot_data, show_plot=show_plots)
        if save_plots:
            imsave(os.path.join(save_dir, save_name + '_pi.png'), plot_img)

    if plot_eoc and 'out_eoc' in data_dict:
        plot_img = plot_and_get_image(np.squeeze(data_dict['out_eoc']))
        imsave(os.path.join(save_dir, save_name + '_eoc.png'), plot_img)

    # Same for every sample.
    if 'gmm_mu' in data_dict:
        gmm_mu_img = plot_matrix_and_get_image(data_dict['gmm_mu'])
        gmm_sigma_img = plot_matrix_and_get_image(data_dict['gmm_sigma'])
        if save_plots:
            imsave(os.path.join(save_dir, 'gmm_mu.png'), gmm_mu_img)
            imsave(os.path.join(save_dir, 'gmm_sigma.png'), gmm_sigma_img)

    return plot_data


def do_evaluation(config):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    Model_cls = getattr(sys.modules[__name__], config['model_cls'])
    Dataset_cls = getattr(sys.modules[__name__], config['dataset_cls'])

    batch_size = 1
    data_sequence_length = None
    # Load validation dataset to fetch statistics.
    if issubclass(Dataset_cls, HandWritingDatasetConditional):
        validation_dataset = Dataset_cls(config['validation_data'], var_len_seq=True, use_bow_labels=config['use_bow_labels'])
    elif issubclass(Dataset_cls, HandWritingDataset):
        validation_dataset = Dataset_cls(config['validation_data'], var_len_seq=True)
    else:
        raise Exception("Unknown dataset class.")

    strokes = tf.compat.v1.placeholder(tf.float32, shape=[batch_size, data_sequence_length, sum(validation_dataset.input_dims)])
    targets = tf.compat.v1.placeholder(tf.float32, shape=[batch_size, data_sequence_length, sum(validation_dataset.target_dims)])
    sequence_length = tf.compat.v1.placeholder(tf.int32, shape=[batch_size])

    # Create inference graph.
    with tf.name_scope("validation"):
        inference_model = Model_cls(config,
                                    reuse=False,
                                    input_op=strokes,
                                    target_op=targets,
                                    input_seq_length_op=sequence_length,
                                    input_dims=validation_dataset.input_dims,
                                    target_dims=validation_dataset.target_dims,
                                    batch_size=batch_size,
                                    mode="validation",
                                    data_processor=validation_dataset)
        inference_model.build_graph()
        inference_model.create_image_summary(validation_dataset.prepare_for_visualization)

    # Create sampling graph.
    with tf.name_scope("sampling"):
        model = Model_cls(config,
                          reuse=True,
                          input_op=strokes,
                          target_op=None,
                          input_seq_length_op=sequence_length,
                          input_dims=validation_dataset.input_dims,
                          target_dims=validation_dataset.target_dims,
                          batch_size=batch_size,
                          mode="sampling",
                          data_processor=validation_dataset)
        model.build_graph()

    # Create a session object and initialize parameters.
    sess = tf.compat.v1.Session()
    # Restore computation graph.
    try:
        saver = tf.compat.v1.train.Saver()
        # Restore variables.
        if config['checkpoint_id'] is None:
            checkpoint_path = tf.train.latest_checkpoint(config['model_dir'])
        else:
            checkpoint_path = os.path.join(config['model_dir'], config['checkpoint_id'])

        print("Loading model " + checkpoint_path)
        saver.restore(sess, checkpoint_path)
    except:
        raise Exception("Model is not found.")

    
    kargs = dict()
    kargs['conditional_inputs'] = None
    kargs['eoc_threshold'] = eoc_threshold
    kargs['cursive_threshold'] = cursive_threshold
    kargs['use_sample_mean'] = True

    print("Generating samples...")
    for real_img_idx in reference_sample_ids:
        _, stroke_model_input, _ = validation_dataset.fetch_sample(real_img_idx)
        stroke_sample = stroke_model_input[:, :, 0:3]

        if run_biased_sampling:
            inference_results = inference_model.reconstruct_given_sample(session=sess, inputs=stroke_model_input)

        if run_original_sample:
            svg_path = os.path.join(config['eval_dir'], "real_image_"+str(real_img_idx)+'.svg')
            visualize.draw_stroke_svg(validation_dataset.undo_normalization(validation_dataset.samples[real_img_idx], detrend_sample=False), factor=0.001, svg_filename=svg_path)

        if concat_ref_and_synthetic_samples:
            reference_sample_in_img = stroke_sample
        else:
            reference_sample_in_img = None

        # Conditional handwriting synthesis.

        for text_id, text in enumerate(conditional_texts):
            kargs['conditional_inputs'] = text
            if config.get('use_real_pi_labels', False) and isinstance(model, VRNNGMM):
                if run_biased_sampling:
                    biased_sampling_results = model.sample_biased(session=sess, seq_len=seq_len,
                                                                    prev_state=inference_results[0]['state'],
                                                                    prev_sample=reference_sample_in_img,
                                                                    **kargs)

                    save_name = 'synthetic_biased_ref(' + str(real_img_idx) + ')_(' + str(text_id) + ')'
                    synthetic_sample = validation_dataset.undo_normalization(biased_sampling_results[0]['output_sample'][0], detrend_sample=False)
                    if save_plots:
                        plot_eval_details(biased_sampling_results[0], synthetic_sample, config['eval_dir'], save_name)

                    # Without beautification: set False
                    # Apply beautification: set True.
                    kargs['use_sample_mean'] = True
                    biased_sampling_results = model.sample_biased(session=sess, seq_len=seq_len,
                                                                    prev_state=inference_results[0]['state'],
                                                                    prev_sample=reference_sample_in_img,
                                                                    **kargs)

                    save_name = 'synthetic_biased_sampled_ref(' + str(real_img_idx) + ')_(' + str(text_id) + ')'
                    synthetic_sample = validation_dataset.undo_normalization(biased_sampling_results[0]['output_sample'][0], detrend_sample=False)
                    if save_plots:
                        plot_eval_details(biased_sampling_results[0], synthetic_sample, config['eval_dir'], save_name)

            else:
                if run_biased_sampling:
                    biased_sampling_results = model.sample_biased(session=sess, seq_len=seq_len,
                                                                    prev_state=inference_results[0]['state'],
                                                                    prev_sample=reference_sample_in_img)

                    save_name = 'synthetic_biased_ref(' + str(real_img_idx) + ')_(' + str(text_id) + ')'
                    synthetic_sample = validation_dataset.undo_normalization(biased_sampling_results[0]['output_sample'][0], detrend_sample=False)
                    if save_plots:
                        plot_eval_details(biased_sampling_results[0], synthetic_sample, config['eval_dir'], save_name)

    sess.close()


if __name__ == '__main__':
    config_dict = json.load(open ('./model/config.json', 'r'))
    config_dict['model_dir'] = "./model"
    config_dict['checkpoint_id'] = None
    config_dict['eval_dir'] = './results\\./model'
    
    if not os.path.exists(config_dict['eval_dir']):
        os.makedirs(config_dict['eval_dir'])

    do_evaluation(config_dict)