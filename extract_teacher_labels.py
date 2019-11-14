from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from bert import modeling
import collections
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "output_file", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer(
    "max_predictions_per_seq", 20,
    "Maximum number of masked LM predictions per sequence. "
    "Must match data generation.")


flags.DEFINE_integer("batch_size", 32, "Total batch size.")


def _decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.to_int32(t)
    example[name] = t

  return example


def produce_dataset(input_files,
                    max_seq_length,
                    max_predictions_per_seq,
                    batch_size,
                    num_cpu_threads=4):
    name_to_features = {
            "input_ids":
                tf.FixedLenFeature([max_seq_length], tf.int64),
            "input_mask":
                tf.FixedLenFeature([max_seq_length], tf.int64),
            "segment_ids":
                tf.FixedLenFeature([max_seq_length], tf.int64),
            "masked_lm_positions":
                tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_ids":
                tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_weights":
                tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
            "next_sentence_labels":
                tf.FixedLenFeature([1], tf.int64),
        }

    d = tf.data.TFRecordDataset(input_files)
    d = d.repeat(1)
    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=False))
    return d


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions):
  """Get loss and log probs for the masked LM."""
  input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        "output_bias",
        shape=[bert_config.vocab_size],
        initializer=tf.zeros_initializer())
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    masked_lm_probs = tf.nn.softmax(logits,axis=-1)

  return masked_lm_probs


def get_next_sentence_output(bert_config, input_tensor):
  """Get loss and log probs for the next sentence prediction."""

  # Simple binary classification. Note that 0 is "next sentence" and 1 is
  # "random sentence". This weight matrix is not used after pre-training.
  with tf.variable_scope("cls/seq_relationship"):
    output_weights = tf.get_variable(
        "output_weights",
        shape=[2, bert_config.hidden_size],
        initializer=modeling.create_initializer(bert_config.initializer_range))
    output_bias = tf.get_variable(
        "output_bias", shape=[2], initializer=tf.zeros_initializer())

    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    next_sentence_probs = tf.nn.softmax(logits, axis=-1)

    return next_sentence_probs


def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)

  return output_tensor


def model_fn(features, bert_config, init_checkpoint):  # pylint: disable=unused-argument

    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    masked_lm_positions = features["masked_lm_positions"]

    model = modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False)

    masked_lm_log_probs = get_masked_lm_output(bert_config,
                                               model.get_sequence_output(),
                                               model.get_embedding_table(),
                                               masked_lm_positions)

    tvars = tf.trainable_variables()

    initialized_variable_names = {}

    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)
    return masked_lm_log_probs

def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info("*** Input Files ***")
    for input_file in input_files:
        tf.logging.info("  %s" % input_file)

    my_data = produce_dataset(input_files=input_files,
                              max_seq_length=FLAGS.max_seq_length,
                              max_predictions_per_seq=FLAGS.max_predictions_per_seq,
                              batch_size=FLAGS.batch_size,
                              num_cpu_threads=4)
    my_data = my_data.make_one_shot_iterator()
    text_feature = my_data.get_next()
    count =0

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    masked_lm_prob = model_fn(text_feature, bert_config, FLAGS.init_checkpoint)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.python_io.TFRecordWriter(FLAGS.output_file)

        while True:
            try:
                feat, mask_lm_prob = sess.run([text_feature, masked_lm_prob])

                input_ids = feat["input_ids"]
                input_mask = feat["input_mask"]
                segment_ids = feat["segment_ids"]
                masked_lm_positions = feat["masked_lm_positions"]
                masked_lm_ids = feat["masked_lm_ids"]
                masked_lm_weights = feat["masked_lm_weights"]
                next_sentence_labels = feat["next_sentence_labels"]

                cur_batch_size, seq_len = input_ids.shape
                # shape = min(batch_size,num_examples)*pred_per_seq, vocab_size
                _, pred_per_seq = masked_lm_positions.shape
                new_feat = collections.OrderedDict()
                for i in range(cur_batch_size):
                    new_feat["input_ids"] = create_int_feature(input_ids[i])
                    new_feat["input_mask"] = create_int_feature(input_mask[i])
                    new_feat["segment_ids"] = create_int_feature(segment_ids[i])
                    new_feat["masked_lm_positions"] = create_int_feature(masked_lm_positions[i])
                    new_feat["masked_lm_ids"] = create_int_feature(masked_lm_ids[i])
                    new_feat["masked_lm_weights"] = create_float_feature(masked_lm_weights[i])
                    new_feat["next_sentence_labels"] = create_int_feature([next_sentence_labels[i][0]])

                    #retrieve predictions for (pred_per_seq) number of words, and flatten
                    distribution = mask_lm_prob[i*pred_per_seq:(i+1)*pred_per_seq].reshape(-1)
                    new_feat["masked_lm_probs"] = create_float_feature(distribution)

                    tf_example = tf.train.Example(features=tf.train.Features(feature=new_feat))
                    writer.write(tf_example.SerializeToString())
                    count += 1

            except tf.errors.OutOfRangeError:
                print("we outtie")
                break
        print("closing writer")
        print("")
        print("")
        print("Number of Sequences Written:")
        print(count)

        writer.close()


if __name__ =="__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_file")
    flags.mark_flag_as_required("bert_config_file")
    tf.app.run()

