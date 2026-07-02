import tensorflow as tf


def libero_dataset_transform(trajectory):
    """Map modified_libero_rlds -> Octo expected format.
    obs keys: image (primary), wrist_image (wrist), state (6D EEF) -> proprio; action 7-dim."""
    trajectory["observation"]["proprio"] = tf.cast(trajectory["observation"]["state"], tf.float32)
    trajectory["action"] = tf.cast(trajectory["action"], tf.float32)
    return trajectory
