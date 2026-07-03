from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder

from octo.utils.spec import ModuleSpec

LIBERO_DATA_DIR = ("/home/islab/.cache/huggingface/hub/"
                   "datasets--openvla--modified_libero_rlds/snapshots/"
                   "6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551")


def get_config(config_string="libero_object_no_noops"):
    # config_string == RLDS suite dir name (one of the four LIBERO suites)
    suite = config_string or "libero_object_no_noops"
    mode = "full"
    task = "language_conditioned"

    FINETUNING_KWARGS = {
        "name": suite,
        "data_dir": LIBERO_DATA_DIR,
        "image_obs_keys": {"primary": "image", "wrist": "wrist_image"},
        "proprio_obs_key": "proprio",
        "language_key": "language_instruction",
        "action_proprio_normalization_type": "normal",
        "action_normalization_mask": [True, True, True, True, True, True, False],
        "standardize_fn": ModuleSpec.create(
            "octo.libero_transform:libero_dataset_transform"),
    }

    frozen_keys = None  # full fine-tune

    max_steps = FieldReference(50000)
    window_size = FieldReference(default=1)

    config = dict(
        pretrained_path=placeholder(str),
        pretrained_step=placeholder(int),
        batch_size=128,
        shuffle_buffer_size=100000,
        num_steps=max_steps,
        log_interval=100,
        eval_interval=1000000,   # effectively disable val/viz (no sim env needed)
        save_interval=10000,
        save_dir=placeholder(str),
        seed=42,
        wandb=dict(project="octo_libero", group=placeholder(str), entity=placeholder(str)),
        dataset_kwargs=FINETUNING_KWARGS,
        modality=task,
        finetuning_mode=mode,
        window_size=window_size,
        optimizer=dict(
            learning_rate=dict(name="cosine", init_value=0.0, peak_value=3e-4,
                               warmup_steps=2000, decay_steps=max_steps, end_value=0.0),
            weight_decay=0.01,
            clip_gradient=1.0,
            frozen_keys=frozen_keys,
            grad_accumulation_steps=None,
        ),
        val_kwargs=dict(val_shuffle_buffer_size=1000, num_val_batches=8),
        viz_kwargs=dict(eval_batch_size=128, trajs_for_metrics=100,
                        trajs_for_viz=8, samples_per_state=8),
    )

    goal_relabeling_strategy = None
    keep_image_prob = 0.0

    traj_transform_kwargs = dict(
        window_size=window_size,
        action_horizon=4,
        goal_relabeling_strategy=goal_relabeling_strategy,
        task_augment_strategy="delete_task_conditioning",
        task_augment_kwargs=dict(keep_image_prob=keep_image_prob),
    )
    workspace_augment_kwargs = dict(
        random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        random_saturation=[0.9, 1.1],
        random_hue=[0.05],
        augment_order=["random_resized_crop", "random_brightness",
                       "random_contrast", "random_saturation", "random_hue"],
    )
    wrist_augment_kwargs = dict(
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        random_saturation=[0.9, 1.1],
        random_hue=[0.05],
        augment_order=["random_brightness", "random_contrast",
                       "random_saturation", "random_hue"],
    )
    frame_transform_kwargs = dict(
        resize_size={"primary": (256, 256), "wrist": (128, 128)},
        image_augment_kwargs=dict(primary=workspace_augment_kwargs,
                                  wrist=wrist_augment_kwargs),
    )
    config["frame_transform_threads"] = 16
    config["traj_transform_kwargs"] = traj_transform_kwargs
    config["frame_transform_kwargs"] = frame_transform_kwargs
    return ConfigDict(config)
