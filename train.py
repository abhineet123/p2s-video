import os
import shutil
from absl import logging
import time
import numpy as np

import utils
from models import model_utils


def run(cfg, train_datasets, val_datasets, tasks, train_steps, val_steps, steps_per_epoch, num_train_examples,
        strategy, model_lib, tf):
    if cfg.train.pt:
        assert cfg.pretrained, "cfg.pretrained must be provided to load pt and continue training from pretrained model"
        cfg.model.pretrained_ckpt = cfg.pretrained

    # def single_val_step(examples):
    #     preprocessed_outputs = [
    #         t.preprocess_batched(e, training=False) for e, t in zip(examples, tasks)]
    #     val_outputs = [trainer.val_step(o, t) for o, t in zip(preprocessed_outputs, tasks)]
    #     return val_outputs

    with strategy.scope():
        trainer = model_lib.TrainerRegistry.lookup(cfg.model.name)(
            cfg, model_dir=cfg.model_dir,
            num_train_examples=num_train_examples, train_steps=train_steps)
        train_data_iters = [iter(dataset) for dataset in train_datasets]
        summary_writer = tf.summary.create_file_writer(cfg.model_dir)

        # model = trainer.model
        # trainable_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
        # non_trainable_params = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
        # total_params = trainable_params + non_trainable_params
        #
        # print('\n\n')
        # print(f'trainable_params: {trainable_params}')
        # print(f'non_trainable_params: {non_trainable_params}')
        # print(f'total_params: {total_params}')
        # print('\n\n')

        # exit()

        is_greater = lambda x, y: x > y
        is_smaller = lambda x, y: x < y
        is_better = dict(
            # loss=is_smaller,
            loss_notpad=is_smaller,
            # correct_pc=is_greater,
            accuracy_notpad=is_greater,
        )
        best_val_metrics = dict(
            # loss=np.inf,
            loss_notpad=np.inf,
            # correct_pc=0,
            accuracy_notpad=0,
        )

        val_ckpt_managers = {
            metric_name: tf.train.CheckpointManager(
                tf.train.Checkpoint(model=trainer.model), os.path.join(cfg.model_dir, f'best-val-{metric_name}'), 1)
            for metric_name in best_val_metrics.keys()
        }

        @tf.function
        def validate_multiple_steps(data_iterators):
            val_step = lambda xs, ts=tasks: trainer.val_step(xs, ts)

            progbar = None
            if cfg.eager:
                progbar = tf.keras.utils.Progbar(val_steps)

            for step_id in tf.range(val_steps):
                with tf.name_scope(''):
                    strategy.run(val_step, ([next(it) for it in data_iterators],))
                    # if val_outputs is not None:
                    #     val_outputs = [strategy.gather(o, axis=0) for o in val_outputs]
                    # loss_, loss_notpad_, correct_pc_, accuracy_notpad_ = val_outputs[0]
                    # loss = loss.write(step_id, tf.cast(loss_, tf.float32))
                    # loss_notpad = loss_notpad.write(step_id, tf.cast(loss_notpad_, tf.float32))
                    # correct_pc = correct_pc.write(step_id, tf.cast(correct_pc_, tf.float32))
                    # accuracy_notpad = accuracy_notpad.write(step_id, tf.cast(accuracy_notpad_, tf.float32))

                    if cfg.eager:
                        progbar.add(1)

            # return loss.stack(), loss_notpad.stack(), correct_pc.stack(), accuracy_notpad.stack()

        @tf.function
        def train_multiple_steps(data_iterators, tasks, progress_log_path):
            """
            ts=tasks is just specifying the default value for optional arg ts
            strategy is needed to get the num_replicas_in_sync to divide the gradient and compute its mean
            """
            train_step = lambda xs, ts=tasks: trainer.train_step(xs, ts, strategy)

            """
            train_steps = num_samples * num_epochs / batch_size
            """
            progbar = None
            if cfg.eager:
                progbar = tf.keras.utils.Progbar(steps_per_epoch)

            for step_id in tf.range(steps_per_epoch):  # using tf.range prevents unroll.
                with tf.name_scope(''):  # prevent `while_` prefix for variable names.
                    strategy.run(train_step, ([next(it) for it in data_iterators],))

                if not cfg.eager:
                    tf.print(step_id, ' / ', steps_per_epoch, output_stream=f"file://{progress_log_path}")
                    continue

                # global_step_ = trainer.optimizer.iterations.numpy()
                # global_step_ = global_step_.item()
                # if global_step_ >= 1:
                #     model = trainer.model
                #     trainable_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
                #     non_trainable_params = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
                #     total_params = trainable_params + non_trainable_params
                #
                #     print('\n\n')
                #     print(f'trainable_params: {trainable_params}')
                #     print(f'non_trainable_params: {non_trainable_params}')
                #     print(f'total_params: {total_params}')
                #     print('\n\n')

                if step_id == 0 and cfg.train.check_ckpt:
                    model_utils.check_ckpt_match(cfg.model_dir, trainer._model,
                                                 trainer.ckpt_vars_p)
                    # trainer.check_checkpoint_restored()

                if cfg.debug:
                    trainer.sample_to_tb()

                progbar.add(1)

        global_step = trainer.optimizer.iterations
        cur_step = global_step.numpy()
        timestamp = time.time()
        cur_epoch = 0
        # if not cfg.eager:
        #     print('compiling graph...')
        # trainer.checkpoint_manager.save(cur_step)
        # ckpt_vars_0 = utils.save_ckpt_vars(cfg.model_dir)
        is_seg = 'segmentation' in cfg.task.name

        if is_seg:
            rle_lens = cfg.dataset.train.rle_lens
            rle_lens_str = '\n'.join(rle_lens)
            rle_lens_path = os.path.join(cfg.model_dir, "rle_lens.txt")
            with open(rle_lens_path, 'w') as fid:
                fid.write(rle_lens_str)

        best_val_metrics_json = os.path.join(cfg.model_dir, "best_val_metrics.json")

        if os.path.exists(best_val_metrics_json):
            print(f'loading best_val_metrics from {best_val_metrics_json}')
            import json

            with open(best_val_metrics_json, 'r') as f:
                best_val_metrics = json.loads(f.read())
            print(f'best_val_metrics:\n{best_val_metrics}')

        from tqdm import tqdm
        n_epochs = train_steps // steps_per_epoch

        model_name = os.path.basename(cfg.model_dir)
        pbar = tqdm(total=n_epochs, desc=model_name)

        progress_log_path = os.path.join(cfg.model_dir, "progress_log.txt")
        if not cfg.eager:
            print(f"\n\nuse this to monitor the intra-epoch training progress:")
            print(f"watch tail -1 {progress_log_path}\n\n")

        while cur_step < train_steps:
            cur_epoch += 1
            # tf.print(f'Training epoch {cur_epoch} with {steps_per_epoch} steps...')
            with summary_writer.as_default():
                train_multiple_steps(train_data_iters, tasks, progress_log_path)
                os.remove(progress_log_path)

                """
                this check happens after the first forward pass because of deferred restoration 
                in tf.train.Checkpoint
                which only restores many of the variables after they are created in the first call 
                when the input shape becomes available
                """
                # if cfg.train.check_ckpt:
                #     trainer.check_checkpoint_restored()

                cur_step = global_step.numpy()

                trainer.checkpoint_manager.save(cur_step)

                steps_per_sec = steps_per_epoch / (time.time() - timestamp)
                timestamp = time.time()
                train_metrics_dict = {}
                with tf.name_scope('train'):
                    nan_metric = 0
                    for metric_name, metric_val in trainer.metrics.items():
                        metric_val_np = metric_val.result().numpy()
                        train_metrics_dict[metric_name] = metric_val_np

                        if np.isnan(metric_val_np):
                            logging.error(f'NaN value found for training {metric_name} so terminating training')
                            nan_metric = 1
                            break
                        tf.summary.scalar(metric_name, metric_val_np, global_step)
                    if nan_metric:
                        break
                    lr = trainer.learning_rate(tf.cast(global_step, dtype=tf.float32))
                    tf.summary.scalar('lr', lr, global_step)
                    tf.summary.scalar('steps_per_sec', steps_per_sec, global_step)

                if cfg.train.val_epochs and cur_epoch % cfg.train.val_epochs == 0:
                    tf.print(f'validating epoch {cur_epoch} with {val_steps} steps...')

                    val_data_iters = [iter(dataset) for dataset in val_datasets]

                    validate_multiple_steps(val_data_iters)

                    with tf.name_scope('val'):

                        val_metrics_dict = dict()
                        nan_metric = 0
                        for metric_name, metric_val in trainer.val_metrics.items():
                            metric_val_np = metric_val.result().numpy().item()
                            val_metrics_dict[metric_name] = metric_val_np
                            if np.isnan(metric_val_np):
                                logging.error(f'NaN value found for validation {metric_name} so terminating training')
                                nan_metric = 1
                                break

                            tf.summary.scalar(metric_name, metric_val_np, global_step)

                            try:
                                best_metric_val = best_val_metrics[metric_name]
                            except KeyError:
                                continue
                            else:
                                if is_better[metric_name](metric_val_np, best_metric_val):
                                    import json
                                    print(f'found better validation {metric_name}: {metric_val_np}')
                                    best_val_metrics[metric_name] = metric_val_np
                                    val_ckpt_managers[metric_name].save(cur_step)
                                    with open(best_val_metrics_json, 'w') as f:
                                        f.write(json.dumps(best_val_metrics, indent=4))
                        if nan_metric:
                            break

                summary_writer.flush()
            pbar.update(1)
            pbar.set_description(f'{model_name} {cur_step}')

            # progress = cur_step / float(train_steps) * 100
            # eta = (train_steps - cur_step) / steps_per_sec / 60.
            # logging.info(f'Completed steps {cur_step} / {train_steps} ({progress:.2f}%), ETA {eta:.2f} mins')
            trainer.reset()
        logging.info('###########################################')
        logging.info('Training complete...')
        logging.info('###########################################')
