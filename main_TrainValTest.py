"""
Copyright (C) 2022 King Saud University, Saudi Arabia
SPDX-License-Identifier: Apache-2.0

Replication of EEG-ATCNet main_TrainValTest.py (Train-Val-Test split).
Author: Hamdi Altaheri
"""

import os
import sys
import shutil
import time
import logging
import json
import signal
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

# Prefer GPU and log device (so 4070 etc. is used instead of CPU)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        _log = logging.getLogger(__name__)
        _log.info("GPU DETECTED: %s", [gpu.name for gpu in gpus])
        for i, gpu in enumerate(gpus):
            _log.info("  GPU %d: %s", i, gpu)
    except RuntimeError as e:
        _log = logging.getLogger(__name__)
        _log.warning("GPU config failed: %s", e)
else:
    _log = logging.getLogger(__name__)
    _log.warning("No GPU found; using CPU")
    _log.warning("  CUDA built: %s", tf.test.is_built_with_cuda())
    _log.warning("  Available devices: %s", tf.config.list_physical_devices())

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

import models
from preprocess import (
    get_data,
    load_user_preprocessed,
    load_bci2a_summary_npz,
    get_preprocessing_info_bci2a,
    standardize_fit_train_transform_train_val_test,
    standardize_fit_train_return_scalers,
    standardize_apply_scalers,
)


def draw_learning_curves(history, sub, results_path):
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy - subject: ' + str(sub))
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(os.path.join(results_path, 'subject_{}_acc.png'.format(sub)))
    plt.close()
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss - subject: ' + str(sub))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(os.path.join(results_path, 'subject_{}_loss.png'.format(sub)))
    plt.close()


def draw_confusion_matrix(cf_matrix, sub, results_path, classes_labels):
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=classes_labels)
    disp.plot()
    disp.ax_.set_xticklabels(classes_labels, rotation=12)
    plt.title('Confusion Matrix of Subject: ' + str(sub))
    plt.savefig(os.path.join(results_path, 'subject_' + str(sub) + '.png'))
    plt.close()


def draw_performance_barChart(num_sub, metric, label, results_path):
    fig, ax = plt.subplots()
    x = list(range(1, num_sub + 1))
    ax.bar(x, metric, 0.5, label=label)
    ax.set_ylabel(label)
    ax.set_xlabel("Subject")
    ax.set_xticks(x)
    ax.set_title('Model ' + label + ' per subject')
    ax.set_ylim([0, 1])
    plt.savefig(os.path.join(results_path, 'bar_' + label.replace(' ', '_') + '.png'))
    plt.close()


def train(dataset_conf, train_conf, results_path):
    if os.path.exists(results_path):
        shutil.rmtree(results_path)
    os.makedirs(results_path)
    in_exp = time.time()
    best_models = open(os.path.join(results_path, "best models.txt"), "w")
    log_write = open(os.path.join(results_path, "log.txt"), "w")
    dataset = dataset_conf.get('name')
    n_sub = dataset_conf.get('n_sub')
    data_path = dataset_conf.get('data_path')
    isStandard = dataset_conf.get('isStandard')
    LOSO = dataset_conf.get('LOSO', False)
    batch_size = train_conf.get('batch_size')
    epochs = train_conf.get('epochs')
    lr = train_conf.get('lr')
    LearnCurves = train_conf.get('LearnCurves', False)
    n_train = train_conf.get('n_train')
    model_name = train_conf.get('model')
    from_logits = train_conf.get('from_logits', False)
    acc = np.zeros((n_sub, n_train))
    kappa = np.zeros((n_sub, n_train))

    for sub in range(n_sub):
        print('\nTraining on subject ', sub + 1)
        log_write.write('\nTraining on subject ' + str(sub + 1) + '\n')
        BestSubjAcc = 0
        bestTrainingHistory = []
        X_train, _, y_train_onehot, _, _, _ = get_data(
            data_path, sub, dataset, LOSO=LOSO, isStandard=isStandard)
        X_train, X_val, y_train_onehot, y_val_onehot = train_test_split(
            X_train, y_train_onehot, test_size=0.2, random_state=42)

        for train_run in range(n_train):
            tf.random.set_seed(train_run + 1)
            np.random.seed(train_run + 1)
            in_run = time.time()
            filepath = os.path.join(results_path, 'saved models', 'run-{}'.format(train_run + 1))
            os.makedirs(filepath, exist_ok=True)
            filepath = os.path.join(filepath, 'subject-{}.h5'.format(sub + 1))
            model = getModel(model_name, dataset_conf, from_logits)
            model.compile(
                loss=CategoricalCrossentropy(from_logits=from_logits),
                optimizer=Adam(learning_rate=lr),
                metrics=['accuracy'])
            callbacks = [
                ModelCheckpoint(filepath, monitor='val_loss', verbose=0,
                               save_best_only=True, save_weights_only=True, mode='min'),
                ReduceLROnPlateau(monitor="val_loss", factor=0.90, patience=20, verbose=0, min_lr=0.0001),
            ]
            history = model.fit(
                X_train, y_train_onehot,
                validation_data=(X_val, y_val_onehot),
                epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1)
            model.load_weights(filepath)
            y_pred = model.predict(X_val)
            if from_logits:
                y_pred = tf.nn.softmax(y_pred).numpy().argmax(axis=-1)
            else:
                y_pred = y_pred.argmax(axis=-1)
            labels = y_val_onehot.argmax(axis=-1)
            acc[sub, train_run] = accuracy_score(labels, y_pred)
            kappa[sub, train_run] = cohen_kappa_score(labels, y_pred)
            out_run = time.time()
            info = 'Subject: {} seed {} time: {:.1f} m valid_acc: {:.4f} valid_loss: {:.3f}'.format(
                sub + 1, train_run + 1, (out_run - in_run) / 60, acc[sub, train_run],
                min(history.history['val_loss']))
            print(info)
            log_write.write(info + '\n')
            if BestSubjAcc < acc[sub, train_run]:
                BestSubjAcc = acc[sub, train_run]
                bestTrainingHistory = history
        best_run = np.argmax(acc[sub, :])
        best_models.write('/saved models/run-{}/subject-{}.h5\n'.format(best_run + 1, sub + 1))
        if LearnCurves and bestTrainingHistory:
            draw_learning_curves(bestTrainingHistory, sub + 1, results_path)

    out_exp = time.time()
    head1 = head2 = ' '
    for sub in range(n_sub):
        head1 = head1 + 'sub_{} '.format(sub + 1)
        head2 = head2 + '----- '
    head1 = head1 + ' average'
    head2 = head2 + ' -------'
    info = '\n---------------------------------\nValidation performance (acc %):\n' + head1 + '\n' + head2
    for run in range(n_train):
        info = info + '\nSeed {}: '.format(run + 1)
        for sub in range(n_sub):
            info = info + '{:.2f} '.format(acc[sub, run] * 100)
        info = info + ' {:.2f} '.format(np.average(acc[:, run]) * 100)
    info = info + '\n---------------------------------\nAverage acc - all seeds: '
    info = info + '{:.2f} %\n\nTrain Time - all seeds: {:.1f} min\n---------------------------------\n'.format(
        np.average(acc) * 100, (out_exp - in_exp) / 60)
    print(info)
    log_write.write(info + '\n')
    best_models.close()
    log_write.close()


def test(model, dataset_conf, results_path):
    log_write = open(os.path.join(results_path, "log.txt"), "a")
    dataset = dataset_conf.get('name')
    n_classes = dataset_conf.get('n_classes')
    n_sub = dataset_conf.get('n_sub')
    data_path = dataset_conf.get('data_path')
    isStandard = dataset_conf.get('isStandard')
    LOSO = dataset_conf.get('LOSO', False)
    classes_labels = dataset_conf.get('cl_labels')
    runs = os.listdir(os.path.join(results_path, "saved models"))
    acc = np.zeros((n_sub, len(runs)))
    kappa = np.zeros((n_sub, len(runs)))
    cf_matrix = np.zeros([n_sub, len(runs), n_classes, n_classes])
    inference_time = 0
    for sub in range(n_sub):
        _, _, _, X_test, _, y_test_onehot = get_data(
            data_path, sub, dataset, LOSO=LOSO, isStandard=isStandard)
        for seed in range(len(runs)):
            model.load_weights(os.path.join(
                results_path, 'saved models', runs[seed], 'subject-{}.h5'.format(sub + 1)))
            t0 = time.time()
            y_pred = model.predict(X_test).argmax(axis=-1)
            inference_time = (time.time() - t0) / X_test.shape[0]
            labels = y_test_onehot.argmax(axis=-1)
            acc[sub, seed] = accuracy_score(labels, y_pred)
            kappa[sub, seed] = cohen_kappa_score(labels, y_pred)
            cf_matrix[sub, seed, :, :] = confusion_matrix(labels, y_pred, normalize='true')
    head1 = head2 = ' '
    for sub in range(n_sub):
        head1 = head1 + 'sub_{} '.format(sub + 1)
        head2 = head2 + '----- '
    head1 = head1 + ' average'
    head2 = head2 + ' -------'
    info = '\n---------------------------------\nTest performance (acc & k-score):\n' + head1 + '\n' + head2
    for run in range(len(runs)):
        info = info + '\nSeed {}: '.format(run + 1)
        info_acc = '(acc %) '
        info_k = ' (k-sco) '
        for sub in range(n_sub):
            info_acc = info_acc + '{:.2f} '.format(acc[sub, run] * 100)
            info_k = info_k + '{:.3f} '.format(kappa[sub, run])
        info_acc = info_acc + ' {:.2f} '.format(np.average(acc[:, run]) * 100)
        info_k = info_k + ' {:.3f} '.format(np.average(kappa[:, run]))
        info = info + info_acc + '\n' + info_k
    info = info + '\n----------------------------------\nAverage - all seeds (acc %): '
    info = info + '{:.2f}\n (k-sco): {:.3f}\n\nInference time: {:.2f} ms per trial\n----------------------------------\n'.format(
        np.average(acc) * 100, np.average(kappa), inference_time * 1000)
    print(info)
    log_write.write(info + '\n')
    draw_performance_barChart(n_sub, acc.mean(1), 'Accuracy', results_path)
    draw_performance_barChart(n_sub, kappa.mean(1), 'k-score', results_path)
    draw_confusion_matrix(cf_matrix.mean((0, 1)), 'All', results_path, classes_labels)
    log_write.close()


class EpochCallbackWrapper(tf.keras.callbacks.Callback):
    """Keras callback to report epoch metrics to GUI and support early stop from GUI."""
    def __init__(self, epoch_callback=None, get_should_stop=None, log_epoch=True):
        super().__init__()
        self.epoch_callback = epoch_callback
        self.get_should_stop = get_should_stop
        self.log_epoch = log_epoch
        self._total_epochs = None

    def on_train_begin(self, logs=None):
        self._total_epochs = self.params.get('epochs', 500)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        tl = float(logs.get('loss', 0))
        ta = float(logs.get('accuracy', 0))
        vl = float(logs.get('val_loss', 0))
        va = float(logs.get('val_accuracy', 0))
        if self.log_epoch:
            import logging as _log
            _log.getLogger(__name__).info(
                "Epoch %4d/%d   train_acc=%.4f  val_acc=%.4f   train_loss=%.4f  val_loss=%.4f",
                epoch + 1, self._total_epochs, ta, va, tl, vl)
        if self.epoch_callback:
            self.epoch_callback(epoch + 1, tl, ta, vl, va)
        if self.get_should_stop and self.get_should_stop():
            self.model.stop_training = True


class BatchProgressLogger(tf.keras.callbacks.Callback):
    """Log batch progress so GUI log shows activity (first epoch can be slow on CPU)."""
    def __init__(self, log_batches=True):
        super().__init__()
        self.log_batches = log_batches

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch = epoch + 1
        self._steps = self.params.get('steps', None)

    def on_batch_end(self, batch, logs=None):
        if not self.log_batches or self._steps is None:
            return
        if self._epoch <= 2 or (batch + 1) % 10 == 0 or batch + 1 == self._steps:
            import logging as _log
            _log.getLogger(__name__).info("  batch %d/%d", batch + 1, self._steps)


class HistoryTracker(tf.keras.callbacks.Callback):
    """Track training history for saving plots and JSON."""
    def __init__(self):
        super().__init__()
        self.history = {
            'loss': [], 'accuracy': [],
            'val_loss': [], 'val_accuracy': []
        }

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            self.history['loss'].append(float(logs.get('loss', 0)))
            self.history['accuracy'].append(float(logs.get('accuracy', 0)))
            self.history['val_loss'].append(float(logs.get('val_loss', 0)))
            self.history['val_accuracy'].append(float(logs.get('val_accuracy', 0)))


class SavePlotEveryNEpochs(tf.keras.callbacks.Callback):
    """Save training curves every N epochs (overwrites same file) so you can check progress without GUI."""
    def __init__(self, history_tracker, results_dir, prefix="training", save_every=20):
        super().__init__()
        self.history_tracker = history_tracker
        self.results_dir = results_dir
        self.prefix = prefix
        self.save_every = save_every

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_every != 0:
            return
        if not self.history_tracker.history['loss']:
            return
        try:
            path = save_training_curves(
                self.history_tracker.history,
                self.results_dir,
                self.prefix,
            )
            _log = logging.getLogger(__name__)
            _log.info("Saved training plot (epoch %d) -> %s", epoch + 1, path)
        except Exception as e:
            logging.getLogger(__name__).warning("Could not save plot at epoch %d: %s", epoch + 1, e)


class EarlyStoppingAfterEpoch(tf.keras.callbacks.Callback):
    """Early stopping that only starts monitoring after start_epoch (for TF < 2.11)."""
    def __init__(self, monitor='val_accuracy', patience=80, start_epoch=100, mode='max', **kwargs):
        super().__init__(**kwargs)
        self.monitor = monitor
        self.patience = patience
        self.start_epoch = start_epoch
        self.mode = mode  # 'max' for accuracy (higher is better), 'min' for loss
        self.best = None
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs is None or epoch + 1 < self.start_epoch:
            return
        current = logs.get(self.monitor)
        if current is None:
            return
        improved = (self.best is None or
                    (current > self.best if self.mode == 'max' else current < self.best))
        if improved:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                print(f"\nEarlyStopping: no improvement in {self.monitor} for {self.patience} epochs (after epoch {epoch + 1}).")


def _make_early_stopping(start_epoch=100, patience=80):
    """Early stopping based on validation accuracy: stop if no improvement for patience epochs after start_epoch."""
    try:
        return EarlyStopping(
            monitor='val_accuracy', patience=patience, verbose=1,
            mode='max', restore_best_weights=False,
            start_from_epoch=start_epoch,
        )
    except TypeError:
        return EarlyStoppingAfterEpoch(
            monitor='val_accuracy', patience=patience, start_epoch=start_epoch, mode='max',
        )


# Global flag for graceful shutdown
_shutdown_requested = False


def _signal_handler(signum, frame):
    """Handle SIGINT (Ctrl+C) gracefully."""
    global _shutdown_requested
    _shutdown_requested = True
    logging.getLogger(__name__).warning("\n⚠️  Shutdown requested (Ctrl+C). Finishing current epoch and saving progress...")


def save_training_curves(history, results_dir, prefix="training"):
    """Save training curves (loss and accuracy plots)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history['loss']) + 1)
    ax1.plot(epochs, history['loss'], 'b-', label='Train Loss', alpha=0.8)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, history['accuracy'], 'b-', label='Train Accuracy', alpha=0.8)
    ax2.plot(epochs, history['val_accuracy'], 'r-', label='Val Accuracy', alpha=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f'{prefix}_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    return plot_path


def write_training_report(results_dir, preprocessing_info, training_config, test_metrics,
                         mode='fixed', subject_id=None, training_history=None, report_prefix=None):
    """Write a detailed text report (raw data, preprocessing, training flow, hyperparameters, results)."""
    report_name = f'{report_prefix}_training_report.txt' if report_prefix else 'training_report.txt'
    report_path = os.path.join(results_dir, report_name)
    lines = [
        '=' * 70,
        'TRAINING REPORT',
        '=' * 70,
        '',
        '--- 1. RAW DATA ---',
    ]
    if preprocessing_info.get('source') == 'user_provided':
        lines.append('Source: User-provided preprocessed data')
        lines.append('Path: {}'.format(preprocessing_info.get('preprocessed_path', '')))
        lines.append('Note: {}'.format(preprocessing_info.get('note', '')))
        if preprocessing_info.get('sampling_rate_hz') is not None:
            lines.append('  Sampling rate: {} Hz (downsampled)'.format(preprocessing_info['sampling_rate_hz']))
        if preprocessing_info.get('motor_imagery_segment_description'):
            lines.append('  Motor imagery segment: {}'.format(preprocessing_info['motor_imagery_segment_description']))
        if preprocessing_info.get('standardization'):
            lines.append('  Standardization: {}'.format(preprocessing_info['standardization']))
    else:
        raw = preprocessing_info.get('raw_data', {})
        for k, v in raw.items():
            lines.append('  {}: {}'.format(k, v))
        lines.append('')
        lines.append('--- 2. PREPROCESSING (method used) ---')
        seg = preprocessing_info.get('trial_segment', {})
        for k, v in seg.items():
            lines.append('  {}: {}'.format(k, v))
        lines.append('  Standardization: {}'.format(preprocessing_info.get('standardization', 'N/A')))
        lines.append('  Shuffle: {}'.format(preprocessing_info.get('shuffle', 'N/A')))
        lines.append('  Train/val split: {}'.format(preprocessing_info.get('train_val_split', 'N/A')))
        lines.append('')
        lines.append('  Filtering: {}'.format(raw.get('filtering', 'N/A')))
        lines.append('  Downsampling: {}'.format(raw.get('downsampling', 'N/A')))
        lines.append('  Motor imagery period: {}'.format(preprocessing_info.get('trial_segment', {}).get('motor_imagery_period_description', 'N/A')))

    lines.extend([
        '',
        '--- 3. TRAINING FLOW ---',
        '  Train on session 1 (AxxT) 80% train / 20% validation.',
        '  Best model selected by validation loss (ModelCheckpoint).',
        '  Test evaluation on session 2 (AxxE) only.',
        '',
        '--- 4. TRAINING HYPERPARAMETERS ---',
    ])
    for k, v in (training_config or {}).items():
        lines.append('  {}: {}'.format(k, v))

    lines.extend([
        '',
        '--- 5. TEST RESULTS ---',
    ])
    for k, v in (test_metrics or {}).items():
        if k == 'confusion_matrix':
            continue
        lines.append('  {}: {}'.format(k, v))
    cm = (test_metrics or {}).get('confusion_matrix')
    if cm is not None:
        lines.append('  confusion_matrix (rows=true, cols=pred):')
        for row in (cm.tolist() if hasattr(cm, 'tolist') else cm):
            lines.append('    ' + str(row))

    if training_history and training_history.get('loss'):
        lines.extend([
            '',
            '--- 6. TRAINING HISTORY (epochs completed) ---',
            '  Total epochs run: {}'.format(len(training_history['loss'])),
            '  Final train loss: {:.4f}  train acc: {:.4f}'.format(
                training_history['loss'][-1], training_history['accuracy'][-1]),
            '  Final val loss: {:.4f}  val acc: {:.4f}'.format(
                training_history['val_loss'][-1], training_history['val_accuracy'][-1]),
        ])
    lines.extend(['', '=' * 70, ''])
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return report_path


def save_confusion_matrix(cm, classes_labels, results_dir, prefix="test"):
    """Save confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes_labels, yticklabels=classes_labels,
           title='Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], '.2f'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f'{prefix}_confusion_matrix.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    return plot_path


def run_fixed_subject_atcnet(
    data_path,
    subject_id,
    results_dir,
    model_name='ATCNet',
    epochs=500,
    batch_size=64,
    lr=0.001,
    epoch_callback=None,
    get_should_stop=None,
    from_logits=False,
    verbose=0,
    preprocessed_data_path=None,
    preprocessing_style='original',
):
    """
    Train and test one subject with ATCNet (or another model from getModel).
    subject_id: 1-based (1..9 for BCI2a).
    preprocessed_data_path: optional path to folder containing preprocessed.npz (or X_train.npy, ...).
      If provided and valid, skips internal preprocessing and uses these arrays.
    Returns dict with 'test_metrics', 'training_history', etc.
    """
    _log = logging.getLogger(__name__)
    _log.info("-------- Training config --------")
    _log.info("Mode: FIXED SUBJECT (single subject)")
    _log.info("Subject ID: %d (of 9)", subject_id)
    _log.info("Data path: %s", data_path)
    _log.info("Dataset: BCI2a (session 1 = train/val, session 2 = test)")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        _log.info("Device: GPU %s", [g.name for g in gpus])
    else:
        _log.info("Device: CPU (no GPU available)")

    dataset = 'BCI2a'
    # Set default in_samples based on preprocessing style
    if preprocessing_style == 'cosupervisor_no_filter':
        in_samples = 448  # 3.5 s * 128 Hz
    else:
        in_samples = 1125  # 4.5 s * 250 Hz (original)
    n_channels = 22
    n_classes = 4
    classes_labels = ['Left hand', 'Right hand', 'Foot', 'Tongue']
    subject_idx = subject_id - 1  # 0-based for get_data

    file_label = "preprocessed" if preprocessed_data_path else "raw"
    os.makedirs(results_dir, exist_ok=True)
    saved_dir = os.path.join(results_dir, 'saved_models')
    os.makedirs(saved_dir, exist_ok=True)
    filepath = os.path.join(saved_dir, 'subject_{}_{}.weights.h5'.format(subject_id, file_label))

    dataset_conf = {
        'name': dataset, 'n_classes': n_classes, 'cl_labels': classes_labels,
        'n_sub': 9, 'n_channels': n_channels, 'in_samples': in_samples,
        'data_path': data_path, 'isStandard': True, 'LOSO': False
    }

    used_preprocessed_path = None
    data_spec_override = None  # None or dict with n_channels, in_samples, n_classes
    if preprocessed_data_path:
        preprocessed_data_path = os.path.abspath(os.path.expanduser(preprocessed_data_path))
        # Try cosupervisor format: train + test .npz in same folder
        train_npz = os.path.join(preprocessed_data_path, 'bci2a_train_4class_summary.npz')
        test_npz = os.path.join(preprocessed_data_path, 'bci2a_test_4class_summary.npz')
        if os.path.isfile(train_npz) and os.path.isfile(test_npz):
            loaded, data_spec_override = load_bci2a_summary_npz(
                train_npz, test_npz, subject_id=subject_id)
            if loaded is not None:
                X_train, y_train_onehot, X_val, y_val_onehot, X_test, y_test_onehot = loaded
                used_preprocessed_path = preprocessed_data_path
                in_samples = data_spec_override['in_samples']
                n_channels = data_spec_override['n_channels']
                n_classes = data_spec_override['n_classes']
                dataset_conf['in_samples'] = in_samples
                dataset_conf['n_channels'] = n_channels
                dataset_conf['n_classes'] = n_classes
                _log.info("Using cosupervisor preprocessed data: %s", preprocessed_data_path)
                _log.info("  Data spec: n_channels=%d, in_samples=%d, n_classes=%d", n_channels, in_samples, n_classes)
                # MI dataset is not standardized: fit on train only, then transform train/val/test (no leakage)
                X_train, X_val, X_test = standardize_fit_train_transform_train_val_test(
                    X_train, X_val, X_test, n_channels)
                _log.info("  Standardization: fit on train only, then transform train/val/test")
        if used_preprocessed_path is None and os.path.isdir(preprocessed_data_path):
            loaded = load_user_preprocessed(preprocessed_data_path)
            if loaded is not None:
                X_train, y_train_onehot, X_val, y_val_onehot, X_test, y_test_onehot = loaded
                used_preprocessed_path = preprocessed_data_path
                # Infer input shape from loaded data (e.g. 448 samples, 22 ch) so model matches
                in_samples = X_train.shape[3]
                n_channels = X_train.shape[2]
                n_classes = y_train_onehot.shape[1]
                dataset_conf['in_samples'] = in_samples
                dataset_conf['n_channels'] = n_channels
                dataset_conf['n_classes'] = n_classes
                _log.info("Using user-provided preprocessed data from: %s", used_preprocessed_path)
                _log.info("  Inferred shape: n_channels=%d, in_samples=%d, n_classes=%d", n_channels, in_samples, n_classes)
        if used_preprocessed_path is not None and data_spec_override is None:
            data_spec_override = {}
        if used_preprocessed_path is None:
            # User gave preprocessed path but we couldn't load; do NOT fall back to raw (avoid slow get_data)
            raise FileNotFoundError(
                "Preprocessed data path was set but no valid data found. "
                "Expected either 'bci2a_train_4class_summary.npz' and 'bci2a_test_4class_summary.npz' in the folder, "
                "or 'preprocessed.npz' / X_train.npy etc. Path: {}".format(preprocessed_data_path)
            )
    if used_preprocessed_path is None:
        # Load raw BCI2a and run internal preprocessing (no preprocessed path was used).
        # Split first, then standardize fit on train only -> no validation/test leakage.
        if preprocessing_style == 'cosupervisor_no_filter':
            _log.info("Loading raw BCI2a data with cosupervisor-style preprocessing (128 Hz, 0.5-4.0 s, NO FILTER)...")
        else:
            _log.info("Loading raw BCI2a data and running internal preprocessing (get_data)...")
        X_train, _, y_train_onehot, X_test, y_test, y_test_onehot = get_data(
            data_path, subject_idx, dataset, isStandard=False, preprocessing_style=preprocessing_style)
        # Update dataset_conf with correct in_samples based on preprocessing style (already set at top, but ensure consistency)
        dataset_conf['in_samples'] = in_samples
        if preprocessing_style == 'cosupervisor_no_filter':
            _log.info("  Preprocessing: 128 Hz downsampling, epoch 0.5-4.0 s, NO bandpass filter")
        X_train, X_val, y_train_onehot, y_val_onehot = train_test_split(
            X_train, y_train_onehot, test_size=0.2, random_state=42)
        n_ch = X_train.shape[2]
        X_train, X_val, X_test = standardize_fit_train_transform_train_val_test(
            X_train, X_val, X_test, n_ch)
        _log.info("Standardization: fit on train only, then transform train/val/test (no leakage)")
    n_ch = X_train.shape[2]
    _log.info("Train samples: %d  Val samples: %d  Test samples (session 2): %d",
              X_train.shape[0], X_val.shape[0], X_test.shape[0])
    _log.info("Input shape: (1, %d ch, %d time)", n_ch, in_samples)

    preprocessing_info = get_preprocessing_info_bci2a(used_preprocessed_path)
    if data_spec_override and data_spec_override.get('in_samples'):
        preprocessing_info['note'] = (
            'Cosupervisor preprocessed: downsampled to 128 Hz, bandpass 4–40 Hz (order 4), '
            'motor imagery segment 0.5 s–4.0 s (not 1.5 s–6.0 s). Train/val and test from '
            'bci2a_train_4class_summary.npz / bci2a_test_4class_summary.npz; filtered to subject {}. '
            'Per-channel standardization applied: fit on train only, transform train/val/test.'.format(subject_id)
        )
        preprocessing_info['sampling_rate_hz'] = 128
        preprocessing_info['motor_imagery_segment_description'] = '0.5 s to 4.0 s (trial segment; not 1.5 s–6.0 s)'
        preprocessing_info['standardization'] = 'Per-channel StandardScaler; fit on train only, transform train/val/test (no leakage).'
    elif preprocessing_style == 'cosupervisor_no_filter':
        preprocessing_info['note'] = (
            'Cosupervisor-style preprocessing (128 Hz downsampling, epoch 0.5–4.0 s) BUT NO BANDPASS FILTER. '
            'Raw frequencies preserved. Per-channel standardization: fit on train only, transform train/val/test (no leakage).'
        )
        preprocessing_info['sampling_rate_hz'] = 128
        preprocessing_info['motor_imagery_segment_description'] = '0.5 s to 4.0 s (trial segment; not 1.5 s–6.0 s)'
        preprocessing_info['bandpass_filter'] = 'None (raw frequencies preserved)'
        preprocessing_info['standardization'] = 'Per-channel StandardScaler; fit on train only, transform train/val/test (no leakage).'

    # Match original GitHub: no GaussianNoise, dropout=0.3 (paper default)
    model = getModel(model_name, dataset_conf, from_logits=from_logits)
    model.compile(
        loss=CategoricalCrossentropy(from_logits=from_logits),
        optimizer=Adam(learning_rate=lr),
        metrics=['accuracy'])

    # Training config for report (ATCNet default dropout 0.3)
    training_config = {
        'model': model_name,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': lr,
        'optimizer': 'Adam',
        'loss': 'CategoricalCrossentropy (no label smoothing)',
        'dropout (eegn/tcn for ATCNet)': 0.3,
        'val_split': '20%',
        'checkpoint': 'best by val_loss',
        'early_stopping': 'val_accuracy, patience=80, only after epoch 100',
        'reduce_lr': 'ReduceLROnPlateau(val_loss, factor=0.9, patience=20, min_lr=0.0001)',
    }
    
    # Track history for plots
    history_tracker = HistoryTracker()
    
    # Setup signal handler for graceful shutdown (only works in main thread)
    global _shutdown_requested
    _shutdown_requested = False
    try:
        import threading
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, _signal_handler)
    except (ValueError, AttributeError, RuntimeError):
        # Signal handler can only be set in main thread; skip if in worker thread (e.g., GUI)
        pass
    
    def get_should_stop_wrapper():
        return _shutdown_requested or (get_should_stop() if get_should_stop else False)
    
    # Early stopping: only after epoch 100, patience 80
    early_stopping = _make_early_stopping(start_epoch=100, patience=80)
    callbacks = [
        ModelCheckpoint(filepath, monitor='val_loss', verbose=0,
                        save_best_only=True, save_weights_only=True, mode='min'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.90, patience=20, verbose=0, min_lr=0.0001),
        early_stopping,
        EpochCallbackWrapper(epoch_callback=epoch_callback, get_should_stop=get_should_stop_wrapper, log_epoch=True),
        history_tracker,
        SavePlotEveryNEpochs(history_tracker, results_dir, prefix=f"subject_{subject_id}_{file_label}", save_every=20),
    ]
    
    try:
        model.fit(
            X_train, y_train_onehot,
            validation_data=(X_val, y_val_onehot),
            epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=verbose)
    except KeyboardInterrupt:
        _log.warning("Training interrupted, saving progress...")
    finally:
        # Always load best weights and save results
        if os.path.exists(filepath):
            model.load_weights(filepath)
            _log.info("Loaded best weights from: %s", filepath)
        else:
            _log.warning("No checkpoint found, using current weights")

    y_pred_proba = model.predict(X_test)
    if from_logits:
        y_pred_proba = tf.nn.softmax(y_pred_proba).numpy()
    y_pred = y_pred_proba.argmax(axis=-1)
    y_true = y_test_onehot.argmax(axis=-1)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    try:
        roc_auc = roc_auc_score(y_test_onehot, y_pred_proba, multi_class='ovr', average='weighted')
    except Exception:
        roc_auc = None
    cm = confusion_matrix(y_true, y_pred)

    # Save training curves
    if history_tracker.history['loss']:
        curve_path = save_training_curves(history_tracker.history, results_dir, f"subject_{subject_id}_{file_label}")
        _log.info("Saved training curves: %s", curve_path)
    else:
        _log.warning("No training history to save (training may have been interrupted before first epoch)")
    
    # Save confusion matrix plot
    cm_plot_path = save_confusion_matrix(cm, classes_labels, results_dir, f"subject_{subject_id}_{file_label}_test")
    _log.info("Saved confusion matrix: %s", cm_plot_path)
    
    # Save JSON with all results
    results_dict = {
        'config': {
            'mode': 'fixed_subject',
            'subject_id': subject_id,
            'model': model_name,
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'train_samples': int(X_train.shape[0]),
            'val_samples': int(X_val.shape[0]),
            'test_samples': int(X_test.shape[0]),
        },
        'training_history': history_tracker.history,
        'test_metrics': {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc) if roc_auc is not None else None,
            'confusion_matrix': cm.tolist(),
        }
    }
    json_path = os.path.join(results_dir, f"subject_{subject_id}_{file_label}_results.json")
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    _log.info("Saved results JSON: %s", json_path)

    # Detailed text report (raw data, preprocessing, training flow, hyperparameters, results)
    report_path = write_training_report(
        results_dir,
        preprocessing_info,
        training_config,
        results_dict['test_metrics'],
        mode='fixed',
        subject_id=subject_id,
        training_history=history_tracker.history,
        report_prefix=f'subject_{subject_id}_{file_label}',
    )
    _log.info("Saved training report: %s", report_path)
    _log.info("=" * 60)
    _log.info("All results saved to: %s", results_dir)
    _log.info("=" * 60)

    return {
        'test_metrics': {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
        },
        'training_history': history_tracker.history,
    }


def run_cross_subject_atcnet(
    data_path,
    results_dir,
    model_name='ATCNet',
    epochs=500,
    batch_size=64,
    lr=0.001,
    epoch_callback=None,
    get_should_stop=None,
    from_logits=False,
    verbose=0,
    n_subjects=9,
):
    """
    Train one model on all subjects (cross-subject). Session 1 from all subjects -> train/val split.
    Test on each subject's session 2 and report average + per-subject metrics.
    """
    _log = logging.getLogger(__name__)
    _log.info("-------- Training config --------")
    _log.info("Mode: CROSS-SUBJECT (all %d subjects)", n_subjects)
    _log.info("Data path: %s", data_path)
    _log.info("Dataset: BCI2a (session 1 = train/val combined, session 2 = test per subject)")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        _log.info("Device: GPU %s", [g.name for g in gpus])
    else:
        _log.info("Device: CPU (no GPU available)")

    dataset = 'BCI2a'
    in_samples = 1125
    n_channels = 22
    n_classes = 4
    subject_ids = list(range(1, n_subjects + 1))

    os.makedirs(results_dir, exist_ok=True)
    saved_dir = os.path.join(results_dir, 'saved_models')
    os.makedirs(saved_dir, exist_ok=True)
    filepath = os.path.join(saved_dir, 'cross_subject.weights.h5')

    dataset_conf = {
        'name': dataset, 'n_classes': n_classes, 'cl_labels': ['Left hand', 'Right hand', 'Foot', 'Tongue'],
        'n_sub': n_subjects, 'n_channels': n_channels, 'in_samples': in_samples,
        'data_path': data_path, 'isStandard': True, 'LOSO': False
    }

    # Load session 1 from all subjects and concatenate
    X_list, y_list = [], []
    for sub_id in subject_ids:
        sub_idx = sub_id - 1
        X_s, _, y_s_onehot, _, _, _ = get_data(data_path, sub_idx, dataset, isStandard=False)
        X_list.append(X_s)
        y_list.append(y_s_onehot)
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    X_train, X_val, y_train_onehot, y_val_onehot = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42)
    n_ch = X_train.shape[2]

    _log.info("Train samples (80%% of all): %d  Val samples (20%%): %d",
              X_train.shape[0], X_val.shape[0])
    _log.info("Input shape: (1, %d ch, %d time)", n_ch, in_samples)
    _log.info("Standardization: fit on train only, then transform train/val and each subject test")

    scalers = standardize_fit_train_return_scalers(X_train, n_ch)
    standardize_apply_scalers(X_train, scalers, n_ch)
    standardize_apply_scalers(X_val, scalers, n_ch)

    base = getModel(model_name, dataset_conf, from_logits=from_logits, eegn_dropout=0.35, tcn_dropout=0.35)
    inp = tf.keras.layers.Input(shape=(1, n_channels, in_samples))
    x = tf.keras.layers.GaussianNoise(0.15)(inp)
    out = base(x)
    model = tf.keras.Model(inp, out)
    model.compile(
        loss=CategoricalCrossentropy(from_logits=from_logits, label_smoothing=0.1),
        optimizer=Adam(learning_rate=lr),
        metrics=['accuracy'])
    
    # Track history for plots
    history_tracker = HistoryTracker()
    
    # Setup signal handler for graceful shutdown (only works in main thread)
    global _shutdown_requested
    _shutdown_requested = False
    try:
        import threading
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, _signal_handler)
    except (ValueError, AttributeError, RuntimeError):
        # Signal handler can only be set in main thread; skip if in worker thread (e.g., GUI)
        pass
    
    def get_should_stop_wrapper():
        return _shutdown_requested or (get_should_stop() if get_should_stop else False)
    
    # Early stopping: only after epoch 100, patience 80 (same as fixed-subject)
    early_stopping_cross = _make_early_stopping(start_epoch=100, patience=80)
    callbacks = [
        ModelCheckpoint(filepath, monitor='val_loss', verbose=0,
                        save_best_only=True, save_weights_only=True, mode='min'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.90, patience=20, verbose=0, min_lr=0.0001),
        early_stopping_cross,
        EpochCallbackWrapper(epoch_callback=epoch_callback, get_should_stop=get_should_stop_wrapper, log_epoch=True),
        history_tracker,
        SavePlotEveryNEpochs(history_tracker, results_dir, prefix="cross_subject", save_every=20),
    ]
    
    try:
        model.fit(
            X_train, y_train_onehot,
            validation_data=(X_val, y_val_onehot),
            epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=verbose)
    except KeyboardInterrupt:
        _log.warning("Training interrupted, saving progress...")
    finally:
        # Always load best weights and save results
        if os.path.exists(filepath):
            model.load_weights(filepath)
            _log.info("Loaded best weights from: %s", filepath)
        else:
            _log.warning("No checkpoint found, using current weights")

    # Test on each subject's session 2
    accs, precs, recs, f1s, rocs = [], [], [], [], []
    cms = []
    for sub_id in subject_ids:
        sub_idx = sub_id - 1
        _, _, _, X_test, _, y_test_onehot = get_data(data_path, sub_idx, dataset, isStandard=False)
        X_test = standardize_apply_scalers(X_test.copy(), scalers, n_ch)
        y_pred_proba = model.predict(X_test)
        if from_logits:
            y_pred_proba = tf.nn.softmax(y_pred_proba).numpy()
        y_pred = y_pred_proba.argmax(axis=-1)
        y_true = y_test_onehot.argmax(axis=-1)
        accs.append(accuracy_score(y_true, y_pred))
        precs.append(precision_score(y_true, y_pred, average='weighted', zero_division=0))
        recs.append(recall_score(y_true, y_pred, average='weighted', zero_division=0))
        f1s.append(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        try:
            rocs.append(roc_auc_score(y_test_onehot, y_pred_proba, multi_class='ovr', average='weighted'))
        except Exception:
            rocs.append(None)
        cms.append(confusion_matrix(y_true, y_pred))
        _log.info("Subject %d test: acc=%.4f", sub_id, accs[-1])

    acc_mean = float(np.mean(accs))
    prec_mean = float(np.mean(precs))
    rec_mean = float(np.mean(recs))
    f1_mean = float(np.mean(f1s))
    roc_vals = [r for r in rocs if r is not None]
    roc_mean = float(np.mean(roc_vals)) if roc_vals else None
    cm_mean = np.mean(cms, axis=0)

    # Save training curves
    if history_tracker.history['loss']:
        curve_path = save_training_curves(history_tracker.history, results_dir, "cross_subject")
        _log.info("Saved training curves: %s", curve_path)
    else:
        _log.warning("No training history to save")
    
    # Save confusion matrix plot
    cm_plot_path = save_confusion_matrix(cm_mean, dataset_conf['cl_labels'], results_dir, "cross_subject_test")
    _log.info("Saved confusion matrix: %s", cm_plot_path)
    
    # Save JSON with all results
    results_dict = {
        'config': {
            'mode': 'cross_subject',
            'n_subjects': n_subjects,
            'model': model_name,
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'train_samples': int(X_train.shape[0]),
            'val_samples': int(X_val.shape[0]),
        },
        'training_history': history_tracker.history,
        'test_metrics': {
            'accuracy': acc_mean,
            'precision': prec_mean,
            'recall': rec_mean,
            'f1_score': f1_mean,
            'roc_auc': roc_mean,
            'confusion_matrix': cm_mean.tolist(),
        },
        'per_subject': {
            'accuracy': [float(a) for a in accs],
            'precision': [float(p) for p in precs],
            'recall': [float(r) for r in recs],
            'f1_score': [float(f) for f in f1s],
            'roc_auc': [float(r) if r is not None else None for r in rocs],
        }
    }
    json_path = os.path.join(results_dir, "cross_subject_results.json")
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    _log.info("Saved results JSON: %s", json_path)
    _log.info("=" * 60)
    _log.info("All results saved to: %s", results_dir)
    _log.info("=" * 60)

    return {
        'test_metrics': {
            'accuracy': acc_mean,
            'precision': prec_mean,
            'recall': rec_mean,
            'f1_score': f1_mean,
            'roc_auc': roc_mean,
            'confusion_matrix': cm_mean,
        },
        'per_subject': {
            'accuracy': accs,
            'precision': precs,
            'recall': recs,
            'f1_score': f1s,
            'roc_auc': rocs,
        },
        'training_history': history_tracker.history,
    }


def getModel(model_name, dataset_conf, from_logits=False, eegn_dropout=None, tcn_dropout=None):
    n_classes = dataset_conf.get('n_classes')
    n_channels = dataset_conf.get('n_channels')
    in_samples = dataset_conf.get('in_samples')
    if model_name == 'ATCNet':
        eegn_d = eegn_dropout if eegn_dropout is not None else 0.3
        tcn_d = tcn_dropout if tcn_dropout is not None else 0.3
        model = models.ATCNet_(
            n_classes=n_classes, in_chans=n_channels, in_samples=in_samples,
            n_windows=5, attention='mha',
            eegn_F1=16, eegn_D=2, eegn_kernelSize=64, eegn_poolSize=7, eegn_dropout=eegn_d,
            tcn_depth=2, tcn_kernelSize=4, tcn_filters=32, tcn_dropout=tcn_d, tcn_activation='elu')
    elif model_name == 'TCNet_Fusion':
        model = models.TCNet_Fusion(n_classes=n_classes, Chans=n_channels, Samples=in_samples)
    elif model_name == 'EEGTCNet':
        model = models.EEGTCNet(n_classes=n_classes, Chans=n_channels, Samples=in_samples)
    elif model_name == 'EEGNet':
        model = models.EEGNet_classifier(n_classes=n_classes, Chans=n_channels, Samples=in_samples)
    elif model_name == 'EEGNeX':
        model = models.EEGNeX_8_32(n_timesteps=in_samples, n_features=n_channels, n_outputs=n_classes)
    elif model_name == 'DeepConvNet':
        model = models.DeepConvNet(nb_classes=n_classes, Chans=n_channels, Samples=in_samples)
    elif model_name == 'ShallowConvNet':
        model = models.ShallowConvNet(nb_classes=n_classes, Chans=n_channels, Samples=in_samples)
    elif model_name == 'MBEEG_SENet':
        model = models.MBEEG_SENet(nb_classes=n_classes, Chans=n_channels, Samples=in_samples)
    else:
        raise Exception("'{}' model is not supported yet!".format(model_name))
    return model


def run():
    dataset = 'BCI2a'
    if dataset == 'BCI2a':
        in_samples = 1125
        n_channels = 22
        n_sub = 9
        n_classes = 4
        classes_labels = ['Left hand', 'Right hand', 'Foot', 'Tongue']
        data_path = os.path.expanduser('~') + '/BCI Competition IV/BCI Competition IV-2a/BCI Competition IV 2a mat/'
    else:
        raise Exception("'{}' dataset is not supported in this replication.".format(dataset))

    results_path = os.path.join(os.getcwd(), "results")
    os.makedirs(results_path, exist_ok=True)
    dataset_conf = {
        'name': dataset, 'n_classes': n_classes, 'cl_labels': classes_labels,
        'n_sub': n_sub, 'n_channels': n_channels, 'in_samples': in_samples,
        'data_path': data_path, 'isStandard': True, 'LOSO': False
    }
    train_conf = {
        'batch_size': 64, 'epochs': 500, 'patience': 100, 'lr': 0.001, 'n_train': 1,
        'LearnCurves': True, 'from_logits': False, 'model': 'ATCNet'
    }

    # Train the model (set data_path in dataset_conf to your BCI2a folder first)
    # train(dataset_conf, train_conf, results_path)

    # Evaluate
    model = getModel(train_conf.get('model'), dataset_conf)
    # test(model, dataset_conf, results_path)
    print("Replication ready. Set data_path in run() and uncomment train() and test() to run.")


if __name__ == "__main__":
    run()
