"""Active learning in batch and continuous mode, shared by both engines."""
import traceback
from copy import copy
from dataclasses import dataclass, field
from queue import Empty, PriorityQueue, Queue
from threading import Event, Thread
from typing import Any
import time

import numpy as np
import pandas as pd

from .labeler import Labeler, WebUILabeler
from .ml_model import MLModel, normalize_fv_cells
from .utils import (check_driver_collect, check_training_data_matches,
                    feature_vector_width, get_logger, load_training_data_streaming,
                    save_training_data_streaming, type_check)

log = get_logger(__name__)

SELECTION_STRATEGIES = ('entropy', 'random', 'hybrid')

DEFAULT_MAX_LABELED = 550


def _get_pos_negative(batch):
    pos = batch['label'].sum()
    neg = len(batch) - pos
    return pos, neg


class BatchLearner:
    """Batch-mode active learning: train, select a batch of the most uncertain pairs,
    label them, repeat."""

    def __init__(self, model, labeler, batch_size=10, max_iter=50,
                 checkpoint_path='active-matcher-training-data.parquet',
                 selection='entropy'):
        self._check_init_args(model, labeler, batch_size, max_iter, checkpoint_path)

        self._batch_size = batch_size
        if selection not in SELECTION_STRATEGIES:
            raise ValueError(f'selection must be one of {SELECTION_STRATEGIES}, '
                             f'got {selection!r}')
        self._selection = selection
        self._labeler = labeler
        self._model = copy(model)
        self._max_iter = max_iter
        self._checkpoint_path = checkpoint_path
        self.local_training_fvs_ = None
        self._terminate_if_label_everything = False
        self._excluded_pairs = set()

    def _check_init_args(self, model, labeler, batch_size=10, max_iter=50,
                         checkpoint_path='active-matcher-training-data.parquet'):
        type_check(model, 'model', MLModel)
        type_check(labeler, 'labeler', Labeler)
        type_check(batch_size, 'batch_size', int)
        type_check(max_iter, 'max_iter', int)
        if batch_size <= 0 :
            raise ValueError('batch_size must be > 0')
        if max_iter <= 0 :
            raise ValueError('max_iter must be > 0')
        type_check(checkpoint_path, 'checkpoint_path', str)

    def _get_pos_negative(self, batch):
        return _get_pos_negative(batch)

    def _open_pool(self, fvs):
        raise NotImplementedError

    def _label_everything(self, fvs):
        opened = not hasattr(fvs, 'everything')
        pool = self._open_pool(fvs) if opened else fvs
        check_driver_collect(
            pool.count(), 'labeling every feature vector (the pool is smaller than '
            'batch_size x max_iter, so active learning would label all of it)',
            'Lower max_iter or batch_size so active learning selects a subset.',
        )
        batch = pool.everything()
        batch['label'] = batch[['id1', 'id2']].apply(lambda x: float(self._labeler(*x.values)), axis=1)
        batch['labeled_in_iteration'] = -2.0
        self.local_training_fvs_ = batch
        try:
            pool.train(self._model, self.local_training_fvs_)
        finally:
            if opened:
                pool.close()
        return self.local_training_fvs_

    def train(self, fvs, seeds):
        """Run active learning from the seeds over the feature vectors and return the
        labeled pairs."""
        pool = self._open_pool(fvs)
        try:
            n_fvs = pool.count()

            existing_training_data = load_training_data_streaming(self._checkpoint_path, log)
            check_training_data_matches(
                existing_training_data,
                feature_vector_width(seeds) or pool.vector_width(),
                self._checkpoint_path)

            if existing_training_data is not None:
                self.local_training_fvs_ = existing_training_data
                if _beyond(existing_training_data, seeds):
                    log.warning(
                        f'Resuming active learning from {len(self.local_training_fvs_)} '
                        f'labeled examples in {self._checkpoint_path!r}; the supplied '
                        f'seeds are ignored. Delete that file or pass '
                        f'checkpoint_path=<new path> to start fresh.')
            else:
                self.local_training_fvs_ = seeds.copy()
                save_training_data_streaming(self.local_training_fvs_, self._checkpoint_path, log)

            max_itr = self._max_iter
            if n_fvs <= len(self.local_training_fvs_) + (self._batch_size * max_itr):
                if self._terminate_if_label_everything:
                    log.info('the pool is no larger than the labels to collect: labeling every pair in it')
                    return self._label_everything(pool)
                else:
                    log.info('the pool is no larger than the labels to collect: active learning will label every pair in it')

            if existing_training_data is not None:
                log.info('Training initial model with existing data')
                normalize_fv_cells(self.local_training_fvs_, 'feature_vectors')
                pool.train(self._model, self.local_training_fvs_)
                log.info('Initial model training complete')
            else:
                self.local_training_fvs_ = seeds.copy()

            self.local_training_fvs_['labeled_in_iteration'] = -1
            normalize_fv_cells(self.local_training_fvs_, 'feature_vectors')

            total_pos, total_neg = self._get_pos_negative(self.local_training_fvs_)
            if total_pos == 0 or total_neg == 0:
                raise RuntimeError(
                    f'active learning needs both matches and non-matches to start, but '
                    f'the seeds are {int(total_pos)} positive / {int(total_neg)} '
                    f'negative. create_seeds ranks candidates by the score column and '
                    f'labels from both ends, so this means one end contained no example '
                    f'of its class -- common when a score is inflated by records '
                    f'agreeing on empty or null fields. Raise nseeds, supply seeds '
                    f'directly, or use a score column that separates the classes.')

            log.info(f'max iter = {max_itr}')
            i = 0
            label = None
            if isinstance(self._labeler, WebUILabeler):
                log.warning("Records are almost ready to be labeled.")

            while i < max_itr and label != -1:
                log.info(f'starting iteration {i}')
                log.info('training model')
                pool.train(self._model, self.local_training_fvs_)

                log.info('selecting and labeling new examples')
                new_labeled_batch = pool.batch(
                    self._model, self.local_training_fvs_[['id1', 'id2']],
                    self._excluded_pairs, self._batch_size, self._selection, i)

                labels = []
                labeled_indices = []
                user_stopped = False

                for idx, row in new_labeled_batch.iterrows():
                    label = float(self._labeler(row['id1'], row['id2']))
                    if label == -1.0:
                        user_stopped = True
                        break
                    labels.append(label)
                    labeled_indices.append(idx)
                    if label not in (0.0, 1.0):
                        self._excluded_pairs.add((int(row['id1']), int(row['id2'])))

                new_labeled_batch = new_labeled_batch.loc[labeled_indices].copy()
                new_labeled_batch['label'] = labels
                new_labeled_batch = new_labeled_batch[(new_labeled_batch['label'] == 0.0) | (new_labeled_batch['label'] == 1.0)]
                new_labeled_batch['labeled_in_iteration'] = i

                pos, neg = self._get_pos_negative(new_labeled_batch)
                total_pos += pos
                total_neg += neg

                save_training_data_streaming(new_labeled_batch, self._checkpoint_path, log)

                self.local_training_fvs_ = pd.concat([self.local_training_fvs_, new_labeled_batch], ignore_index=True)
                log.info(f'new batch positive = {pos} negative = {neg}, total positive = {total_pos} negative = {total_neg}')

                if n_fvs <= len(self.local_training_fvs_):
                    log.info('all fvs labeled, terminating active learning')
                    break
                if user_stopped:
                    log.info('user stopped labeling, terminating active learning')
                    break
                i += 1
        finally:
            pool.close()

        labeled_data = self.local_training_fvs_.dropna(subset=['label']).copy()
        if isinstance(self._labeler, WebUILabeler):
            log.warning("Active learning is complete.")
        return labeled_data


class ContinuousLearner:
    """Continuous-mode active learning: label pairs while the model is retrained in the
    background."""

    def __init__(self, model, labeler, queue_size=50, max_labeled=DEFAULT_MAX_LABELED,
                 on_demand_stop=True,
                 checkpoint_path='active-matcher-training-data.parquet',
                 candidate_window_size=None):
        self._check_init_args(model, labeler, queue_size, max_labeled, on_demand_stop, checkpoint_path)
        if candidate_window_size is not None:
            type_check(candidate_window_size, 'candidate_window_size', int)
            if candidate_window_size <= 0:
                raise ValueError('candidate_window_size must be > 0')
        self._candidate_window_size = candidate_window_size

        self._queue_size = queue_size
        self._max_labeled = max_labeled
        self._on_demand_stop = on_demand_stop
        self._min_batch_size = 3
        self._labeler = labeler
        self._model = copy(model)
        self._checkpoint_path = checkpoint_path
        self.local_training_fvs_ = None
        self._terminate_if_label_everything = False
        self._training_error = None
        self._training_join_timeout = 300

    def _check_init_args(self, model, labeler, queue_size, max_labeled, on_demand_stop, checkpoint_path):
        type_check(model, 'model', MLModel)
        type_check(labeler, 'labeler', Labeler)
        type_check(queue_size, 'queue_size', int)
        type_check(max_labeled, 'max_labeled', int)
        if queue_size <= 0 :
            raise ValueError('queue_size must be > 0')
        if max_labeled <= 0 :
            raise ValueError('max_labeled must be > 0')
        type_check(on_demand_stop, 'on_demand_stop', bool)
        type_check(checkpoint_path, 'checkpoint_path', str)

    def _get_pos_negative(self, batch):
        return _get_pos_negative(batch)

    def _open_pool(self, fvs):
        raise NotImplementedError

    def _label_everything(self, fvs):
        opened = not hasattr(fvs, 'everything')
        pool = self._open_pool(fvs) if opened else fvs
        check_driver_collect(
            pool.count(), 'labeling every feature vector (the pool is smaller than the '
            'labeling budget, so active learning would label all of it)',
            'Lower max_labeled so active learning selects a subset.',
        )
        batch = pool.everything()
        batch['label'] = batch[['id1', 'id2']].apply(lambda x: float(self._labeler(*x.values)), axis=1)
        batch['labeled_in_iteration'] = -2.0
        self.local_training_fvs_ = batch
        try:
            pool.train(self._model, self.local_training_fvs_)
        finally:
            if opened:
                pool.close()
        return copy(self._model)

    def train(self, fvs, seeds):
        """Run active learning from the seeds over the feature vectors and return the
        labeled pairs."""
        pool = self._open_pool(fvs)
        to_be_label_queue = PriorityQueue(self._queue_size)
        labeled_queue = Queue(self._queue_size * 5)

        stop_training = Event()
        pool_queued = Event()

        existing_training_data = load_training_data_streaming(self._checkpoint_path, log)
        check_training_data_matches(
            existing_training_data,
            feature_vector_width(seeds) or pool.vector_width(),
            self._checkpoint_path)

        if existing_training_data is not None:
            if _beyond(existing_training_data, seeds):
                log.warning(
                    f'Resuming active learning from {len(existing_training_data)} labeled '
                    f'examples in {self._checkpoint_path!r}; the supplied seeds are '
                    f'ignored. Delete that file or pass checkpoint_path=<new path> to '
                    f'start fresh.')
            seeds = existing_training_data

        if not self._on_demand_stop:
            remaining_examples = self._max_labeled
            log.info(f'Remaining examples to label: {remaining_examples}')

        self._training_error = None
        self.local_training_fvs_ = None

        training_thread = Thread(
            target=self._training_loop,
            args=(to_be_label_queue, labeled_queue, stop_training, pool_queued, pool, seeds),
            daemon=True,
        )
        training_thread.start()
        nlabeled = len(seeds)
        terminate = False
        pos, neg = self._get_pos_negative(seeds.dropna(subset=['label']))
        if isinstance(self._labeler, WebUILabeler):
            log.warning("Records are almost ready to be labeled.")
        while (not terminate and self._on_demand_stop) or (nlabeled < self._max_labeled and not self._on_demand_stop):
            if pool_queued.is_set() and to_be_label_queue.empty():
                break
            try:
                example = to_be_label_queue.get(timeout=10).item
            except Empty:
                if not training_thread.is_alive():
                    if self._training_error is not None:
                        raise RuntimeError(
                            'active learning failed: the training thread raised an exception'
                        ) from self._training_error
                    if self.local_training_fvs_ is not None:
                        return self.local_training_fvs_.dropna(subset=['label']).copy()
                    raise RuntimeError(
                        'label queue is empty and the training thread is not alive, '
                        'with no training data produced')
            else:
                example['label'] = float(self._labeler(example['id1'], example['id2']))
                if example['label'] == -1.0:
                    log.info('user stopped labeling, terminating active learning')
                    terminate = True
                elif example['label'] != 2.0:
                    labeled_queue.put(example)
                    pos += 1 if example['label'] == 1.0 else 0
                    neg += 1 if example['label'] == 0.0 else 0
                    log.info(f'total positive = {pos} negative = {neg}')
                    nlabeled += 1
        stop_training.set()
        training_thread.join(timeout=self._training_join_timeout)
        if training_thread.is_alive():
            log.warning(
                f'training thread did not stop within {self._training_join_timeout}s; '
                f'continuing with the data labeled so far (the thread is a daemon and '
                f'will not block interpreter exit)'
            )

        if self._training_error is not None:
            raise RuntimeError(
                'active learning failed: the training thread raised an exception'
            ) from self._training_error

        labeled_data = self.local_training_fvs_.dropna(subset=['label']).copy()
        if isinstance(self._labeler, WebUILabeler):
            log.warning("Active learning is complete.")
        return labeled_data

    def _take_labels(self, labeled_queue, iteration):
        new_examples = []
        while not labeled_queue.empty():
            new_examples.append(labeled_queue.get())
        if len(new_examples) != 0:
            df = pd.DataFrame(new_examples)

            df['labeled_in_iteration'] = float(iteration)
            self.local_training_fvs_.loc[df.index, 'label'] = df['label']
            self.local_training_fvs_.loc[df.index, 'labeled_in_iteration'] = df['labeled_in_iteration']

            save_training_data_streaming(df, self._checkpoint_path, log)

    def _training_loop(self, to_be_label_queue, labeled_queue, stop_event, pool_queued, pool, seeds):
        try:
            try:
                n_fvs = pool.count()

                if n_fvs <= len(seeds) + self._max_labeled and not self._on_demand_stop:
                    if self._terminate_if_label_everything:
                        log.info('the pool is no larger than the labels to collect: labeling every pair in it')
                        return self._label_everything(pool)
                    else:
                        log.info('the pool is no larger than the labels to collect: active learning will label every pair in it')

                self.local_training_fvs_ = _by_pair(seeds.copy())
                self.local_training_fvs_['labeled_in_iteration'] = -1.0
                normalize_fv_cells(self.local_training_fvs_, 'feature_vectors')

                total_pos, total_neg = self._get_pos_negative(self.local_training_fvs_)
                if total_pos == 0 or total_neg == 0:
                    raise RuntimeError('both positive and negative vectors are required for training')

                i = 0
                while not stop_event.is_set():
                    while (self._queue_size - to_be_label_queue.qsize() < self._min_batch_size
                           and not stop_event.is_set()):
                        time.sleep(1)

                    if stop_event.is_set():
                        break

                    self._take_labels(labeled_queue, i)

                    labeled_only = self.local_training_fvs_.dropna(subset=['label'])
                    pool.train(self._model, labeled_only)

                    batch_size = self._queue_size - to_be_label_queue.qsize()
                    new_labeled_batch = _by_pair(pool.top_entropy(
                        self._model, self.local_training_fvs_[['id1', 'id2']],
                        batch_size, i, self._candidate_window_size))

                    new_labeled_batch['label'] = np.nan
                    self.local_training_fvs_ = pd.concat(
                        [self.local_training_fvs_, new_labeled_batch.drop(columns='entropy')])

                    for row_id, r in new_labeled_batch.iterrows():
                        ent = r.pop('entropy')
                        to_be_label_queue.put(PQueueItem(ent, r))

                    i += 1
                    if n_fvs <= len(self.local_training_fvs_):
                        log.info('every pair in the pool is queued: active learning ends '
                                 'when they are labeled')
                        pool_queued.set()
                        stop_event.wait()

                self._take_labels(labeled_queue, i)
                self.local_training_fvs_ = self.local_training_fvs_.dropna(subset=['label'])
                pool.train(self._model, self.local_training_fvs_)
            finally:
                pool.close()
        except Exception as e:
            self._training_error = e
            log.error(traceback.format_exc())

        return copy(self._model)


def _beyond(checkpoint, seeds):
    if seeds is None:
        return True
    if hasattr(seeds, 'toPandas'):
        seeds = seeds.select('id1', 'id2').toPandas()
    given = set(zip(seeds['id1'].tolist(), seeds['id2'].tolist()))
    return not set(zip(checkpoint['id1'].tolist(), checkpoint['id2'].tolist())) <= given


def _by_pair(frame):
    frame.index = pd.MultiIndex.from_arrays(
        [frame['id1'].to_numpy(np.int64), frame['id2'].to_numpy(np.int64)],
        names=['pair_id1', 'pair_id2'])
    return frame


@dataclass(order=True)
class PQueueItem:
    """A pair in the queue of continuous active learning, ordered by its entropy."""
    entropy : float
    item : Any=field(compare=False)
