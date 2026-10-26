"""Labelers: the sources of match and non-match labels for pairs of records."""

from abc import abstractmethod, ABC
import sys
import time
import shutil
import textwrap
import pandas as pd
import numpy as np
import os
import atexit
import signal
import socket
import threading
import logging
import tempfile
import subprocess

from .frames import has_filter, is_spark
from .utils import get_logger

log = get_logger(__name__)

_WEB_EXTRA_MSG = (
    "'{dep}' is required for this labeler but is not installed. "
    "Install the browser labeling support of Iris with: pip install 'iris[web]'"
)


def _import_tabulate():
    try:
        from tabulate import tabulate
    except ImportError as e:
        raise ImportError(_WEB_EXTRA_MSG.format(dep="tabulate")) from e
    return tabulate


LABELS = (1.0, 0.0, 2.0, -1.0)


def label_of(labeler, id1, id2):
    """Return the label `labeler` gives the pair `(id1, id2)`, checked to be 1.0, 0.0,
    2.0 or -1.0."""
    value = labeler(id1, id2)
    try:
        label = float(value)
    except (TypeError, ValueError):
        label = None
    if label not in LABELS:
        raise ValueError(f'{type(labeler).__name__} gave {value!r} as the label of '
                         f'({id1}, {id2}); a label is 1.0 (match), 0.0 (non-match), 2.0 '
                         f'(unsure) or -1.0 (stop)')
    return label


class Labeler(ABC):
    """The base class of labelers, which return 1.0 for a match and 0.0 for a non-
    match."""
    @abstractmethod
    def __call__(self, id1: int, id2: int):
        pass

    def finish(self, n_labeled: int, stopped: bool = False):
        """Report that labeling ended, after `n_labeled` labels, stopped by the user
        when `stopped`."""
        pass

    def close(self):
        """Release the resources the labeler holds, such as servers or files."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def _get_row(self, df, row_id):
        rows = self.__dict__.setdefault('_rows', {})
        key = (id(df), row_id)
        if key not in rows:
            finders = self.__dict__.setdefault('_finders', {})
            if id(df) not in finders:
                finders[id(df)] = _finder(df, self._id_col)
            row = finders[id(df)](row_id)
            if row is None:
                raise KeyError(f"No row with {self._id_col}={row_id}")
            rows[key] = row
        return rows[key]


LOOKUP_COLLECT_ROWS = 100_000


def _finder(df, id_col):
    if isinstance(df, pd.DataFrame):
        index = pd.Index(df[id_col])

        def find(row_id):
            if row_id not in index:
                return None
            return df.iloc[index.get_loc(row_id)].to_dict()
        return find
    if is_spark(df):
        if df.count() <= LOOKUP_COLLECT_ROWS:
            rows = {r[id_col]: r.asDict() for r in df.collect()}
            return lambda row_id: rows.get(row_id)

        def find(row_id):
            value = row_id.item() if isinstance(row_id, np.generic) else row_id
            found = df.filter(df[id_col] == value).limit(1).collect()
            return found[0].asDict() if found else None
        return find
    import pyarrow as pa
    import pyarrow.parquet as pq

    if isinstance(df, pa.Table):
        ids = df.column(id_col).to_numpy()
        read = lambda pos: df.slice(pos, 1).to_pylist()[0]
    elif has_filter(df) or getattr(df, 'partitioning', None) is not None:
        ids = df.to_table(columns=[id_col]).column(id_col).to_numpy()
        read = lambda pos: df.take([pos]).to_pylist()[0]
    else:
        ids = df.to_table(columns=[id_col]).column(id_col).to_numpy()
        groups = [(f, g, pq.ParquetFile(f).metadata.row_group(g).num_rows)
                  for f in df.files for g in range(pq.ParquetFile(f).metadata.num_row_groups)]
        starts = np.cumsum([0] + [n for _, _, n in groups])

        def read(pos):
            k = int(np.searchsorted(starts, pos, side='right')) - 1
            path, group, _ = groups[k]
            rows = pq.ParquetFile(path).read_row_group(group)
            return rows.slice(pos - int(starts[k]), 1).to_pylist()[0]
    order = np.argsort(ids, kind='stable')
    ordered = ids[order]

    def find(row_id):
        i = int(np.searchsorted(ordered, row_id))
        if i == len(ordered) or ordered[i] != row_id:
            return None
        return read(int(order[i]))
    return find


def _whole(a, b):
    a, b = pd.Series(a).reset_index(drop=True), pd.Series(b).reset_index(drop=True)
    if not (pd.api.types.is_float_dtype(a) or pd.api.types.is_float_dtype(b)):
        return a, b
    if not all(pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x)
               for x in (a, b)):
        return a, b
    va, vb = a.to_numpy(np.float64), b.to_numpy(np.float64)
    known = ~(np.isnan(va) | np.isnan(vb))
    va, vb = va[known], vb[known]
    if not (np.all(va == np.floor(va)) and np.all(vb == np.floor(vb))
            and np.all(np.abs(va) < 2**53) and np.all(np.abs(vb) < 2**53)):
        return a, b
    return pd.Series(va.astype(np.int64)), pd.Series(vb.astype(np.int64))


class _GoldLookup:
    def __init__(self, gold, symmetric=False):
        if is_spark(gold):
            gold = gold.toPandas()
        self._symmetric = symmetric
        self._type_checked = False
        self._a = self._b = None
        self._pairs_cache = None

        a, b = gold['id1'], gold['id2']
        a, b = _whole(a, b)
        if len(a) and pd.api.types.is_integer_dtype(a) and pd.api.types.is_integer_dtype(b):
            if a.hasnans or b.hasnans:
                raise ValueError(f'GoldLabeler: the gold pairs contain null ids '
                                 f'({int(a.isna().sum())} in id1, {int(b.isna().sum())} in '
                                 f'id2); every gold pair names two records, so drop the pairs '
                                 f'without an id')
            ai = np.asarray(a, dtype=np.int64)
            bi = np.asarray(b, dtype=np.int64)
            if symmetric:
                ai, bi = (np.concatenate([ai, bi]), np.concatenate([bi, ai]))
            order = np.lexsort((bi, ai))
            self._a, self._b = np.ascontiguousarray(ai[order]), np.ascontiguousarray(bi[order])
            self._id1_types = self._id2_types = {int}
            n = len(self._a)
        else:
            self._pairs_cache = set(zip(a, b))
            if symmetric:
                self._pairs_cache |= {(y, x) for x, y in self._pairs_cache}
            self._id1_types = {type(x) for x, _ in self._pairs_cache}
            self._id2_types = {type(y) for _, y in self._pairs_cache}
            n = len(self._pairs_cache)
        if not n:
            log.warning('GoldLabeler: the gold set is empty, so every pair will be labeled 0.0')

    @classmethod
    def from_sorted(cls, id1, id2):
        self = cls.__new__(cls)
        self._symmetric = False
        self._type_checked = False
        self._pairs_cache = None
        self._a, self._b = id1, id2
        self._id1_types = self._id2_types = {int}
        if not len(id1):
            log.warning('GoldLabeler: the gold set is empty, so every pair will be labeled 0.0')
        return self

    @property
    def _pairs(self):
        if self._pairs_cache is None:
            self._pairs_cache = set(zip(self._a.tolist(), self._b.tolist()))
        return self._pairs_cache

    def _contains(self, id1, id2):
        if self._a is None:
            return (id1, id2) in self._pairs_cache
        try:
            k1, k2 = int(id1), int(id2)
        except (TypeError, ValueError):
            return False
        lo = np.searchsorted(self._a, k1, 'left')
        hi = np.searchsorted(self._a, k1, 'right')
        if lo == hi:
            return False
        j = int(np.searchsorted(self._b[lo:hi], k2, 'left'))
        return j < hi - lo and int(self._b[lo + j]) == k2

    def contains_many(self, id1, id2):
        if self._a is None:
            pairs = self._pairs_cache
            return np.fromiter(((x, y) in pairs for x, y in zip(id1, id2)), bool,
                               count=len(id1))
        from . import metrics
        return metrics.is_member_sorted(id1, id2, self._a, self._b)

    def _check_types(self, id1, id2):
        self._type_checked = True
        for value, name, gold_types in ((id1, 'id1', self._id1_types), (id2, 'id2', self._id2_types)):
            if not gold_types or isinstance(value, bool):
                continue
            numeric = (int, float, np.integer, np.floating)
            if isinstance(value, numeric) and all(issubclass(t, numeric) for t in gold_types):
                continue
            if type(value) not in gold_types:
                raise TypeError(
                    f'GoldLabeler: {name}={value!r} has type {type(value).__name__} but the gold set '
                    f'holds {name} values of type {sorted(t.__name__ for t in gold_types)}. Every pair '
                    f'would be labeled 0.0. Normalize the gold and record id dtypes to match.'
                )

    def label(self, id1, id2):
        if self._contains(id1, id2):
            return 1.0
        if not self._type_checked:
            self._check_types(id1, id2)
        return 0.0


class GoldLabeler(Labeler):
    """A labeler that labels a pair 1.0 when it is in a gold set of pairs and 0.0
    otherwise."""
    def __init__(self, gold, symmetric: bool = False):
        self._lookup = _GoldLookup(gold, symmetric)

    @classmethod
    def from_sorted(cls, id1, id2):
        """Return a labeler over integer gold pairs that are distinct and sorted by
        `(id1, id2)`."""
        labeler = cls.__new__(cls)
        labeler._lookup = _GoldLookup.from_sorted(id1, id2)
        return labeler

    @property
    def _gold(self):
        return self._lookup._pairs

    def __call__(self, id1, id2):
        return self._lookup.label(id1, id2)


class DelayedGoldLabeler(Labeler):
    """A gold labeler that waits a fixed time before each label, to simulate a
    person."""
    def __init__(self, gold, delay_secs, symmetric: bool = False):
        self._lookup = _GoldLookup(gold, symmetric)
        self._delay_secs = delay_secs

    @property
    def _gold(self):
        return self._lookup._pairs

    def __call__(self, id1, id2):
        time.sleep(self._delay_secs)
        return self._lookup.label(id1, id2)


class CLILabeler(Labeler):
    """A labeler that shows each pair in the terminal and reads the label from the
    keyboard."""
    def __init__(self, a_df, b_df, id_col: str = '_id'):
        self._a_df = a_df
        self._b_df = b_df
        self._id_col = id_col
        self._all_fields = None
        self._current_fields = None

    def __call__(self, id1, id2):
        row1 = self._get_row(self._a_df, id1)
        row2 = self._get_row(self._b_df, id2)

        if self._all_fields is None:
            self._all_fields = list(row1.keys())
        if self._current_fields is None:
            self._current_fields = set(self._all_fields)

        print("Do these refer to the same concept?")
        print("=" * 80)
        self._print_row(row1, row2, fields=self._current_fields)
        print("-" * 80)
        label = None
        while label not in ('y', 'n', 'u'):
            label = input('Enter y[es], n[o], u[nsure], h[elp], or s[top]: ').strip().lower()
            if label.startswith('h'):
                self._help_interactive(row1, row2)
                print("=" * 80)
                self._print_row(row1, row2, fields=self._current_fields)
                print("-" * 80)
                label = None 
            elif label.startswith('s'):
                confirm = input('Are you sure? Enter y[es] or n[o]: ').strip().lower()
                if confirm == 'y':
                    return -1.0
                else:
                    label = None

        print('-----------------------------------------------------------------------------------------------')
        return 1.0 if label == 'y' else 0.0 if label == 'n' else 2.0

    def _help_interactive(self, row1, row2):
        while True:
            table = []
            for idx, field in enumerate(self._all_fields, 1):
                in_current = 'x' if field in self._current_fields else ''
                a_val = str(row1.get(field, ''))
                b_val = str(row2.get(field, ''))
                table.append([str(idx), field, in_current, a_val, b_val])

            headers = ['id', 'all fields', 'current', 'A', 'B']

            term_w = shutil.get_terminal_size((120, 20)).columns

            id_w = max(len(headers[0]), len(str(len(self._all_fields)))) + 1
            field_w = min(30, max(len(headers[1]), max(len(r[1]) for r in table)))
            current_w = max(len(headers[2]), 1) + 1

            n_cols = 5
            padding = 4 + 3 * (n_cols - 1)
            available = term_w - (id_w + field_w + current_w + padding)

            colA_w = max(10, available // 2)
            colB_w = max(10, available - colA_w)
            widths = (id_w, field_w, current_w, colA_w, colB_w)

            def wrap_row(cells):
                wrapped = [
                    textwrap.wrap(cells[i], width=widths[i], break_long_words=True) or ['']
                    for i in range(5)
                ]
                height = max(map(len, wrapped))
                for i in range(5):
                    wrapped[i] += [''] * (height - len(wrapped[i]))

                out = []
                for line_idx in range(height):
                    out.append(
                        "│ {:<{idw}} │ {:<{fw}} │ {:<{cw}} │ {:<{aw}} │ {:<{bw}} │".format(
                            wrapped[0][line_idx], wrapped[1][line_idx], wrapped[2][line_idx],
                            wrapped[3][line_idx], wrapped[4][line_idx],
                            idw=id_w, fw=field_w, cw=current_w, aw=colA_w, bw=colB_w
                        )
                    )
                return out

            def horiz(left, mid, right):
                return (
                    left +
                    "─" * (id_w + 2) + mid +
                    "─" * (field_w + 2) + mid +
                    "─" * (current_w + 2) + mid +
                    "─" * (colA_w + 2) + mid +
                    "─" * (colB_w + 2) + right
                )

            top = horiz("┌", "┬", "┐")
            sep = horiz("├", "┼", "┤")
            bot = horiz("└", "┴", "┘")

            print(top)
            for line in wrap_row(headers):
                print(line)
            print(sep)

            for idx, row in enumerate(table):
                for line in wrap_row(row):
                    print(line)
                if idx < len(table) - 1:
                    print(sep)
            print(bot)

            cmd = input("Enter a[dd], r[emove], or e[xit]: ").strip().lower()
            if cmd.startswith('a'):
                idxs = input("Comma-separated indices to add: ").split(',')
                for s in idxs:
                    s = s.strip()
                    if s.isdigit() and 1 <= int(s) <= len(self._all_fields):
                        self._current_fields.add(self._all_fields[int(s) - 1])
                    else:
                        print(f"Bad index: {s}")
            elif cmd.startswith('r'):
                idxs = input("Comma-separated indices to remove: ").split(',')
                for s in idxs:
                    s = s.strip()
                    if s.isdigit() and 1 <= int(s) <= len(self._all_fields):
                        self._current_fields.discard(self._all_fields[int(s) - 1])
                    else:
                        print(f"Bad index: {s}")
            elif cmd.startswith('e'):
                break
            else:
                print("Unknown command. Use a, r, or e.")

    def _print_row(self, row1, row2, fields):
        row1_dict = {k: str(v) for k, v in row1.items() if k in fields}
        row2_dict = {k: str(v) for k, v in row2.items() if k in fields}

        term_w = shutil.get_terminal_size((120, 20)).columns
        field_w = max(len("Field"), max(map(len, row1_dict))) + 2
        remaining = term_w - (field_w + 10) 
        colA_w = remaining // 2
        colB_w = remaining - colA_w
        widths = (field_w, colA_w, colB_w)

        def wrap_row(cells):
            wrapped = [
                textwrap.wrap(cells[i], width=widths[i], break_long_words=True) or ['']
                for i in range(3)
            ]
            height = max(map(len, wrapped))
            for i in range(3):
                wrapped[i] += [''] * (height - len(wrapped[i]))

            lines = []
            for j in range(height):
                line = "│ {:<{fw}} │ {:<{aw}} │ {:<{bw}} │".format(
                    wrapped[0][j], wrapped[1][j], wrapped[2][j],
                    fw=field_w, aw=colA_w, bw=colB_w
                )
                lines.append(line)
            return lines

        hline = "├" + "─" * (field_w + 2) + "┼" + "─" * (colA_w + 2) + "┼" + "─" * (colB_w + 2) + "┤"
        top = "┌" + "─" * (field_w + 2) + "┬" + "─" * (colA_w + 2) + "┬" + "─" * (colB_w + 2) + "┐"
        bot = "└" + "─" * (field_w + 2) + "┴" + "─" * (colA_w + 2) + "┴" + "─" * (colB_w + 2) + "┘"

        print(top)
        for line in wrap_row(("Field", "From A", "From B")):
            print(line)
        print(hline)

        for index, key in enumerate(row1_dict):
            a_val = row1_dict[key]
            b_val = row2_dict.get(key, '')
            for line in wrap_row((key, a_val, b_val)):
                print(line)
            if index < len(row1_dict) - 1:
                print(hline)
        print(bot)

    def finish(self, n_labeled: int, stopped: bool = False):
        """Print that labeling ended and the number of pairs labeled."""
        print()
        if stopped:
            print(f"Labeling stopped. {n_labeled} pair(s) labeled.")
        else:
            print(f"Labeling complete. {n_labeled} pair(s) labeled.")


    @staticmethod
    def _print_dict(d):
        table = list(d.items())
        print(_import_tabulate()(table, headers=('field', 'value'), tablefmt="github"))

class CustomLabeler(Labeler):
    """A labeler whose labels come from `label_pair`, which a subclass implements."""
    def __init__(self, a_df, b_df, id_col: str = '_id'):
        self._a_df = a_df
        self._b_df = b_df
        self._id_col = id_col


    def __call__(self, id1, id2):
        row1 = self._get_row(self._a_df, id1)
        row2 = self._get_row(self._b_df, id2)
        label = self.label_pair(row1, row2)
        return label

    @abstractmethod
    def label_pair(self, row1, row2):
        """Return the label of the pair of records `row1` and `row2`."""
        pass

    @staticmethod
    def _print_dict(d):
        table = list(d.items())
        print(_import_tabulate()(table, headers=('field', 'value'), tablefmt="github"))

class WebUILabeler(Labeler):
    """A labeler that shows each pair in a web page and takes the label from the user
    there."""
    def __init__(self, a_df, b_df, id_col: str = '_id', flask_port: int = 5005, streamlit_port: int = 8501, flask_host: str = '127.0.0.1'):
        self._a_df = a_df
        self._b_df = b_df
        self._id_col = id_col
        self._all_fields = None
        self._current_fields = None
        self._flask_port = flask_port
        self._streamlit_port = streamlit_port
        self._flask_host = flask_host
        self._lock = threading.Lock()
        self._current_pair = None
        self._current_fields_mem = None
        self._label = None
        self._pair_seq = 0
        self._flask_server = None
        try:
            from flask import Flask
        except ImportError as e:
            raise ImportError(_WEB_EXTRA_MSG.format(dep="flask")) from e
        self._flask_app = Flask(__name__)
        self._column_order = list(a_df.columns)
        self._setup_flask_routes()
        self._flask_thread = None
        self._streamlit_proc = None
        self._app_path = None
        self._cleaned_up = False
        self._server_started = False
        self._done = False
        self._done_stopped = False
        self._n_labeled = 0

    def _setup_flask_routes(self):
        from flask import jsonify, request

        @self._flask_app.route('/get_pair', methods=['GET'])
        def get_pair():
            with self._lock:
                if self._current_pair is not None:
                    return jsonify({
                        'row1': self._current_pair[0],
                        'row2': self._current_pair[1],
                        'fields': list(self._current_fields_mem),
                        'pair_seq': self._pair_seq,
                    })
                else:
                    return jsonify({'status': 'waiting'}), 204

        @self._flask_app.route('/submit_label', methods=['POST'])
        def submit_label():
            data = request.get_json()
            with self._lock:
                submitted_seq = data.get('pair_seq')
                if submitted_seq is not None and submitted_seq != self._pair_seq:
                    return jsonify({
                        'status': 'stale',
                        'expected_pair_seq': self._pair_seq,
                    }), 409
                self._label = data.get('label')
            return jsonify({'status': 'ok'})

        @self._flask_app.route('/update_fields', methods=['POST'])
        def update_fields():
            data = request.get_json()
            with self._lock:
                self._current_fields = set(data.get('fields', []))
                self._current_fields_mem = self._current_fields
            return jsonify({'status': 'ok'})

        @self._flask_app.route('/done', methods=['GET'])
        def done():
            with self._lock:
                return jsonify({
                    'done': self._done,
                    'stopped': self._done_stopped,
                    'n_labeled': self._n_labeled,
                })

    @staticmethod
    def _port_in_use(host, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
                return False
            except OSError:
                return True

    def _ensure_server_started(self):
        if self._server_started:
            return

        werkzeug_log = logging.getLogger('werkzeug')
        werkzeug_log.setLevel(logging.ERROR)
        try:
            from werkzeug.serving import make_server
            self._flask_server = make_server(
                self._flask_host, self._flask_port, self._flask_app, threaded=True
            )
        except (OSError, SystemExit) as e:
            raise RuntimeError(
                f"could not bind the labeler's Flask port {self._flask_port} on "
                f"{self._flask_host}: {e}. A previous labeling session may still be "
                f"running -- stop it (e.g. `lsof -ti tcp:{self._flask_port} | xargs kill`), "
                f"or pass a different flask_port to WebUILabeler."
            ) from e

        self._flask_thread = threading.Thread(
            target=self._flask_server.serve_forever, daemon=True
        )
        self._flask_thread.start()

        try:
            app_code = self._streamlit_app_code()
            with tempfile.NamedTemporaryFile('w', delete=False, suffix='.py') as f:
                f.write(app_code)
                self._app_path = f.name
            streamlit_cmd = [sys.executable, "-m", "streamlit", "run", self._app_path, "--server.port", str(self._streamlit_port), "--server.headless", "true"]
            self._streamlit_proc = subprocess.Popen(
                streamlit_cmd, env=os.environ.copy(), start_new_session=True
            )
            atexit.register(self._cleanup)
            self._server_started = True
            self._await_ready()
        except Exception:
            self._cleanup()
            raise

    def _await_ready(self, timeout: float = 30.0):
        deadline = time.time() + timeout
        url = f'http://{self._flask_host}:{self._flask_port}/done'
        while time.time() < deadline:
            if self._streamlit_proc is not None and self._streamlit_proc.poll() is not None:
                raise RuntimeError(
                    f'the Streamlit labeling UI exited immediately with code '
                    f'{self._streamlit_proc.returncode}. Check that streamlit is '
                    f'installed (pip install "iris[web]") and that port '
                    f'{self._streamlit_port} is free.'
                )
            try:
                import urllib.request
                with urllib.request.urlopen(url, timeout=1):
                    return
            except Exception:
                time.sleep(0.2)
        raise RuntimeError(
            f'the labeling backend did not become ready within {timeout:.0f}s '
            f'(flask port {self._flask_port}, streamlit port {self._streamlit_port})'
        )

    def __call__(self, id1, id2):
        self._ensure_server_started()
        row1 = self._get_row(self._a_df, id1)
        row2 = self._get_row(self._b_df, id2)
        if self._all_fields is None:
            self._all_fields = list(row1.keys())
        if self._current_fields is None:
            self._current_fields = set(self._all_fields)
        with self._lock:
            while self._current_pair is not None:
                self._lock.release()
                time.sleep(0.1)
                self._lock.acquire()
            self._current_pair = (row1, row2)
            self._pair_seq += 1
            self._current_fields_mem = self._current_fields if self._current_fields else set(self._all_fields)
            self._label = None
        checks = 0
        while True:
            with self._lock:
                label = self._label
            if label is not None:
                break
            checks += 1
            if checks % 10 == 0 and self._streamlit_proc is not None \
                    and self._streamlit_proc.poll() is not None:
                raise RuntimeError(
                    f'the Streamlit labeling UI exited (code '
                    f'{self._streamlit_proc.returncode}) while waiting for a label'
                )
            time.sleep(0.2)
        with self._lock:
            self._current_pair = None
            self._label = None
        return label

    def finish(self, n_labeled: int, stopped: bool = False):
        """Show the end of labeling in the web page, then stop its servers."""
        if self._cleaned_up:
            return
        with self._lock:
            self._done = True
            self._done_stopped = stopped
            self._n_labeled = n_labeled

        if self._streamlit_proc is not None:
            time.sleep(2)
        self._cleanup()

    def close(self):
        """Stop the web servers the labeler started."""
        self._cleanup()

    def _cleanup(self):
        if self._cleaned_up:
            return
        self._cleaned_up = True

        server = getattr(self, '_flask_server', None)
        if server is not None:
            try:
                server.shutdown()
                server.server_close()
            except Exception:
                pass
            self._flask_server = None
        if self._flask_thread is not None:
            self._flask_thread.join(timeout=5)
            self._flask_thread = None
        self._server_started = False

        try:
            atexit.unregister(self._cleanup)
        except Exception:
            pass

        proc = self._streamlit_proc
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    proc.kill()
                proc.wait()
        self._streamlit_proc = None

        if self._app_path is not None:
            try:
                os.unlink(self._app_path)
            except OSError:
                pass
            self._app_path = None

    def _streamlit_app_code(self):
        column_order = self._column_order
        return f'''
import streamlit as st
import requests
import time
import os
import json
import tempfile
import textwrap
import pandas as pd

st.title("Iris Web Labeler")

FLASK_URL = "http://{self._flask_host}:{self._flask_port}"

if 'last_pair' not in st.session_state:
    st.session_state['last_pair'] = None

if 'selected_fields' not in st.session_state:
    st.session_state['selected_fields'] = {column_order}

try:
    done_resp = requests.get(FLASK_URL + '/done', timeout=5)
    if done_resp.status_code == 200:
        done_data = done_resp.json()
        if done_data.get('done', False):
            n = done_data.get('n_labeled', 0)
            if done_data.get('stopped', False):
                st.info(f"Labeling stopped. {{n}} pair(s) labeled.")
            else:
                st.success(f"Labeling complete! {{n}} pair(s) labeled.")
            st.stop()
except Exception:
    pass

if st.session_state.get('stopped', False):
    st.write("Labeling stopped.")
else:
    pair = None
    try:
        resp = requests.get(FLASK_URL + '/get_pair', timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            row1 = data['row1']
            row2 = data['row2']
            fields = data['fields']
            pair_seq = data.get('pair_seq')
            st.session_state['last_pair'] = (row1, row2, fields, pair_seq)
            pair = (row1, row2, fields, pair_seq)
        if resp.status_code == 204:
            pair = None
    except Exception as e:
        pair = st.session_state['last_pair']

    if pair is not None:
        row1, row2, fields, pair_seq = pair
        st.subheader("Do these two records refer to the same entity?")
        all_fields = {column_order}
        
        def update_backend_fields(new_fields):
            try:
                requests.post(FLASK_URL + '/update_fields', json={{'fields': new_fields}}, timeout=10)
            except Exception as e:
                st.error(f"Failed to update fields: {{e}}")
        
        current_selection = fields
        
        selected_fields = st.multiselect(
            "Fields to display:",
            options=all_fields,
            default=current_selection,
            key='field_selector',
        )
        
        if set(selected_fields) != set(current_selection):
            update_backend_fields(selected_fields)
        def wrap(val, width=40):
            return textwrap.fill(str(val), width=width, break_long_words=False)
        table_data = []
        ordered_fields = [f for f in all_fields if f in selected_fields]
        for f in ordered_fields:
            a_val = wrap(row1.get(f, ''), width=40)
            b_val = wrap(row2.get(f, ''), width=40)
            table_data.append((f, a_val, b_val))  # Do NOT wrap f
        table_html = """<table>
<thead>
<tr>
  <th style='white-space:nowrap'>Field</th>
  <th style='max-width:300px;word-break:break-word;white-space:pre-wrap;'>From A</th>
  <th style='max-width:300px;word-break:break-word;white-space:pre-wrap;'>From B</th>
</tr>
</thead>
<tbody>
"""
        for f, a_val, b_val in table_data:
            table_html += f"<tr><td style='white-space:nowrap'>{{f}}</td><td style='max-width:300px;word-break:break-word;white-space:pre-wrap;'>{{a_val}}</td><td style='max-width:300px;word-break:break-word;white-space:pre-wrap;'>{{b_val}}</td></tr>"
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)
        
        st.markdown("""
        <style>
        .stButton > button {{
            background-color: #87CEEB !important;
            color: #000000 !important;
            border: 2px solid #4682B4 !important;
            font-weight: bold !important;
            border-radius: 8px !important;
            padding: 8px 16px !important;
        }}
        .stButton > button:hover {{
            background-color: #4682B4 !important;
            color: white !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
        }}
        </style>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        if 'label_sent' not in st.session_state:
            st.session_state['label_sent'] = False
        def send_label(label):
            try:
                r = requests.post(
                    FLASK_URL + '/submit_label',
                    json={{'label': label, 'pair_seq': pair_seq}},
                    timeout=10,
                )
                if r.status_code == 409:
                    st.warning("That pair is no longer current - reloading the current pair.")
                    st.session_state['last_pair'] = None
                    return
                st.session_state['label_sent'] = True
                if label == -1.0:
                    st.session_state['stopped'] = True
            except Exception as e:
                st.error(f"Failed to send label: {{e}}")
        with col1:
            if st.button("Yes (y)", key="yes_btn"):
                send_label(1.0)
        with col2:
            if st.button("No (n)", key="no_btn"):
                send_label(0.0)
        with col3:
            if st.button("Unsure (u)", key="unsure_btn"):
                send_label(2.0)
        with col4:
            if st.button("Stop (s)", key="stop_btn"):
                send_label(-1.0)
        if st.session_state['label_sent']:
            if st.session_state.get('stopped', False):
                st.write("Labeling stopped.")
            else:
                st.success("Label sent! Waiting for next pair...")
                st.session_state['label_sent'] = False
                time.sleep(0.3)  # Give backend time to update
                st.rerun()
    else:
        st.write("No pair to label. Waiting...")
        time.sleep(0.2)
        st.rerun()
'''


