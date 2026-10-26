"""The disk a step of the local engine plans to hold, checked before it writes, and the
error it stops with when the disk fills while it runs."""
import contextlib
import errno
import os
import re
import shutil
import tempfile
import threading

from .._core.storage import SPILL_DIR_ENV, spill_dir

REMEDY = f'Set {SPILL_DIR_ENV} to a directory on a larger volume, or free space.'

SHORT_WRITE = re.compile(r'^\d+ requested and \d+ written$')

_RUNNING = threading.local()


def _steps():
    return _RUNNING.__dict__.setdefault('steps', [])


def scratch_root():
    """Return the directory that scratch directories are made in."""
    return spill_dir() or tempfile.gettempdir()


def existing(path):
    """Return `path`, or its nearest parent that exists, or None."""
    probe = os.path.abspath(os.fspath(path))
    while not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            return None
        probe = parent
    return probe


def volume(path):
    """Return the device of the volume that holds `path`, or None when it is unknown."""
    probe = existing(path)
    try:
        return None if probe is None else os.stat(probe).st_dev
    except OSError:
        return None


def _free(path):
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return None


def _gib(nbytes):
    if nbytes < 2**20:
        return f'{nbytes / 2**10:.1f} KiB'
    if nbytes < 2**30:
        return f'{nbytes / 2**20:.1f} MiB'
    return f'{nbytes / 2**30:.2f} GiB'


def _where(probe):
    root = os.path.abspath(scratch_root())
    try:
        inside = os.path.commonpath([root, probe]) == root
    except ValueError:
        inside = False
    return root if inside else probe


def _names(names):
    names = list(dict.fromkeys(names))
    return names[0] if len(names) == 1 else f'{", ".join(names[:-1])} and {names[-1]}'


class DiskStep:
    """The disk that one step of the local engine plans to hold, phase by phase, and the
    checks it has made."""

    def __init__(self, name, remedy):
        self.name = name
        self.remedy = remedy
        self.phases = []
        self.at = -1
        self.recent = {}
        self.outputs = []

    def phase(self, name, kept=(), scratch=()):
        """Plan the next phase, or the named one again: the bytes kept until the step ends and
        the scratch held while it runs, as `(nbytes, directory)` pairs."""
        planned = (name, [(int(b), volume(d), d) for b, d in kept if b > 0],
                   [(int(b), volume(d), d) for b, d in scratch if b > 0])
        names = [p[0] for p in self.phases]
        if name in names:
            self.phases[names.index(name)] = planned
        else:
            self.phases.append(planned)

    def need(self, device, start):
        """Return the most bytes that the phases from `start` on hold at once on the
        volume `device`, and the names of those that hold any."""
        most = kept = 0
        names = []
        for name, keeps, scratch in self.phases[start:]:
            if any(v == device for _, v, _ in keeps + scratch):
                names.append(name)
            kept += sum(b for b, v, _ in keeps if v == device)
            most = max(most, kept + sum(b for b, v, _ in scratch if v == device))
        return most, names

    def begin(self, name=None):
        """Start the phase `name` once each volume is checked to hold the most that it and
        the later phases need at once; with no name, check the whole plan."""
        start = 0 if name is None else [p[0] for p in self.phases].index(name)
        if name is not None:
            self.at = start
        places = {}
        for _, keeps, scratch in self.phases[start:]:
            for _, device, directory in keeps + scratch:
                places.setdefault(device, directory)
        for device, directory in places.items():
            most, names = self.need(device, start)
            outer, more = _later(device, self)
            check(directory, most + outer, name or f'what {self.name} writes', self.remedy,
                  [n for n in names if n != name] + more)


def current():
    """Return the step that is running."""
    return _steps()[-1]


def output(path):
    """Name a directory that the running step made for its result, to be removed when the
    step fails."""
    if _steps():
        _steps()[-1].outputs.append(path)


def _later(device, skip=None):
    total, names = 0, []
    for step in _steps():
        if step is skip:
            continue
        most, found = step.need(device, step.at + 1)
        total += most
        names += found
    return total, names


def _step():
    return _steps()[0].name if _steps() else None


def check(path, need, what, remedy=REMEDY, counted=()):
    """Raise an error before any work when the volume of `path` cannot hold `need` bytes
    for `what`, and remember the check for the error of a disk that fills later."""
    probe = existing(path)
    if probe is None:
        return
    free = _free(probe)
    if free is None:
        return
    if _steps():
        _steps()[0].recent.setdefault(volume(probe), []).append(
            ('check', probe, need, free, what, remedy))
    if free < need:
        counts = (f' The need counts {_names(counted)}, which {_step()} writes there.'
                  if counted else '')
        raise RuntimeError(f'not enough free disk for {what} in {_where(probe)!r}: need '
                           f'about {_gib(need)}, {_gib(free)} free.{counts} {remedy}')


def require(nbytes, path, what, remedy=REMEDY, kept=None):
    """Raise an error before any work when the volume of `path` cannot hold `nbytes` for
    `what` (`kept` of them past this phase) beside what later phases hold there."""
    later, names = _later(volume(path))
    kept = nbytes if kept is None else kept
    need = max(int(nbytes), int(kept) + later)
    check(path, need, what, remedy, names if need > nbytes else ())


class Meter:
    """A piece written in parts and checked at the bytes its parts take on disk: after
    the first part, and each time the work done doubles, the volume must hold the rest."""

    def __init__(self, what, path, total, kept=True):
        self.what = what
        self.path = path
        self.total = total
        self.kept = kept
        self.next = 0
        self.events = _steps()[0].recent.setdefault(volume(path), []) if _steps() else []
        self.events.append(('meter', existing(path), what, self))

    def tight(self, most):
        """Return whether the volume may lack room for `most` bytes beside what later phases
        hold, so a first part should be written and measured before the rest."""
        probe = existing(self.path)
        free = None if probe is None else _free(probe)
        later, _ = _later(volume(self.path))
        return free is not None and free < max(most, (most if self.kept else 0) + later)

    def finish(self):
        """Mark the piece written, so that a full disk later is not blamed on it."""
        self.events[:] = [e for e in self.events if e[0] != 'meter' or e[3] is not self]

    def wrote(self, done, written, pending=0):
        """Check the rest of the piece once `done` of its `total` units are written in
        `written` bytes, `pending` bytes more being on disk already for parts not done."""
        if done >= self.total:
            self.finish()
        if done <= 0 or done < self.next or done >= self.total or written <= 0:
            return
        self.next = 2 * done
        rest = max(0, int(written * (self.total - done) / done) - int(pending))
        remedy = _steps()[0].remedy if _steps() else REMEDY
        require(rest, self.path, f'the rest of {self.what}, at the {_gib(written)} its first '
                f'{done / self.total:.1%} took', remedy, None if self.kept else 0)


def measured(run, fn, tasks, workers, meter, most, done, **kwargs):
    """Return `run(fn, tasks, workers, ...)`, calling `done` per task; on a volume that may
    lack room for `most` bytes, the first task runs alone and is measured first."""
    tasks = list(tasks)
    if len(tasks) < 2 or not meter.tight(most):
        out = run(fn, tasks, workers, done=done, **kwargs)
    else:
        out = run(fn, tasks[:1], 1, done=done, **kwargs)
        out += run(fn, tasks[1:], workers, done=lambda i, result: done(i + 1, result),
                   **kwargs)
    meter.finish()
    return out


def written_in(directory, names=None):
    """Return the bytes of the files `names` in `directory` (all when None), and of the
    temporary files being written there."""
    done = pending = 0
    try:
        entries = list(os.scandir(directory))
    except OSError:
        return 0, 0
    for entry in entries:
        try:
            size = entry.stat().st_size
        except OSError:
            continue
        if entry.name.startswith('.') and entry.name.endswith('.tmp'):
            pending += size
        elif names is None or entry.name in names:
            done += size
    return done, pending


def _disk_full(e):
    return e.errno == errno.ENOSPC or (e.errno is None and
                                       SHORT_WRITE.match(str(e)) is not None)


def _full_error(step, e):
    where = getattr(e, 'filename', None)
    if where:
        device = volume(where)
    else:
        held = [d for d, events in step.recent.items() if events]
        device = min(held, key=lambda d: _free(step.recent[d][-1][1]) or 0) if held else None
    found = (step.recent.get(device) or [None])[-1]
    if found is not None and found[0] == 'check':
        _, probe, need, free, what, remedy = found
        return (f'{step.name} ran out of disk space in {_where(probe)!r}. When it checked '
                f'{what}, it estimated about {_gib(need)} would be needed there, and '
                f'{_gib(free)} was free; the data needed more than that. {remedy}')
    probe = found[1] if found is not None else (existing(where) if where else None)
    probe = probe or scratch_root()
    free = _free(probe)
    now = f'; {_gib(free)} is free now' if free is not None else ''
    if found is not None:
        return (f'{step.name} ran out of disk space in {_where(probe)!r} while it wrote '
                f'{found[2]}, before a first part of it was done to measure the room it '
                f'needs{now}. {step.remedy}')
    return (f'{step.name} ran out of disk space in {_where(probe)!r}, where it had made no '
            f'estimate{now}. {step.remedy}')


@contextlib.contextmanager
def disk_step(name, remedy=REMEDY):
    """Run one step of the local engine with its later phases counted in disk checks, and a
    full disk turned into an error naming the step, directory, estimate and free space."""
    step = DiskStep(name, remedy)
    _steps().append(step)
    try:
        yield step
    except BaseException as e:
        for path in step.outputs:
            shutil.rmtree(path, ignore_errors=True)
        if isinstance(e, OSError) and step is _steps()[0] and _disk_full(e):
            raise OSError(errno.ENOSPC, _full_error(step, e)) from e
        raise
    finally:
        _steps().remove(step)
