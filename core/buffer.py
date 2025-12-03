import queue


class DoubleBuffer:
    """Latest-value buffer using a tiny queue to reduce race/overwrite issues."""

    def __init__(self, maxsize=1):
        self.queue = queue.Queue(maxsize=maxsize or 1)
        self._last = None

    def write(self, frame):
        try:
            # Drop the oldest item to avoid blocking when full
            self.queue.get_nowait()
        except queue.Empty:
            pass
        self.queue.put(frame)

    def read(self, timeout=None):
        try:
            item = (
                self.queue.get(timeout=timeout)
                if timeout is not None
                else self.queue.get_nowait()
            )
            self._last = item
            return item
        except queue.Empty:
            return self._last
