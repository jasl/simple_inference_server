import threading

from app.threadpool import DaemonThreadPoolExecutor

RESULT_TIMEOUT_SEC = 1.0


def test_daemon_threadpool_runs_tasks_on_daemon_threads() -> None:
    with DaemonThreadPoolExecutor(max_workers=1, thread_name_prefix="test-daemon") as executor:
        fut = executor.submit(lambda: threading.current_thread().daemon)
        assert fut.result(timeout=RESULT_TIMEOUT_SEC) is True

