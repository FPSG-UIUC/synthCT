# Implements generic work queue for synthesis

import greenstalk
import json
import random

import redis
import pickle
from loguru import logger

from multiprocessing import Queue
from multiprocessing.managers import SyncManager
from queue import PriorityQueue


class MyManager(SyncManager):
    def __init__(self):
        super().__init__()


MyManager.register("PriorityQueue", PriorityQueue)


def Manager():
    m = MyManager()
    m.start()
    return m


class WorkQueue:
    def push(self, item):
        raise NotImplementedError

    def put(self, item, timeout=None):
        raise NotImplementedError

    def pop(self):
        raise NotImplementedError

    def get(self, timeout=None):
        raise NotImplementedError


class ListWQ(WorkQueue):
    def __init__(self, items=[]):
        self.m = Manager()
        self.wq = self.m.PriorityQueue()
        for item in items:
            self.put(item)

    def push(self, item):
        self.wq.put(item)

    def put(self, item, timeout=None, priority=10):
        ctr = random.randrange(10000000)
        self.wq.put((priority, ctr, item), timeout=timeout)

    def pop(self, priority=False):
        item = self.wq.get()
        if not priority:
            return item[1]
        return item

    def get(self, timeout=None):
        from synthesis.synth_task import RosetteSynthesisTask
        item = self.wq.get(timeout=timeout)[2]
        if isinstance(item, RosetteSynthesisTask):
            item.job_id = None
        return item

    def delete(self, job):
        # No-op
        pass


class BeanstalkWQ(WorkQueue):
    def __init__(self, conn):
        self.tasks = redis.Redis(
            host=conn['redis']['host'],
            port=conn['redis']['port'])

        self.wq = greenstalk.Client(
            (conn['beanstalkd']['host'],
             conn['beanstalkd']['port']),
            use=conn['beanstalkd']['wq'],
            watch=conn['beanstalkd']['wq'])

    def push(self, item):
        self.wq.put(item)

    def put(self, item, timeout=172800, priority=10):
        task_id = item.id
        serialized = pickle.dumps(item)
        self.tasks.set(task_id, serialized)
        self.wq.put(task_id, ttr=timeout, priority=priority)

    def pop(self):
        job = self.wq.reserve()
        self.wq.bury(job)
        current = job.body

        task = pickle.loads(self.tasks.get(current))
        task.job_id = job.id

        return task

    def delete(self, job):
        self.wq.delete(job)

    def get(self, timeout=None):
        return self.pop()
