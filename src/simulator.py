"""
Simple time-stepped simulator for tasks and VMs.
"""

import numpy as np
import random
from collections import deque


class Task:
    def __init__(self, id, arrival, size):
        self.id = id
        self.arrival = arrival
        self.size = size
        self.remaining = size
        self.start = None
        self.finish = None
        self.assigned_vm = None

    def __repr__(self):
        return f"<Task id={self.id} size={self.size} arrival={self.arrival:.2f}>"


class VM:
    def __init__(self, id, rate):
        self.id = id
        self.rate = rate            # work units per time unit
        self.queue = deque()
        self.time_busy = 0.0        # total time busy

    def __repr__(self):
        return f"<VM id={self.id} rate={self.rate:.2f}>"


class Simulator:
    def __init__(self,
                 num_vms=4,
                 sim_time=1000,
                 arrival_rate=0.5,
                 strategy=None,
                 batch_interval=10,
                 seed=0):
        self.num_vms = num_vms
        self.sim_time = sim_time
        self.arrival_rate = arrival_rate
        self.strategy = strategy
        self.batch_interval = batch_interval
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)

        # Heterogeneous VM processing rates
        self.vms = [
            VM(i, rate)
            for i, rate in enumerate(np.random.uniform(10, 30, size=num_vms))
        ]

    # ---------- Task generation ----------
    def generate_tasks(self):
        tasks = []
        t = 0.0
        tid = 0
        while t < self.sim_time:
            inter_arrival = np.random.exponential(1.0 / self.arrival_rate)
            t += inter_arrival
            if t >= self.sim_time:
                break
            size = random.randint(10, 200)
            tasks.append(Task(tid, t, size))
            tid += 1
        return tasks

    # ---------- Main simulation ----------
    def run(self):
        tasks = self.generate_tasks()
        waiting = []         # tasks not yet assigned
        completed = []       # finished tasks

        t = 0.0
        dt = 1.0
        task_idx = 0
        next_batch = self.batch_interval

        # Main time loop
        while t <= self.sim_time:
            # 1) Add arriving tasks
            while task_idx < len(tasks) and tasks[task_idx].arrival <= t:
                waiting.append(tasks[task_idx])
                task_idx += 1

            # 2) Assign waiting tasks using chosen strategy
            if getattr(self.strategy, "name", "") == "PSO":
                if t >= next_batch or (task_idx == len(tasks) and waiting):
                    # batch assign
                    self.strategy.assign_batch(waiting, self.vms, t)
                    waiting = []
                    next_batch += self.batch_interval
            else:
                if waiting:
                    self.strategy.assign_batch(waiting, self.vms, t)
                    waiting = []

            # 3) Process tasks on each VM for dt
            for vm in self.vms:
                if vm.queue:
                    current = vm.queue[0]
                    work = vm.rate * dt
                    current.remaining = max(0.0, current.remaining - work)
                    vm.time_busy += dt

                    if current.remaining <= 1e-8:
                        current.finish = t + dt
                        completed.append(current)
                        vm.queue.popleft()
                        if vm.queue and vm.queue[0].start is None:
                            vm.queue[0].start = current.finish

            t += dt

        # 4) Finish remaining tasks after sim_time
        extra = 0.0
        while any(vm.queue for vm in self.vms):
            for vm in self.vms:
                if vm.queue:
                    cur = vm.queue[0]
                    ttf = cur.remaining / vm.rate
                    extra += ttf
                    cur.remaining = 0.0
                    cur.finish = self.sim_time + extra
                    completed.append(cur)
                    vm.queue.popleft()
                    if vm.queue and vm.queue[0].start is None:
                        vm.queue[0].start = cur.finish

        # ---------- Metrics ----------
        if not completed:
            return {
                "tasks_submitted": 0,
                "tasks_completed": 0,
                "avg_response_time": float("nan"),
                "makespan": float("nan"),
                "throughput": float("nan"),
                "utilization": float("nan"),
            }

        avg_resp = np.mean([c.finish - c.arrival for c in completed])
        makespan = max(c.finish for c in completed) - min(
            c.arrival for c in completed
        )
        total_time = self.sim_time + extra
        throughput = len(completed) / total_time
        utilization = sum(vm.time_busy for vm in self.vms) / (self.num_vms * total_time)

        return {
            "tasks_submitted": len(tasks),
            "tasks_completed": len(completed),
            "avg_response_time": float(avg_resp),
            "makespan": float(makespan),
            "throughput": float(throughput),
            "utilization": float(utilization),
        }
