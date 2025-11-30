# PSO-based load balancing strategy

import numpy as np
import random


class PSOStrategy:
    def __init__(self,
                 num_vms,
                 batch_interval=10,
                 particles=30,
                 iterations=60,
                 seed=0):
        self.num_vms = num_vms
        self.batch_interval = batch_interval
        self.particles = particles
        self.iterations = iterations
        self.seed = seed
        self.name = "PSO"

        random.seed(seed)
        np.random.seed(seed)

    def assign_batch(self, tasks, vms, current_time):
        """Assign a batch of tasks to VMs using PSO."""
        if not tasks:
            return

        num_tasks = len(tasks)
        vm_rates = np.array([vm.rate for vm in vms])

        # Initialize particles
        positions = [
            np.random.randint(0, self.num_vms, num_tasks)
            for _ in range(self.particles)
        ]
        velocities = [np.zeros(num_tasks) for _ in range(self.particles)]

        p_best = [p.copy() for p in positions]
        p_best_scores = [self.fitness(p, tasks, vm_rates) for p in positions]

        best_index = int(np.argmin(p_best_scores))
        g_best = p_best[best_index].copy()
        g_best_score = p_best_scores[best_index]

        # PSO loop
        for _ in range(self.iterations):
            for i in range(self.particles):
                r1 = np.random.rand(num_tasks)
                r2 = np.random.rand(num_tasks)

                velocities[i] = (
                    0.5 * velocities[i]
                    + 1.0 * r1 * (p_best[i] - positions[i])
                    + 1.0 * r2 * (g_best - positions[i])
                )

                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(
                    np.rint(positions[i]), 0, self.num_vms - 1
                ).astype(int)

                score = self.fitness(positions[i], tasks, vm_rates)
                if score < p_best_scores[i]:
                    p_best[i] = positions[i].copy()
                    p_best_scores[i] = score
                    if score < g_best_score:
                        g_best = positions[i].copy()
                        g_best_score = score

        # Apply best assignment
        for task, vm_idx in zip(tasks, g_best):
            vm_idx = int(vm_idx)
            task.assigned_vm = vm_idx
            task.start = max(task.arrival, current_time)
            vms[vm_idx].queue.append(task)

    def fitness(self, assignment, tasks, vm_rates):
        """Load balance fitness: std of VM loads."""
        loads = np.zeros(len(vm_rates))
        for task, vm in zip(tasks, assignment):
            loads[int(vm)] += task.size / vm_rates[int(vm)]
        return float(np.std(loads))
