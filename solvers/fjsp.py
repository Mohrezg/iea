import random
import numpy as np
import time
from collections import defaultdict
from deap import base, creator, tools
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from copy import deepcopy


class Operation:
    """Represents a single operation of a job."""
    def __init__(self, op_id, job_id, op_index, workstation, duration):
        self.id = op_id
        self.job_id = job_id
        self.op_index = op_index
        self.workstation = workstation
        self.duration = duration


class FlexibleJobShopSolver:

    def __init__(self, workstations, machines_per_ws, job_routings, processing_times):

        self.workstation_names = workstations
        self.num_workstations = len(workstations)
        self.machines_per_ws = machines_per_ws
        self.job_routings = job_routings
        self.num_jobs = len(job_routings)
        self.p_times = processing_times

        # Build operations list
        self.ops_by_id = []
        op_id = 0
        for j in range(self.num_jobs):
            for idx, ws in enumerate(self.job_routings[j]):
                dur = self._get_processing_time(j, ws)
                self.ops_by_id.append(Operation(op_id, j, idx, ws, dur))
                op_id += 1
        self.num_ops = len(self.ops_by_id)
        self.op_dict = {op.id: op for op in self.ops_by_id}

        # Setup DEAP
        self._setup_deap()

    def _get_processing_time(self, job, ws):
        return self.p_times.get((ws, job), 0)

    # ----------------------------------------------------------------------
    # Chromosome encoding / decoding
    # ----------------------------------------------------------------------
    def generate_random_chromosome(self):
        machines = [random.randint(0, self.machines_per_ws[op.workstation] - 1)
                    for op in self.ops_by_id]
        keys = [random.random() for _ in range(self.num_jobs)]
        return {'machines': machines, 'keys': keys}

    def decode_chromosome(self, chrom):
        machines_assign = chrom['machines']
        keys = chrom['keys']
        ops_per_machine = defaultdict(list)
        for op in self.ops_by_id:
            m = machines_assign[op.id]
            ops_per_machine[(op.workstation, m)].append(op)

        # Sort operations on each machine by job key (for sequence)
        for (ws, m), op_list in ops_per_machine.items():
            op_list.sort(key=lambda op: keys[op.job_id])

        machine_available = {}
        for ws in range(self.num_workstations):
            for m in range(self.machines_per_ws[ws]):
                machine_available[(ws, m)] = 0

        job_ready = [0] * self.num_jobs
        schedule = defaultdict(list)

        for (ws, m), op_list in ops_per_machine.items():
            for op in op_list:
                start = max(machine_available[(ws, m)], job_ready[op.job_id])
                end = start + op.duration
                schedule[(ws, m)].append((start, end, op.job_id, op.op_index))
                machine_available[(ws, m)] = end
                job_ready[op.job_id] = end

        makespan = max(job_ready)
        return makespan, schedule

    def eval_makespan(self, chrom):
        makespan, _ = self.decode_chromosome(chrom)
        return (makespan,)

    # ----------------------------------------------------------------------
    # Diversity measures
    # ----------------------------------------------------------------------
    def calculate_diversity(self, population):
        if len(population) < 2:
            return 0.0
        diversity_sum = 0
        count = 0
        sample_size = min(50, len(population))
        indices = random.sample(range(len(population)), sample_size)
        for i in range(sample_size):
            for j in range(i + 1, sample_size):
                ind1 = population[indices[i]]['machines']
                ind2 = population[indices[j]]['machines']
                diff = sum(a != b for a, b in zip(ind1, ind2)) / len(ind1)
                diversity_sum += diff
                count += 1
        return diversity_sum / count if count > 0 else 0.0

    def distance(self, ind1, ind2):
        m_dist = sum(a != b for a, b in zip(ind1['machines'], ind2['machines'])) / len(ind1['machines'])
        k_dist = sum(abs(a - b) for a, b in zip(ind1['keys'], ind2['keys'])) / len(ind1['keys'])
        return (m_dist + k_dist) / 2

    def crowding_tournament_selection(self, population, k, tournsize, crowd_penalty=0.1):
        chosen = []
        for _ in range(k):
            aspirants = random.sample(population, tournsize)
            aspirants.sort(key=lambda x: x.fitness.values[0])
            best = aspirants[0]
            min_dist = float('inf')
            for other in population:
                if other != best:
                    min_dist = min(min_dist, self.distance(best, other))
            adjusted_fitness = best.fitness.values[0] - crowd_penalty * min_dist * 100
            chosen.append(best)
        return chosen

    # ----------------------------------------------------------------------
    # DEAP setup
    # ----------------------------------------------------------------------
    def _setup_deap(self):
        # Create fitness and individual types if not already created
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", dict, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", tools.initIterate, creator.Individual,
                              self.generate_random_chromosome)
        self.toolbox.register("evaluate", self.eval_makespan)
        self.toolbox.register("mate", self.cx_two_point_dict)
        self.toolbox.register("mutate", self.mutate_dict, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def cx_two_point_dict(self, ind1, ind2):
        m1, m2 = ind1['machines'], ind2['machines']
        k1, k2 = ind1['keys'], ind2['keys']
        L = len(m1)
        pt1 = random.randint(0, L - 1)
        pt2 = random.randint(pt1, L - 1)
        for i in range(L):
            if pt1 <= i <= pt2:
                m1[i], m2[i] = m2[i], m1[i]
        for i in range(self.num_jobs):
            if random.random() < 0.5:
                k1[i], k2[i] = k2[i], k1[i]
        del ind1.fitness.values
        del ind2.fitness.values
        return ind1, ind2

    def mutate_dict(self, ind, indpb=0.2, strength=1.0):
        if random.random() < indpb:
            num_mutations = max(1, int(len(ind['machines']) * 0.1 * strength))
            for _ in range(num_mutations):
                idx = random.randrange(len(ind['machines']))
                op = self.ops_by_id[idx]
                ws = op.workstation
                ind['machines'][idx] = random.randint(0, self.machines_per_ws[ws] - 1)
            del ind.fitness.values
        if random.random() < indpb:
            num_mutations = max(1, int(len(ind['keys']) * 0.2 * strength))
            for _ in range(num_mutations):
                j = random.randrange(self.num_jobs)
                ind['keys'][j] = random.random()
            del ind.fitness.values
        return (ind,)

    def cataclysmic_mutation(self, ind, severity=0.5):
        for idx in range(len(ind['machines'])):
            if random.random() < severity:
                op = self.ops_by_id[idx]
                ws = op.workstation
                ind['machines'][idx] = random.randint(0, self.machines_per_ws[ws] - 1)
        for j in range(self.num_jobs):
            if random.random() < severity:
                ind['keys'][j] = random.random()
        del ind.fitness.values
        return ind

    def clone_individual(self, ind):
        new_ind = creator.Individual({'machines': ind['machines'][:], 'keys': ind['keys'][:]})
        new_ind.fitness.values = ind.fitness.values
        return new_ind

    # ----------------------------------------------------------------------
    # SPT heuristic for seeding
    # ----------------------------------------------------------------------
    def spt_schedule(self):
        machine_available = {}
        for ws in range(self.num_workstations):
            for m in range(self.machines_per_ws[ws]):
                machine_available[(ws, m)] = 0
        job_ready = [0] * self.num_jobs
        job_next_op = [0] * self.num_jobs
        schedule = defaultdict(list)
        unfinished = set(range(self.num_jobs))

        while unfinished:
            candidates = []
            for j in unfinished:
                op_idx = job_next_op[j]
                if op_idx >= len(self.job_routings[j]):
                    continue
                ws = self.job_routings[j][op_idx]
                proc = self._get_processing_time(j, ws)
                for m in range(self.machines_per_ws[ws]):
                    start = max(machine_available[(ws, m)], job_ready[j])
                    candidates.append((start, j, op_idx, ws, m, proc))
            if not candidates:
                break
            candidates.sort(key=lambda x: (x[5], x[0]))  # shortest processing time, earliest start
            start, j, op_idx, ws, m, proc = candidates[0]
            end = start + proc
            machine_available[(ws, m)] = end
            job_ready[j] = end
            job_next_op[j] += 1
            schedule[(ws, m)].append((start, end, j, op_idx))
            if job_next_op[j] == len(self.job_routings[j]):
                unfinished.remove(j)
        makespan = max(job_ready)
        return makespan, schedule

    def schedule_to_chromosome(self, schedule):
        machines_assign = [0] * self.num_ops
        for (ws, m), ops in schedule.items():
            for (start, end, job, op_idx) in ops:
                for op in self.ops_by_id:
                    if op.job_id == job and op.op_index == op_idx:
                        machines_assign[op.id] = m
                        break
        job_first_start = [float('inf')] * self.num_jobs
        for (ws, m), ops in schedule.items():
            for (start, end, job, op_idx) in ops:
                if op_idx == 0:
                    job_first_start[job] = min(job_first_start[job], start)
        max_start = max(job_first_start) if self.num_jobs > 0 else 1
        keys = [s / max_start if max_start > 0 else 0.0 for s in job_first_start]
        return {'machines': machines_assign, 'keys': keys}

    def generate_spt_individual(self):
        _, sched = self.spt_schedule()
        return creator.Individual(self.schedule_to_chromosome(sched))

    def generate_mixed_population(self, pop_size, spt_fraction=0.4):
        num_spt = int(pop_size * spt_fraction)
        pop = []
        for _ in range(num_spt):
            pop.append(self.generate_spt_individual())
        for _ in range(pop_size - num_spt):
            pop.append(self.toolbox.individual())
        return pop

    # ----------------------------------------------------------------------
    # Enhanced Tabu Search (local improvement)
    # ----------------------------------------------------------------------
    class EnhancedTabuSearch:
        def __init__(self, solver, tabu_tenure=50, max_iter=100, neighborhood_size=30):
            self.solver = solver
            self.tabu_tenure = tabu_tenure
            self.max_iter = max_iter
            self.neighborhood_size = neighborhood_size

        def generate_neighbors(self, current, num_neighbors=30):
            neighbors = []
            # Operator 1: Single machine change
            for _ in range(num_neighbors // 3):
                idx = random.randrange(len(current['machines']))
                op = self.solver.ops_by_id[idx]
                ws = op.workstation
                new_m = random.randint(0, self.solver.machines_per_ws[ws] - 1)
                if new_m != current['machines'][idx]:
                    neigh = self.solver.clone_individual(current)
                    neigh['machines'][idx] = new_m
                    fit = self.solver.eval_makespan(neigh)[0]
                    move = ('m1', idx, current['machines'][idx], new_m)
                    neighbors.append((neigh, fit, move))
            # Operator 2: Swap two job keys
            for _ in range(num_neighbors // 3):
                if random.random() < 0.5 and self.solver.num_jobs >= 2:
                    j1, j2 = random.sample(range(self.solver.num_jobs), 2)
                    neigh = self.solver.clone_individual(current)
                    neigh['keys'][j1], neigh['keys'][j2] = neigh['keys'][j2], neigh['keys'][j1]
                    fit = self.solver.eval_makespan(neigh)[0]
                    move = ('swap', j1, j2)
                    neighbors.append((neigh, fit, move))
            # Operator 3: Key perturbation
            for _ in range(num_neighbors // 3):
                j = random.randrange(self.solver.num_jobs)
                new_key = max(0, min(1, current['keys'][j] + random.gauss(0, 0.2)))
                neigh = self.solver.clone_individual(current)
                neigh['keys'][j] = new_key
                fit = self.solver.eval_makespan(neigh)[0]
                move = ('key', j, current['keys'][j], new_key)
                neighbors.append((neigh, fit, move))
            return neighbors

        def improve(self, individual):
            current = self.solver.clone_individual(individual)
            best = self.solver.clone_individual(current)
            best_fit = best.fitness.values[0]
            tabu_list = {}
            for iteration in range(self.max_iter):
                neighbors = self.generate_neighbors(current, self.neighborhood_size)
                if not neighbors:
                    break
                neighbors.sort(key=lambda x: x[1])
                found = False
                for neigh, fit, move in neighbors:
                    if move[0] == 'm1':
                        inv_move = ('m1', move[1], move[3], move[2])
                    elif move[0] == 'swap':
                        inv_move = ('swap', move[2], move[1])
                    else:
                        inv_move = ('key', move[1], move[3], move[2])
                    is_tabu = inv_move in tabu_list and tabu_list[inv_move] > iteration
                    if fit < best_fit:
                        current = neigh
                        best_fit = fit
                        best = self.solver.clone_individual(neigh)
                        found = True
                        tabu_list[inv_move] = iteration + self.tabu_tenure
                        break
                    elif not is_tabu and not found:
                        current = neigh
                        found = True
                        tabu_list[inv_move] = iteration + self.tabu_tenure
                if not found:
                    neigh, fit, move = neighbors[0]
                    current = neigh
                tabu_list = {k: v for k, v in tabu_list.items() if v > iteration}
            return best

    # ----------------------------------------------------------------------
    # Main GA loop
    # ----------------------------------------------------------------------
    def run_ga(self, pop_size=200, ngen=150,
               base_mut_prob=0.3, max_mut_prob=0.9,
               cx_prob=0.8, elite_size=3,
               stagnation_limit=25, diversity_threshold=0.15,
               spt_fraction=0.3, verbose=True,
               time_limit=None):  # <-- NEW PARAMETER
        """
        Run the hybrid GA and return the best individual and logbook.
        If time_limit is given (seconds), stop when this time is exceeded.
        """
        start_time = time.time()

        pop = self.generate_mixed_population(pop_size, spt_fraction)

        # Evaluate initial population
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        hof = tools.HallOfFame(1)
        hof.update(pop)

        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("std", np.std)

        ts = self.EnhancedTabuSearch(self, tabu_tenure=40, max_iter=60, neighborhood_size=25)

        best_ever = hof[0].fitness.values[0]
        stagnation_counter = 0
        diversity_history = []

        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals", "diversity", "mut_rate"] + (stats.fields if stats else [])

        if verbose:
            print(f"Gen 0: Best = {best_ever}")

        gen = 1
        # Loop condition: continue while gen <= ngen AND (no time limit OR time not exceeded)
        while gen <= ngen and (time_limit is None or time.time() - start_time < time_limit):
            diversity = self.calculate_diversity(pop)
            diversity_history.append(diversity)

            # Adaptive mutation rate
            if stagnation_counter > stagnation_limit // 2:
                current_mut_prob = min(max_mut_prob, base_mut_prob + (stagnation_counter / stagnation_limit) * 0.4)
            elif diversity < diversity_threshold:
                current_mut_prob = min(max_mut_prob, base_mut_prob + 0.2)
            else:
                current_mut_prob = base_mut_prob

            # Selection and reproduction
            pop.sort(key=lambda x: x.fitness.values[0])
            elites = [self.clone_individual(ind) for ind in pop[:elite_size]]

            offspring = self.crowding_tournament_selection(pop, len(pop) - elite_size, 3, crowd_penalty=0.05)
            offspring = [self.clone_individual(ind) for ind in offspring]

            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cx_prob:
                    self.toolbox.mate(child1, child2)

            # Adaptive mutation
            for mutant in offspring:
                if random.random() < current_mut_prob:
                    self.mutate_dict(mutant, indpb=0.3, strength=1.0 + stagnation_counter / 20)

            nevals = 0
            if stagnation_counter >= stagnation_limit:
                if verbose:
                    print(f"  -> Cataclysmic mutation at gen {gen}!")
                pop.sort(key=lambda x: x.fitness.values[0])
                keep_elite = max(1, pop_size // 10)
                survivors = [self.clone_individual(ind) for ind in pop[:keep_elite]]
                for _ in range(pop_size - keep_elite):
                    new_ind = self.toolbox.individual()
                    self.cataclysmic_mutation(new_ind, severity=0.5)
                    survivors.append(new_ind)
                pop = survivors
                stagnation_counter = stagnation_limit // 2

                invalid_after_cataclysm = [ind for ind in pop if not ind.fitness.valid]
                fitnesses = list(map(self.toolbox.evaluate, invalid_after_cataclysm))
                for ind, fit in zip(invalid_after_cataclysm, fitnesses):
                    ind.fitness.values = fit
                nevals = len(invalid_after_cataclysm)
            else:
                invalid_off = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = list(map(self.toolbox.evaluate, invalid_off))
                for ind, fit in zip(invalid_off, fitnesses):
                    ind.fitness.values = fit
                pop = elites + offspring
                nevals = len(invalid_off)

            # Local search periodically
            if gen % 8 == 0 and gen > 10:
                pop.sort(key=lambda x: x.fitness.values[0])
                for i in range(min(3, len(pop))):
                    improved = ts.improve(pop[i])
                    if improved.fitness.values[0] < pop[i].fitness.values[0]:
                        pop[i] = improved

            hof.update(pop)

            current_best = hof[0].fitness.values[0]
            if current_best < best_ever:
                best_ever = current_best
                stagnation_counter = 0
                if verbose:
                    print(f"Gen {gen}: New best = {best_ever}, Diversity = {diversity:.3f}")
            else:
                stagnation_counter += 1

            if diversity < 0.05 and gen > 30:
                if verbose:
                    print(f"  -> Low diversity ({diversity:.3f}), injecting random individuals...")
                num_random = pop_size // 4
                pop = pop[:-num_random] + [self.toolbox.individual() for _ in range(num_random)]
                invalid_ind = [ind for ind in pop[-num_random:] if not ind.fitness.valid]
                fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

            if gen % 10 == 0 and verbose:
                record = stats.compile(pop) if stats else {}
                logbook.record(gen=gen, nevals=nevals,
                               diversity=diversity, mut_rate=current_mut_prob, **record)
                if time_limit is not None:
                    remaining = max(0, time_limit - (time.time() - start_time))
                    print(f"Gen {gen}: Min={record.get('min', 'N/A')}, "
                          f"Avg={record.get('avg', 'N/A'):.1f}, "
                          f"Std={record.get('std', 'N/A'):.1f}, "
                          f"Div={diversity:.3f}, Mut={current_mut_prob:.2f}, "
                          f"Time left={remaining:.1f}s")
                else:
                    print(f"Gen {gen}: Min={record.get('min', 'N/A')}, "
                          f"Avg={record.get('avg', 'N/A'):.1f}, "
                          f"Std={record.get('std', 'N/A'):.1f}, "
                          f"Div={diversity:.3f}, Mut={current_mut_prob:.2f}")

            gen += 1

        if verbose and time_limit is not None and time.time() - start_time >= time_limit:
            print(f"\nTime limit ({time_limit}s) reached. Stopping after {gen-1} generations.")

        return hof[0], logbook

    # ----------------------------------------------------------------------
    # Utility: export schedule for GUI plotting
    # ----------------------------------------------------------------------
    def schedule_to_list(self, schedule):
        """
        Convert schedule dictionary to a list of operations with:
        (workstation_index, machine_index, start, end, job_id, op_index)
        """
        ops_list = []
        for (ws, m), ops in schedule.items():
            for (start, end, job, op_idx) in ops:
                ops_list.append((ws, m, start, end, job, op_idx))
        return ops_list


# ----------------------------------------------------------------------
# Example usage (runs only if script is executed directly)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # This is the same woven label case study, but now passed as parameters
    workstation_names = ["Ultrasonic Splitting", "Coating", "Sizing", "Joining", "Cut and Fold"]
    machines_per_ws = [5, 1, 4, 6, 12]

    job_routings = [
        [2,3,4], [2,3,4], [2,3,4], [2,3,4],
        [0,2,3,4], [0,2,3,4], [0,2,3,4], [0,2,3,4],
        [0,2,3,4], [0,2,3,4], [0,2,3,4], [0,2,3,4],
        [0,2,3,4], [0,2,3,4], [0,2,3,4],
        [0,1,2,3,4]
    ]

    p_times = {
        (0,4):6, (0,5):6, (0,6):6, (0,7):6, (0,8):6, (0,9):8, (0,10):8, (0,11):6,
        (0,12):6, (0,13):5, (0,14):5, (0,15):5,
        (1,15):2,
        (2,0):1, (2,1):4, (2,2):4, (2,3):4, (2,4):4, (2,5):4, (2,6):4, (2,7):4,
        (2,8):4, (2,9):5, (2,10):5, (2,11):4, (2,12):4, (2,13):3, (2,14):3, (2,15):3,
        (3,0):2, (3,1):5, (3,2):4, (3,3):4, (3,4):5, (3,5):4, (3,6):4, (3,7):4,
        (3,8):4, (3,9):6, (3,10):6, (3,11):4, (3,12):4, (3,13):4, (3,14):4, (3,15):4,
        (4,0):5, (4,1):12, (4,2):11, (4,3):11, (4,4):12, (4,5):11, (4,6):11, (4,7):11,
        (4,8):12, (4,9):12, (4,10):22, (4,11):16, (4,12):16, (4,13):14, (4,14):14, (4,15):13
    }

    random.seed(42)
    np.random.seed(42)

    print("="*70)
    print("ENHANCED HYBRID GA FOR FJSP (Woven Label Case Study)")
    print("="*70)

    solver = FlexibleJobShopSolver(workstation_names, machines_per_ws, job_routings, p_times)

    start_t = time.time()
    # Example: run for 60 seconds (set time_limit=None to run for ngen generations)
    best_ind, logbook = solver.run_ga(pop_size=200, ngen=200,
                                      base_mut_prob=0.3, max_mut_prob=0.9,
                                      cx_prob=0.8, elite_size=3,
                                      stagnation_limit=25, diversity_threshold=0.15,
                                      time_limit=60)  # <-- set desired time limit in seconds
    best_makespan, best_sched = solver.decode_chromosome(best_ind)
    elapsed = time.time() - start_t

    print(f"\n{'='*70}")
    print(f"OPTIMAL MAKESPAN: {best_makespan} slots ({best_makespan*30/60:.1f} hours)")
    print(f"Total runtime: {elapsed:.2f} seconds")
    print(f"{'='*70}")