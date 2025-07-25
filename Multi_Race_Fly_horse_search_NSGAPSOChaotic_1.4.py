# Optimized the standard NSGA_III using the PSO and Chaos mechanisms
import time
import numpy as np
from Astar_25D_road_Amphibious_6 import *  # Ensure path planning functions are included
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, freeze_support
from loky import get_reusable_executor
import math
import matplotlib.pyplot as plt
from scipy.spatial import distance
from itertools import combinations
# from NSGAPSOChaos_VS_NSGAIII import *

import os

os.environ['LOKY_MAX_CPU_COUNT'] = '60'  # Set maximum cores to use


# Evaluation function
def evaluate_individual(params, terrain_map, road_map, road,  keypoints, remaining_points, start, goal, scaler, phase_shift):
    heuristic_factor, search_radius, search_angle, time_weight = params
    # print("heuristic_factor, search_radius, search_angle, time_weight: ", heuristic_factor, search_radius, search_angle,
    #       time_weight)
    path_nodes, searched_nodes = amphibious_a_star(terrain_map,
                                                   road_map,
                                                   road,
                                                   keypoints,
                                                   remaining_points,
                                                   start,
                                                   goal,
                                                   False,
                                                   phase_shift,
                                                   heuristic_factor,
                                                   search_radius,
                                                   search_angle,
                                                   time_weight,
                                                   scaler)
    if len(path_nodes) < 2:
        # print("Invalid parameter combination:", params)
        # print('path_nodes:', path_nodes)
        return np.inf, np.inf

    else:
        # path_nodes:
        ground_paths_, air_paths = extract_ground_and_air_paths_2(path_nodes, terrain_map, road)
        ground_paths = remove_dead_end(ground_paths_)

        total_time = (calculate_road_path_time_multi(ground_paths, scaler) +
                      calculate_air_path_time_multi(air_paths, scaler))

        total_energy = (calculate_road_path_energy_multi(ground_paths, scaler) +
                        calculate_air_path_energy_multi(air_paths, scaler))
        # print('Total time:', total_time)
        # print('Total energy:', total_energy)
        return total_time, total_energy
    # else:
    #     return np.inf, np.inf
    # print("evaluate_individual finished!")
    # if total_time == 0 or total_energy == 0:
    #     print("No path found")
    #     return np.inf, np.inf
    # else:
    #     return total_time, total_energy


# Parallel evaluation function
from concurrent.futures import ProcessPoolExecutor  # 更可控的进程池

def evaluate_population(population, terrain_map, road_map, road, keypoints, remaining_points, start, goal, scaler, phase_shift):
    """
    并行评估种群，支持传入 terrain_map, road, start, goal 参数
    """
    # 使用 ProcessPoolExecutor 替代 get_reusable_executor，更可控
    with get_reusable_executor() as executor:
        # 使用 partial 固定 terrain_map, road, start, goal 参数
        from functools import partial
        evaluate_func = partial(evaluate_individual,
                                terrain_map=terrain_map,
                                road_map=road_map,
                                road=road,
                                keypoints=keypoints,
                                remaining_points=remaining_points,
                                start=start,
                                goal=goal,
                                scaler=scaler,
                                phase_shift= phase_shift)

        # 并行评估
        fitness = list(tqdm(executor.map(evaluate_func, population),
                            total=len(population), desc="Evaluating fitness"))
    return np.array(fitness)

# def evaluate_population(population, terrain_map, road, start, goal):
#     fitness = []
#     for individual in tqdm(population, desc="Evaluating fitness"):
#         fitness.append(evaluate_individual(individual, terrain_map, road, start, goal))
#     return np.array(fitness)


# ==== 核心组件实现 ====
class EliteArchive:
    def __init__(self, max_size):
        self.max_size = max_size
        self.solutions = np.empty((0, 4))
        self.fitness = np.empty((0, 2))

    def update(self, population, fitness):
        combined_pop = np.vstack([self.solutions, population])
        combined_fitness = np.vstack([self.fitness, fitness])

        fronts = non_dominated_sort(combined_fitness)
        new_pop = np.empty((0, 4))
        new_fitness = np.empty((0, 2))

        for front in fronts:
            if new_pop.shape[0] + len(front) > self.max_size:
                remaining = self.max_size - new_pop.shape[0]
                ref_points = generate_reference_points(2, 12)
                selected = self.reference_point_selection(
                    combined_pop[front], combined_fitness[front], ref_points, remaining)
                new_pop = np.vstack([new_pop, combined_pop[front][selected]])
                new_fitness = np.vstack([new_fitness, combined_fitness[front][selected]])
                break
            else:
                new_pop = np.vstack([new_pop, combined_pop[front]])
                new_fitness = np.vstack([new_fitness, combined_fitness[front]])

        self.solutions = new_pop
        self.fitness = new_fitness

    @staticmethod
    def reference_point_selection(population, fitness, ref_points, n_select):
        """参考点关联选择"""
        normalized_fitness = (fitness - fitness.min(axis=0)) / (np.ptp(fitness, axis=0) + 1e-6)

        association = [[] for _ in range(len(ref_points))]
        distances = np.linalg.norm(normalized_fitness[:, None] - ref_points, axis=2)
        closest_ref = np.argmin(distances, axis=1)

        for i, r in enumerate(closest_ref):
            association[r].append(i)

        selected = []
        while len(selected) < n_select:
            min_count = np.inf
            for r in range(len(ref_points)):
                if 0 < len(association[r]) < min_count:
                    min_count = len(association[r])
                    target_r = r
            if len(association[target_r]) > 0:
                selected.append(association[target_r].pop())
        return selected

# ================== Standard NSGA-III Implementation ==================

def non_dominated_sort(fitness):
    """Fast non-dominated sort"""
    S = [[] for _ in range(len(fitness))]
    fronts = [[]]
    n = np.zeros(len(fitness), dtype=int)
    rank = np.zeros(len(fitness), dtype=int)

    for i in range(len(fitness)):
        for j in range(len(fitness)):
            if i != j:
                if dominates(fitness[i], fitness[j]):
                    S[i].append(j)
                elif dominates(fitness[j], fitness[i]):
                    n[i] += 1
        if n[i] == 0:
            rank[i] = 0
            fronts[0].append(i)

    i = 0
    while len(fronts[i]) > 0:
        Q = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i += 1
        fronts.append(Q)

    return fronts[:-1]


def dominates(a, b):
    """Returns True if a dominates b"""
    return np.all(a <= b) and np.any(a < b)


def generate_reference_points(num_objectives, num_divisions, inner_point=True):
    """Generate reference points uniformly on the hyperplane"""
    ref_points = []
    if num_objectives == 2:
        for i in range(num_divisions + 1):
            ref_points.append([i / num_divisions, (num_divisions - i) / num_divisions])
    elif num_objectives == 3:
        for i in range(num_divisions + 1):
            for j in range(num_divisions + 1 - i):
                k = num_divisions - i - j
                ref_points.append([i / num_divisions, j / num_divisions, k / num_divisions])
    else:
        # For higher dimensions, use Das and Dennis's systematic approach
        ref_points = das_dennis(num_divisions, num_objectives)

    ref_points = np.array(ref_points)

    # Remove the all-zero point if inner_point is False
    if not inner_point:
        ref_points = ref_points[np.sum(ref_points, axis=1) > 0]

    return ref_points


def das_dennis(n_partitions, n_dim):
    """Das and Dennis's systematic approach for generating reference points"""
    if n_dim == 1:
        return np.array([[1.0]])

    ref_points = []
    for i in range(n_partitions + 1):
        ref_points_inner = das_dennis(n_partitions - i, n_dim - 1)
        for ref in ref_points_inner:
            ref_points.append(np.concatenate(([i / n_partitions], ref)))

    return np.array(ref_points)


def associate_to_reference_points(population, fitness, ref_points, ideal_point=None):
    """Associate population members to reference points"""
    if ideal_point is None:
        ideal_point = np.min(fitness, axis=0)

    # Translate objectives
    translated_fitness = fitness - ideal_point

    # Calculate reference line vectors
    ref_dirs = ref_points / np.linalg.norm(ref_points, axis=1)[:, np.newaxis]

    # Calculate perpendicular distances
    distances = np.zeros((len(population), len(ref_points)))
    for i in range(len(population)):
        for j in range(len(ref_points)):
            distances[i, j] = perpendicular_distance(translated_fitness[i], ref_dirs[j])

    # Find closest reference point for each individual
    closest_ref = np.argmin(distances, axis=1)

    return closest_ref, distances


def perpendicular_distance(point, ref_dir):
    """Calculate perpendicular distance from point to reference direction"""
    norm_ref_dir = np.linalg.norm(ref_dir)
    if norm_ref_dir == 0:
        return np.inf
    return np.linalg.norm(point - np.dot(point, ref_dir) * ref_dir / (norm_ref_dir ** 2))



# def niching_selection(population, fitness, fronts, ref_points, ideal_point=None):
#     """Select individuals using niching based on reference points"""
#     selected = []
#     remaining = population_size
#
#     # First add all individuals from the first fronts that can be fully accommodated
#     for front in fronts:
#         if len(front) <= remaining:
#             selected.extend(front)
#             remaining -= len(front)
#         else:
#             # Need to select from this front
#             break
#
#     if remaining > 0:
#         # Need to select from the last front that couldn't be fully accommodated
#         last_front = fronts[len(selected) // population_size]
#
#         # Associate individuals to reference points
#         closest_ref, distances = associate_to_reference_points(population[last_front],
#                                                                fitness[last_front],
#                                                                ref_points, ideal_point)
#
#         # Count how many individuals are associated to each reference point
#         ref_counts = np.zeros(len(ref_points), dtype=int)
#         for ref_idx in closest_ref:
#             ref_counts[ref_idx] += 1
#
#         # Select individuals
#         while remaining > 0:
#             # Find reference point with minimum associated individuals
#             min_count = np.min(ref_counts[ref_counts > 0])
#
#             candidates = np.where(ref_counts == min_count)[0]
#
#             # Randomly select one if there are multiple
#             selected_ref = np.random.choice(candidates)
#
#             # Find all individuals in last_front associated with selected_ref
#             associated_indices = np.where(closest_ref == selected_ref)[0]
#
#             if len(associated_indices) > 0:
#                 # Select the individual with smallest perpendicular distance
#                 best_idx = associated_indices[np.argmin(distances[associated_indices, selected_ref])]
#                 selected.append(last_front[best_idx])
#                 remaining -= 1
#
#                 # Remove this individual from consideration
#                 closest_ref[best_idx] = -1  # Mark as selected
#                 ref_counts[selected_ref] -= 1
#
#     return population[selected]

def niching_selection(population, population_size, fitness, fronts, ref_points, ideal_point=None):
    """Select individuals using niching based on reference points"""
    selected = []
    remaining = population_size

    # First add all individuals from the first fronts that can be fully accommodated
    for front in fronts:
        if len(front) <= remaining:
            selected.extend(front)
            remaining -= len(front)
        else:
            # Need to select from this front
            break

    if remaining > 0:
        # Need to select from the last front that couldn't be fully accommodated
        last_front = fronts[len(selected) // population_size]

        # Associate individuals to reference points
        closest_ref, distances = associate_to_reference_points(population[last_front],
                                                             fitness[last_front],
                                                             ref_points, ideal_point)

        # Count how many individuals are associated to each reference point
        ref_counts = np.zeros(len(ref_points), dtype=int)
        for ref_idx in closest_ref:
            ref_counts[ref_idx] += 1

        # Select individuals
        while remaining > 0:
            # Find reference point with minimum associated individuals
            # Handle case where no reference points have positive counts
            if np.any(ref_counts > 0):
                min_count = np.min(ref_counts[ref_counts > 0])
                candidates = np.where(ref_counts == min_count)[0]
            else:
                # If no reference points have positive counts, select randomly
                candidates = np.where(ref_counts == 0)[0]

            # Randomly select one if there are multiple
            selected_ref = np.random.choice(candidates)

            # Find all individuals in last_front associated with selected_ref
            associated_indices = np.where(closest_ref == selected_ref)[0]

            if len(associated_indices) > 0:
                # Select the individual with smallest perpendicular distance
                best_idx = associated_indices[np.argmin(distances[associated_indices, selected_ref])]
                selected.append(last_front[best_idx])
                remaining -= 1

                # Remove this individual from consideration
                closest_ref[best_idx] = -1  # Mark as selected
                ref_counts[selected_ref] -= 1

    return population[selected]

def normalize_fitness(fitness, ideal_point=None, nadir_point=None):
    """Normalize fitness values"""
    if ideal_point is None:
        ideal_point = np.min(fitness, axis=0)
    if nadir_point is None:
        nadir_point = np.max(fitness, axis=0)

    # Avoid division by zero
    nadir_point = np.where(nadir_point == ideal_point, ideal_point + 1e-10, nadir_point)

    return (fitness - ideal_point) / (nadir_point - ideal_point)


def find_extreme_points(fitness):
    num_objectives = fitness.shape[1]
    extreme_points = np.zeros((num_objectives, num_objectives))
    min_weight = 1e-8  # 最小权重值

    for i in range(num_objectives):
        weights = np.full(num_objectives, min_weight)  # 所有目标都有微小权重
        weights[i] = 1.0  # 当前目标权重为1

        asf_values = np.max(fitness / weights, axis=1)
        extreme_points[i] = fitness[np.argmin(asf_values)]

    return extreme_points


def calculate_nadir_point(extreme_points, ideal_point):
    """Calculate nadir point from extreme points"""
    # Solve the linear system to find the intercepts
    num_objectives = extreme_points.shape[0]
    A = extreme_points - ideal_point
    b = np.ones(num_objectives)

    try:
        x = np.linalg.solve(A, b)
        intercepts = ideal_point + 1 / x
        return intercepts
    except np.linalg.LinAlgError:
        # If the system is singular, return the maximum values
        return np.max(extreme_points, axis=0)


# ================== PSO and Chaotic Search Enhancements ==================

def initialize_population(population_size, param_ranges):
    pop = np.zeros((population_size, 4))
    for i in range(4):
        low, high = param_ranges[i]
        pop[:, i] = np.random.uniform(low, high, population_size)
    return np.clip(pop, [r[0] for r in param_ranges], [r[1] for r in param_ranges])


def pso_update_velocity(position, velocity, pbest, gbest, w=0.7, c1=1.5, c2=1.5):
    r1, r2 = np.random.rand(2)
    new_velocity = (w * velocity +
                    c1 * r1 * (pbest - position) +
                    c2 * r2 * (gbest - position))
    return new_velocity


def pso_generate_offspring(population, param_ranges, pbest_pop, gbest, velocity):
    offspring = []
    for i in range(len(population)):
        velocity[i] = pso_update_velocity(population[i], velocity[i], pbest_pop[i], gbest)
        new_ind = population[i] + velocity[i]
        new_ind[-2:] = np.round(new_ind[-2:])
        new_ind = np.clip(new_ind, [r[0] for r in param_ranges], [r[1] for r in param_ranges])
        offspring.append(new_ind)
    return np.array(offspring), velocity


def chaotic_mapping(x, mu=3.9):
    return mu * x * (1 - x)


def generate_chaotic_individual(current_population, param_ranges, chaos_var):
    base = current_population[np.random.randint(len(current_population))]
    chaotic_params = base.copy()
    for i in range(4):
        chaos_var[i] = chaotic_mapping(chaos_var[i])
        chaotic_params[i] = param_ranges[i][0] + chaos_var[i] * (param_ranges[i][1] - param_ranges[i][0])
    return np.clip(chaotic_params, [r[0] for r in param_ranges], [r[1] for r in param_ranges]), chaos_var


def spacing_metric(front_indices, fitness):
    """
    计算第一前沿的 Spacing Metric
    输入:
        front_indices (list): 第一前沿的解的索引（如 [0, 4, 5]）
        fitness (np.ndarray): 所有解的目标值，形状 (n, m)，n=解总数，m=目标数
    返回:
        S (float): spacing 值
    """
    front = np.array([fitness[i] for i in front_indices])  # 通过索引获取目标值
    if front.ndim == 1:
        front = front.reshape(-1, 1)
    n = len(front)
    # if n <= 1:
    #     return 0.0

    distances = np.sum(np.abs(front[:, np.newaxis, :] - front[np.newaxis, :, :]), axis=2)
    np.fill_diagonal(distances, np.inf)
    d = np.min(distances, axis=1)
    d_bar = np.mean(d)
    S = np.sqrt(np.sum((d - d_bar) ** 2) / (n - 1))
    return S


def diversity_metric(front_indices, fitness):
    """
    计算第一前沿的 Diversity Metric
    输入:
        front_indices (list): 第一前沿的解的索引（如 [0, 4, 5]）
        fitness (np.ndarray): 所有解的目标值，形状 (n, m)，n=解总数，m=目标数
    返回:
        D (float): 多样性值
    """
    front = np.array([fitness[i] for i in front_indices])
    if front.ndim == 1:
        front = front.reshape(-1, 1)
    if len(front) <= 1:
        return 0.0

    extreme_points = np.min(front, axis=0), np.max(front, axis=0)
    diagonal_length = np.linalg.norm(extreme_points[1] - extreme_points[0])
    if diagonal_length == 0:
        return 0.0

    distances = np.linalg.norm(front[:, np.newaxis, :] - front[np.newaxis, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    mean_min_distance = np.mean(np.min(distances, axis=1))
    D = mean_min_distance / diagonal_length
    return D


import argparse
# import numpy as np
# import time
# import matplotlib.pyplot as plt
# from multiprocessing import freeze_support


def main(Place, Case, start, goal, scaler, num_generations=50, population_size=100,
         crossover_rate=0.9, mutation_rate=0.8, elite_ratio=0.3, pso_ratio=0.1,
         standard_ratio=0.8, phase_shift_rate=0.6, plot_flag=False):
    freeze_support()

    # 加载保存的结果
    terrain_map = np.load(Place + '_terrain_map_100.npy')
    road_map = np.load(Place + '_road_map_100.npy')
    road = np.load(Place + '_road_2d_100.npy')
    road_3d = np.load(Place + '_road_3d_100.npy')

    keypoints, remaining_points = extract_key_points_from_roadnet(road, goal)

    save_name = Place + "_St_NSGA_PSO_CHAOS_" + Case + "_.npz"

    # Algorithm parameters
    param_ranges = [(1.5, 3), (10, 35), (50, 180), (0.1, 0.9)]  # Parameter ranges

    # 剩下的原有代码...
    # [这里保留原有的算法实现部分，只是把硬编码的参数改为使用函数参数]
    # num_generations = 50
    # population_size = 100
    # crossover_rate = 0.9
    # mutation_rate = 0.8
    # elite_ratio = 0.3
    #
    # # PSO and chaos parameters
    # pso_ratio = 0.1  # 10% of offspring generated by PSO
    # chaos_ratio = 0.1  # 10% of offspring generated by chaos
    # standard_ratio = 0.8  # 80% standard NSGA-III operations
    # phase_shift_rate = 0.6
    # plot_flag = False

    # Algorithm initialization
    population = initialize_population(population_size, param_ranges)
    pbest_pop = population.copy()
    pbest_fitness = np.full((population_size, 2), np.inf)
    archive = EliteArchive(max_size=int(population_size * elite_ratio))
    velocity = np.zeros_like(population)
    gbest = None
    chaos_var = np.random.rand(4)

    # Generate reference points for NSGA-III
    ref_points = generate_reference_points(num_objectives=2, num_divisions=12, inner_point=False)

    if plot_flag:
        # Visualization setup
        plt.ion()
        fig, ax = plt.subplots(figsize=(6, 4))
        plt.title("NSGA-III with PSO and Chaos Optimization")
        ax.set_xlabel('Time Cost (s)')
        ax.set_ylabel('Energy Cost (J)')
        plt.pause(0.1)

    # Track best solutions
    all_solutions = []
    all_fitness = []
    phase_shift = False
    Spacing = []
    Diversity = []
    # Evolutionary loop
    for gen in range(num_generations):
        print(f"\n=== Generation {gen + 1}/{num_generations} ===")
        t1 = time.time()
        if gen >= num_generations * phase_shift_rate and phase_shift == False:
            phase_shift = True
            print("Now shift into global road_point search phase..")
        # 1. Evaluate current population
        fitness = evaluate_population(population, terrain_map, road_map, road, keypoints, remaining_points, start, goal, scaler, phase_shift)

        all_solutions.append(population.copy())
        all_fitness.append(fitness.copy())

        # 3. Update PSO information
        improved_mask = np.any(fitness < pbest_fitness, axis=1)
        pbest_pop[improved_mask] = population[improved_mask]
        pbest_fitness[improved_mask] = fitness[improved_mask]

        # 更新精英库
        archive.update(population, fitness)

        # Non-dominated sorting
        fronts = non_dominated_sort(fitness)

        Spacing.append(spacing_metric(fronts[0], fitness))
        Diversity.append(diversity_metric(fronts[0], fitness))

        if len(fronts) > 0 and len(fronts[0]) > 0:
            gbest = population[fronts[0][np.random.randint(len(fronts[0]))]]

        # 4. Generate offspring using mixed strategies
        n_standard = int(population_size * standard_ratio)
        n_pso = int(population_size * pso_ratio)
        n_chaos = population_size - n_standard - n_pso

        offspring = []
        # Standard NSGA-III operations (crossover + mutation)
        for _ in range(n_standard):
            parent1, parent2 = population[np.random.choice(len(population), 2, replace=False)]

            # crossover
            alpha = np.random.rand(4)
            child = alpha * parent1 + (1 - alpha) * parent2

            # 引入历史最优引导
            if np.random.rand() < crossover_rate and len(pbest_pop) > 0:
                pbest = pbest_pop[np.random.randint(len(pbest_pop))]
                beta = np.random.rand(4)
                elite = archive.solutions[np.random.randint(len(archive.solutions))]
                child = beta * child + 0.3*(1 - beta) * (pbest + gbest + elite)

            child[-2:] = np.round(child[-2:])
            child = np.clip(child, [r[0] for r in param_ranges], [r[1] for r in param_ranges])

            # mutation
            for i in range(4):
                if np.random.rand() < mutation_rate:
                    delta = np.random.normal(0, 0.1) * (param_ranges[i][1] - param_ranges[i][0])

                    # 精英引导
                    if archive.solutions.size > 0:
                        elite = archive.solutions[np.random.randint(len(archive.solutions))][i]
                        delta += 0.15 * (elite - child[i])

                    # 个体历史引导
                    if pbest_pop.size > 0:
                        pbest = pbest_pop[np.random.randint(len(pbest_pop))][i]
                        delta += 0.15 * (pbest - child[i])

                    delta += 0.15 * (gbest[i] - child[i])

                    child[i] += delta

            child[-2:] = np.round(child[-2:])
            child = np.clip(child, [r[0] for r in param_ranges], [r[1] for r in param_ranges])
            offspring.append(child)

        # PSO operations
        if gbest is not None and n_pso > 0:
            pso_indices = np.random.choice(len(population), n_pso, replace=False)
            pso_parents = population[pso_indices]
            pso_pbest = pbest_pop[pso_indices]
            pso_velocity = velocity[pso_indices]

            pso_offspring, updated_velocity = pso_generate_offspring(
                pso_parents, param_ranges, pso_pbest, gbest, pso_velocity)
            offspring.extend(pso_offspring)
            velocity[pso_indices] = updated_velocity

        # Chaotic search
        for _ in range(n_chaos):
            chaotic_ind, chaos_var = generate_chaotic_individual(population, param_ranges, chaos_var)
            offspring.append(chaotic_ind)

        # Evaluate offspring
        offspring = np.array(offspring[:population_size])
        offspring_fitness = evaluate_population(offspring, terrain_map, road_map, road, keypoints, remaining_points, start, goal, scaler, phase_shift)

        # 5. Environmental selection (apply from first generation)
        combined_pop = np.vstack([population, offspring])
        combined_fitness = np.vstack([fitness, offspring_fitness])

        # Calculate ideal and nadir points
        ideal_point = np.min(combined_fitness, axis=0)
        extreme_points = find_extreme_points(combined_fitness)
        nadir_point = calculate_nadir_point(extreme_points, ideal_point)

        # Normalize fitness
        normalized_fitness = normalize_fitness(combined_fitness, ideal_point, nadir_point)

        # Non-dominated sorting
        fronts = non_dominated_sort(normalized_fitness)

        # Niching selection
        population = niching_selection(combined_pop, population_size, normalized_fitness, fronts, ref_points, ideal_point)
        population = population[:population_size]  # Ensure correct size

        if plot_flag:
            # Dynamic visualization - show current generation
            ax.clear()
            ax.scatter(normalized_fitness[:, 0], normalized_fitness[:, 1], c='skyblue', alpha=0.6,
                       label='Current Population')
            # fronts = non_dominated_sort(fitness)
            if len(fronts) > 0:
                # ax.clear()  # 清除当前轴而非重新创建
                first_front = fronts[0]
                ax.scatter(normalized_fitness[first_front, 0], normalized_fitness[first_front, 1],
                           c='red', s=60, alpha=0.8, label='Pareto Front')
            ax.set_title(f"NSGA-PSO-Chaos Generation {gen + 1}")
            ax.legend()
            plt.draw()
            plt.pause(0.1)

        t2 = time.time()
        print(
            f"Generation time: {t2 - t1:.2f}s | Best fitness: {np.min(fitness[:, 0]):.1f}s, {np.min(fitness[:, 1]):.1f}J")

    # Final processing
    if plot_flag:
        plt.ioff()

    # Get final fitness
    all_fitness_ = np.concatenate(all_fitness)
    all_solutions_ = np.concatenate(all_solutions)
    # final_fitness = evaluate_population(population)

    # Get non-dominated solutions
    fronts = non_dominated_sort(all_fitness_)
    first_front_indices = fronts[0]
    first_front = all_solutions_[first_front_indices]
    first_front_fitness = all_fitness_[first_front_indices]

    # Remove duplicates
    seen = set()
    unique_indices = []
    for idx, fit in enumerate(first_front_fitness):
        fit_tuple = tuple(fit)
        if fit_tuple not in seen:
            seen.add(fit_tuple)
            unique_indices.append(idx)

    unique_first_front = first_front[unique_indices]
    unique_first_front_fitness = first_front_fitness[unique_indices]

    # 最终保存结果部分
    np.savez(save_name,
             unique_first_front=unique_first_front,
             unique_first_front_fitness=unique_first_front_fitness,
             Spacing=Spacing,
             Diversity=Diversity)
    print(f"Results saved to {save_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-objective path planning with NSGA-III, PSO and Chaos')

    # 必需参数
    parser.add_argument('--place', type=str, required=True, help='Location name (e.g., dalingshan)')
    parser.add_argument('--case', type=str, required=True, help='Case identifier (e.g., case_2_6_server_1)')
    parser.add_argument('--start', type=int, nargs=2, required=True, help='Start coordinates (x y)')
    parser.add_argument('--goal', type=int, nargs=2, required=True, help='Goal coordinates (x y)')
    parser.add_argument('--scaler', type=float, required=True, help='Scale factor')

    # 可选参数（有默认值）
    parser.add_argument('--generations', type=int, default=30, help='Number of generations')
    parser.add_argument('--pop_size', type=int, default=50, help='Population size')
    parser.add_argument('--crossover', type=float, default=0.9, help='Crossover rate')
    parser.add_argument('--mutation', type=float, default=0.8, help ='Mutation rate')
    parser.add_argument('--elite', type=float, default=0.5, help='Elite ratio')
    parser.add_argument('--pso', type=float, default=0.1, help='PSO ratio')
    parser.add_argument('--chaos', type=float, default=0.1, help='Chaos ratio')
    parser.add_argument('--phase_shift', type=float, default=0.8, help='Phase shift rate')
    parser.add_argument('--plot', action='store_true', help='Enable plotting')

    args = parser.parse_args()

    main(Place=args.place,
         Case=args.case,
         start=tuple(args.start),
         goal=tuple(args.goal),
         scaler=args.scaler,
         num_generations=args.generations,
         population_size=args.pop_size,
         crossover_rate=args.crossover,
         mutation_rate=args.mutation,
         elite_ratio=args.elite,
         pso_ratio=args.pso,
         # chaos_ratio=args.chaos,
         phase_shift_rate=args.phase_shift,
         plot_flag=args.plot)

