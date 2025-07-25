#  where to fly is a huge problem with heavy computation. Now try to avoid unnecessary computation using heuristic.
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line
import math
import time
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
# matplotlib.use('TkAgg')
from Utilities import *

class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0  # Ground cost
        self.h = 0  # Heuristic
        self.f = 0  # Total cost
        self.mode = 'ground'  # 'ground' or 'air'

    def __eq__(self, other):
        return self.position == other.position


def amphibious_a_star(terrain_map, road_map, road, keypoints, remaining_points, start, end, plot_flag=False,
                      phase_shift=False,
                      heuristic_factor=1.0,
                      search_radius=30.0,
                      search_angle=45.0,
                      time_weight=0.5,
                      scaler = 1):
    """
    Modified A* algorithm that searches in a sector towards the goal and
    considers both ground and air movement costs.

    Parameters:
        terrain_map: 2.5D terrain map with elevation data
        road: Array of road network points (x,y)
        start: Starting position (x,y)
        end: Goal position (x,y)
        plot_flag: Control dynamic plot
        heuristic_factor: Weight for heuristic
        search_radius: Radius for sector search (pixels)
        search_angle: Angle spread for sector search (degrees)
        time_weight: Weight for time in cost calculation (0-1)
        energy_weight: Weight for energy in cost calculation (0-1)
    """

    # Check if start and end are on road network
    if not np.any(np.all(road == start, axis=1)):
        raise ValueError("Start point is not on the road network")
    if not np.any(np.all(road == end, axis=1)):
        raise ValueError("End point is not on the road network")

    start = (start[0], start[1])
    end = (end[0], end[1])

    # Initialize start and end nodes
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    start_node.z = terrain_map[start[0], start[1]]

    end_node = Node(None, end)
    end_node.z = terrain_map[end[0], end[1]]

    open_list = []
    closed_list = []
    open_list.append(start_node)
    pre_draw = True
    if plot_flag:

        plt.ion()  # 开启交互模式
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.canvas.manager.set_window_title("Planning process")
        plt.tight_layout()

        dynamic_plot_terrain_and_nodes(terrain_map, road_map, start,None, start, end, pre_draw, ax=ax)
        # 替换原来的dynamic_plot调用
        # update_plot(
        #     terrain_map=terrain_map,
        #     road_map=road_map,
        #     nodes=(start[1], start[0]),  # (x,y)格式
        #     path=None,  # 转换为(y,x)格式
        #     start=start,
        #     goal=goal,
        #     ax=ax
        # )
        pre_draw = False
    t_s = time.time()
    while len(open_list) > 0:
        t0 = time.time()
        # Get node with lowest f cost
        current_node = min(open_list, key=lambda x: x.f)
        # print(current_node.position)
        open_list.remove(current_node)
        closed_list.append(current_node)
        t1 = time.time()
        if plot_flag:
            dynamic_plot_terrain_and_nodes(terrain_map, road_map, current_node.position, None, start, end, pre_draw, ax=ax)
            # update_plot(
            #     terrain_map=terrain_map,
            #     road_map=road_map,
            #     nodes=(current_node.position[1], current_node.position[0]),  # (x,y)格式
            #     path= None,  # 转换为(y,x)格式
            #     start=start,
            #     goal=goal,
            #     ax=ax
            # )

        # Check if we've reached the goal
        # if np.linalg.norm(np.array(current_node.position) - np.array(end)) < 1:
        if  np.array_equal(np.array(current_node.position), np.array(end)):
            path = []
            current = current_node
            while current is not None:
                path.append((current.position[0], current.position[1],
                             terrain_map[current.position[0], current.position[1]],
                             current.mode))
                current = current.parent

            nodes = path[::-1]
            processed_nodes = []
            i = 0
            n = len(nodes)

            while i < n:
                current_node = nodes[i]
                current_type = current_node[3]

                # 检查当前节点是否是 'air'，并且下一个节点也是 'air'
                if current_type == 'air' and i + 1 < n and nodes[i + 1][3] == 'air':
                    # 跳过当前节点（即去掉前面一个 'air'），保留下一个 'air'
                    i += 1  # 直接跳到下一个 'air'，当前不加入
                else:
                    processed_nodes.append(current_node)
                    i += 1

            return processed_nodes, len(open_list) + len(closed_list)

        t2 = time.time()

        # Find candidate points in sector towards goal

        if phase_shift is False:
            candidates = find_candidates_in_sector_2(
                current_node.position, end, keypoints, terrain_map,
                search_radius, search_angle, N=6
            )
            # candidates_2 = find_candidates_in_sector_3(
            #     current_node.position, end, remaining_points, terrain_map,
            #     search_radius, search_angle, N=10
            # )
        else:
            candidates = find_candidates_in_sector_2(
                current_node.position, end, road, terrain_map,
                search_radius, search_angle, N=6
            )

        t3 = time.time()
        if plot_flag:
            for candidate_pos in candidates:
                node_x, node_y = candidate_pos
                ax.scatter(node_x, node_y, c='red', s=20, edgecolors='gray',
                           zorder=4, alpha=0.1, label='planned nodes')
                # update_plot(
                #     terrain_map=terrain_map,
                #     road_map=road_map,
                #     nodes=(node_x, node_y),  # (x,y)格式
                #     path=None,  # 转换为(y,x)格式
                #     start=start,
                #     goal=goal,
                #     ax=ax
                # )

        for candidate_pos in candidates:
            # Create new node
            new_node = Node(current_node, candidate_pos)
            new_node.z = terrain_map[candidate_pos[0], candidate_pos[1]]

            # Skip if already in closed list
            if new_node in closed_list:
                continue

            # Determine movement mode and calculate costs
            # if is_directly_connected(current_node.position, new_node.position, road):
            # Ground movement
            # new_node.mode = 'ground'
            # ground_path = [current_node.position, new_node.position]
            ground_path, _, _  = a_star_3Droad(terrain_map, road, current_node.position, new_node.position, None, heuristic_factor=1)
            if ground_path is not None:
                ground_path_3D = extract_3d_path(ground_path, terrain_map)
                ground_time_cost = calculate_road_path_time(ground_path_3D, scaler)
                ground_energy_cost = calculate_road_path_energy(ground_path_3D, scaler)

                sum_ground_cost = time_weight * ground_time_cost + (1-time_weight)*0.01 * ground_energy_cost
                ground_path_h = [new_node.position, end]
                ground_path_3D_h = extract_3d_path(ground_path_h, terrain_map)
                ground_time_cost_h = calculate_road_path_time(ground_path_3D_h, scaler)
                ground_energy_cost_h = calculate_road_path_energy(ground_path_3D_h, scaler)
                sum_ground_cost_h = time_weight * ground_time_cost_h + (1-time_weight)*0.01 * ground_energy_cost_h
            else:
                sum_ground_cost = sum_ground_cost_h = np.inf
            # else:
                # Air movement
            # new_node.mode = 'air'
            airpath_3D = generate_air_path(current_node.position, new_node.position, terrain_map, ascent_height=0)
            air_time_cost = calculate_air_path_time(airpath_3D, scaler)
            air_energy_cost = calculate_air_path_energy(airpath_3D, scaler)
            sum_air_cost = time_weight * air_time_cost + (1-time_weight)*0.01*air_energy_cost

            airpath_3D_h = generate_air_path(new_node.position, end, terrain_map, ascent_height=0)
            air_time_cost_h = calculate_air_path_time(airpath_3D_h, scaler)
            air_energy_cost_h = calculate_air_path_energy(airpath_3D_h, scaler)
            sum_air_cost_h = time_weight * air_time_cost_h + (1-time_weight)*0.01*air_energy_cost_h

            # print('time cost: ', time_cost)
            # print('energy cost: ', energy_cost)

            # Combined cost with weights
            if sum_ground_cost <= sum_air_cost:
                sum_cost = sum_ground_cost
                sum_cost_h = sum_ground_cost_h
                new_node.mode = 'ground'
            else:
                sum_cost = sum_air_cost
                sum_cost_h = sum_air_cost_h
                new_node.mode = 'air'
            new_node.g = current_node.g + sum_cost
            new_node.h = sum_cost_h
            # # Heuristic (Euclidean distance)
            # new_node.h = np.linalg.norm(
            #     np.array([new_node.position[0], new_node.position[1], new_node.z]) -
            #     np.array([end[0], end[1], terrain_map[end[0], end[1]]])
            # )
            # print(np.linalg.norm(
            #     np.array([new_node.position[0], new_node.position[1], new_node.z]) -
            #     np.array([end[0], end[1], terrain_map[end[0], end[1]]])
            # ), new_node.h)

            new_node.f = new_node.g + heuristic_factor*new_node.h

            # Add to open list if better path found
            existing_node = next(
                (node for node in open_list if node == new_node), None
            )

            if existing_node is None or new_node.g < existing_node.g:
                open_list.append(new_node)
        t4 = time.time()
        # print("Phase 1 time cost is: {:.4f}".format(t1 - t0))
        # print("Phase 2 time cost is: {:.4f}".format(t2 - t1))
        # print("Phase 3 time cost is: {:.4f}".format(t3 - t2))
        # print("Phase 4 time cost is: {:.4f}".format(t4 - t3))  # 注意：这里可能应该是 t4 - t3？
        # t_e = time.time()
        # if t_e-t_s > 300:
        #     print("Take too much time to search!")
        #     break
    # 动态结束
    if plot_flag:
        plt.ioff()
        plt.show()
    return [], None  # No path found



if __name__ == '__main__':
    # Example usage
    Place = "dalingshan"
    start = (9, 32)  # (9, 42) # (10, 78)  # Must be on road network
    goal = (86, 92)  # (76, 65) # (95, 23) # (45, 45) ## Must be on road networkk
    scaler = 40
    # 加载保存的结果
    terrain_map = np.load(Place + '_terrain_map_100.npy')
    road_map = np.load(Place + '_road_map_100.npy')
    road = np.load(Place + '_road_2d_100.npy')
    road_3d = np.load(Place + '_road_3d_100.npy')

    # 转换为numpy数组并构建KD-Tree（建议在外部预构建）
    # road_kdtree = KDTree(np.array(road))
    # find_candidates_in_sector_3.road_kdtree = road_kdtree
    keypoints, remaining_points = extract_key_points_from_roadnet(road, goal)

    print("Num of keypoints and remaining_points: ", len(keypoints), len(remaining_points))
    # Find path with sector search (30 pixel radius, 45 degree angle)
    t1 = time.time()
    path_nodes, searched_nodes = amphibious_a_star(
        terrain_map, road_map, road, keypoints, remaining_points, start, goal,
        plot_flag=True,
        phase_shift=True,
        heuristic_factor=1.58,  # 3
        search_radius=25,  # 25
        search_angle=124,  # 150
        time_weight=0.4,   # 0.25
        # energy_weight=0.5,   # 0.2   1-time_weight
        scaler=scaler
    )
    t2 = time.time()
    print("Run Time cost is: {:.4f}".format(t2 - t1))
    print("nodes: ", path_nodes)
    # print('number of searched nodes: ', searched_nodes)
    # print('number of road_3d nodes: ', len(road_3d))
    if path_nodes:
        ground_paths_, air_paths = extract_ground_and_air_paths(path_nodes, terrain_map, road)
        # print(ground_paths_[-1][-1])
        ground_paths = remove_dead_end(ground_paths_)
        # print(ground_paths[-1][-1])
        # ground_paths = []
        # if len(ground_paths_) > 0:  # 移除死路
        #     for indx in range(len(ground_paths_)):
        #         path_2d = np.array([[int(x), int(y)] for x, y, z in ground_paths_[indx]])
        #         indx_temp, indx_a_temp, indx_b_temp = 0, 0, 0
        #         first_flag = True
        #         for indx_a, point_a in enumerate(path_2d):
        #             for indx_b, point_b in enumerate(path_2d):
        #                 if indx_a < indx_b and point_a[0] == point_b[0] and point_a[1] == point_b[1]:
        #                     print(indx, indx_a, indx_b)
        #                     if first_flag:
        #                         indx_a_temp, indx_b_temp = indx_a, indx_b
        #                         ground_paths.append(ground_paths_[indx][0:indx_a+1])
        #                         first_flag = False
        #                     elif indx_a > indx_b_temp:
        #                         ground_paths.append(ground_paths_[indx][indx_b_temp:indx_a+1])
        #                         indx_a_temp = indx_a
        #                         indx_b_temp = indx_b
        #
        #         ground_paths.append(ground_paths_[indx][indx_b_temp:])

        # print(ground_paths)

        print(f"Ground path length: {len(ground_paths)}")
        print(f"Air path length: {len(air_paths)}")

        # 计算总时间和能量
        total_time = (calculate_road_path_time_multi(ground_paths, scaler) +
                      calculate_air_path_time_multi(air_paths, scaler))

        total_energy = (calculate_road_path_energy_multi(ground_paths, scaler) +
                        calculate_air_path_energy_multi(air_paths, scaler))

        print(f"Total time cost: {total_time:.2f} s")
        print(f"Total energy cost: {total_energy / 1000:.2f} kJ")  # Convert from J to kJ

        # Visualize results
        plot_2d_path(terrain_map, road_map, ground_paths, air_paths, start, goal)

        plot_3d_path(terrain_map, road_3d, ground_paths, air_paths, start, goal)


        print('End..')


