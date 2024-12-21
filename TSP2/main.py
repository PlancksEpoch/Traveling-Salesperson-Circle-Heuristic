import numpy as np
import matplotlib.pyplot as plt

def load_tsp_data_from_file(filename):
    """Load TSP data from a plain text file containing coordinates."""
    cities = []
    with open(filename, 'r') as file:
        start_reading = False
        for line in file:
            if 'NODE_COORD_SECTION' in line:
                start_reading = True
                continue
            if 'EOF' in line:
                break
            if start_reading:
                parts = line.strip().split()
                if len(parts) == 3:  # Assuming format: identifier x y
                    _, x, y = parts
                    cities.append((float(x), float(y)))
    return cities

def load_optimal_tour(filename):
    """Load the optimal tour from a plain text file."""
    tour = []
    with open(filename, 'r') as file:
        start_reading = False
        for line in file:
            if 'TOUR_SECTION' in line:
                start_reading = True
                continue
            if start_reading:
                line = line.strip()
                if line.isdigit():
                    city = int(line)
                    if city == -1:
                        break
                    tour.append(city - 1)  # Convert to zero-indexed list
    if not tour:
        print("Warning: Optimal tour is empty.")
    return tour

def compute_distance_and_angle_matrices(cities):
    """Compute both distance and angle matrices for a list of cities."""
    n = len(cities)
    distance_matrix = np.zeros((n, n))
    angle_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = cities[j][0] - cities[i][0]
                dy = cities[j][1] - cities[i][1]
                distance_matrix[i][j] = np.hypot(dx, dy)
                angle_matrix[i][j] = np.degrees(np.arctan2(dy, dx)) % 360
    return distance_matrix, angle_matrix


def plot_tour(cities, tour, centroid, edge_nodes, title='Traveling Salesman Path'):
    if not tour:
        print(f"Warning: No tour data to plot for {title}. Check the tour data.")
        return

    try:
        # Debugging: Print lengths and sample contents
        print(f"Number of cities: {len(cities)}")
        print(f"Tour length: {len(tour)}")
        print(f"Edge nodes length: {len(edge_nodes)}")
        print(f"Tour indices: {tour}")
        print(f"Edge node indices: {edge_nodes}")

        # Ensure all tour indices are within the range of cities
        if max(tour) >= len(cities) or min(tour) < 0:
            raise ValueError("Tour indices exceed the number of cities or are negative.")

        # Unpack city coordinates for the tour
        x_coords, y_coords = zip(*[cities[i] for i in tour])

        # Debugging: Print coordinate lengths
        print(f"X coordinates length before closing the loop: {len(x_coords)}, Y coordinates length before closing the loop: {len(y_coords)}")

        # Close the loop by adding the first city at the end if it's not already there
        if tour[0] != tour[-1]:
            x_coords += (x_coords[0],)
            y_coords += (y_coords[0],)

        # Debugging: Print coordinate lengths after closing the loop
        print(f"X coordinates length after closing the loop: {len(x_coords)}, Y coordinates length after closing the loop: {len(y_coords)}")

        plt.figure(figsize=(12, 8))
        plt.plot(x_coords, y_coords, 'o-', label='Tour Path')  # Plot the tour path
        plt.scatter(x_coords[0], y_coords[0], color='red', s=200, label='Starting City')  # Starting city

        # Highlight the centroid
        plt.plot(centroid[0], centroid[1], 'x', color='gold', markersize=10, label='Centroid')

        # Highlight edge nodes
        edge_x, edge_y = zip(*[cities[i] for i in edge_nodes])
        plt.scatter(edge_x, edge_y, color='green', s=100, edgecolor='black', label='Edge Nodes')

        # Label each city with its original index
        for i, (x, y) in enumerate(zip(x_coords[:-1], y_coords[:-1])):  # Exclude the last point for labels
            plt.text(x, y, f'{tour[i] + 1}', fontsize=12, color='darkred')

        plt.title(title)
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"Error plotting {title}: {e}")
        print(f"Tour data: {tour}")
        print(f"Edge Nodes: {edge_nodes}")


def find_edge_nodes(cities, centroid, percentage=0.2):
    num_edges = max(int(len(cities) * percentage), 1)

    # Calculate distances from the centroid
    distances_from_centroid = [(index, np.hypot(city[0] - centroid[0], city[1] - centroid[1])) for index, city in enumerate(cities)]

    # Sort by distances
    distances_from_centroid.sort(key=lambda x: x[1], reverse=True)

    # Select edge nodes based on distance
    edge_nodes = [index for index, _ in distances_from_centroid[:num_edges]]

    return edge_nodes

def calculate_centroid(cities):
    x_coords = [city[0] for city in cities]
    y_coords = [city[1] for city in cities]
    centroid = (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))
    return centroid

# def find_next_city(current_index, cities, angle_matrix, distance_matrix, last_direction, edge_nodes, centroid, visited):
#     scores = []
#     current_centroid_angle = np.degrees(np.arctan2(cities[current_index][1] - centroid[1], cities[current_index][0] - centroid[0]))
#
#     for i in range(len(cities)):
#         if i != current_index and i not in visited:
#             current_angle = angle_matrix[current_index][i]
#             angular_deviation = min(abs(current_angle - last_direction), 360 - abs(current_angle - last_direction))
#             distance_score = distance_matrix[current_index][i]
#             is_edge = i in edge_nodes
#             candidate_centroid_angle = np.degrees(np.arctan2(cities[i][1] - centroid[1], cities[i][0] - centroid[0]))
#             angle_from_centroid_deviation = abs(current_centroid_angle - candidate_centroid_angle)
#
#             base_score = 0.2 * angular_deviation + 0.8 * distance_score
#
#             scores.append((base_score, i, is_edge, angle_from_centroid_deviation))
#
#     # Sort scores by the base score
#     scores.sort(key=lambda x: x[0])
#
#     # Check the top three scores
#     if len(scores) >= 3:
#         best_score = scores[0][0]
#         second_best_score = scores[1][0]
#         third_best_score = scores[2][0]
#
#         if (second_best_score - best_score < 10) and (third_best_score - second_best_score < 10):  # Adjust the threshold as needed
#             # Apply edge node preference
#             for idx in range(3):
#                 base_score, i, is_edge, angle_from_centroid_deviation = scores[idx]
#                 if is_edge and angle_from_centroid_deviation < 10:
#                     scores[idx] = (base_score * 0.9, i, is_edge, angle_from_centroid_deviation)
#
#     # Select the city with the lowest (possibly adjusted) score
#     scores.sort(key=lambda x: x[0])
#     best_score, next_city, is_edge, angle_from_centroid_deviation = scores[0]
#     last_direction = angle_matrix[current_index][next_city]
#
#     # Debug print
#     print(f"City: {next_city}, Distance Score: {distance_matrix[current_index][next_city]}, Angular Deviation: {angular_deviation}, Base Score: {base_score}, Final Score: {best_score}")
#
#     return next_city, last_direction

def find_next_city(current_index, path, visited, cities, angle_matrix, distance_matrix, last_direction, edge_nodes, centroid):
    best_score = float('inf')
    next_city = None
    current_centroid_angle = np.degrees(np.arctan2(cities[current_index][1] - centroid[1], cities[current_index][0] - centroid[0]))

    for i in range(len(cities)):
        if i not in visited:
            current_angle = angle_matrix[current_index][i]
            angular_deviation = min(abs(current_angle - last_direction), 360 - abs(current_angle - last_direction))
            distance_score = distance_matrix[current_index][i]
            is_edge = i in edge_nodes
            candidate_centroid_angle = np.degrees(np.arctan2(cities[i][1] - centroid[1], cities[i][0] - centroid[0]))
            angle_from_centroid_deviation = abs(current_centroid_angle - candidate_centroid_angle)

            base_score = 0.2 * angular_deviation + 0.8 * distance_score
            # If base_score is within a certain range, add a slight preference for edge nodes
            if base_score < 100:  # Adjust this threshold as needed
                score = base_score * (0.9 if is_edge and angle_from_centroid_deviation < 10 else 1)
            else:
                score = base_score

            # Debug print
            print(f"City: {i}, Distance Score: {distance_score}, Angular Deviation: {angular_deviation}, Base Score: {base_score}, Final Score: {score}")

            if score < best_score:
                best_score = score
                next_city = i
                last_direction = current_angle  # Update the direction to the best found

    # Debug print
    print(f"Current index: {current_index}, Best score: {best_score}, Next city: {next_city}, Last direction: {last_direction}")

    return next_city, last_direction

def generate_path(starting_city, cities, angle_matrix, distance_matrix, edge_nodes, last_direction, centroid):
    """Generate a path starting from the given city, avoiding crossovers, considering edge nodes."""
    path = [starting_city]  # Start the path with the starting city
    visited = set(path)  # Track visited cities to avoid revisiting them

    while len(visited) < len(cities):
        current_city = path[-1]
        next_city, last_direction = find_next_city(current_city, path, visited, cities, angle_matrix, distance_matrix, last_direction, edge_nodes, centroid)

        # Debug print
        print(f"Current city: {current_city}, Next city: {next_city}, Path: {path}")

        if next_city is not None:
            path.append(next_city)
            visited.add(next_city)
        else:
            print("No valid next city found.")
            break  # Break if no valid next city is found

    return path

# def generate_path(starting_city, cities, angle_matrix, distance_matrix, edge_nodes, last_direction, centroid):
#     """Generate a path starting from the given city, considering edge nodes."""
#     path = [starting_city]
#     visited = set(path)
#
#     while len(visited) < len(cities):
#         current_city = path[-1]
#         if len(path) == 1:  # First leap is distance-dependent
#             # Find the closest city to the starting city
#             closest_city = np.argmin(distance_matrix[current_city][np.setdiff1d(range(len(cities)), list(visited))])
#             next_city = closest_city if closest_city != current_city else (closest_city + 1) % len(cities)
#             last_direction = angle_matrix[current_city][next_city]
#         else:
#             next_city, last_direction = find_next_city(current_city, cities, angle_matrix, distance_matrix, last_direction, edge_nodes, centroid, visited)
#
#         if next_city is not None and next_city not in visited:
#             path.append(next_city)
#             visited.add(next_city)
#         else:
#             print("No valid next city found. Ending path generation.")
#             break
#
#     return path

def main():
    tsp_filename = 'berlin52.tsp'
    opt_tour_filename = 'berlin52.opt.tour'

    cities = load_tsp_data_from_file(tsp_filename)
    optimal_tour = load_optimal_tour(opt_tour_filename)

    print(f"Number of cities loaded: {len(cities)}")
    print(f"Optimal tour length: {len(optimal_tour)}")

    distance_matrix, angle_matrix = compute_distance_and_angle_matrices(cities)
    centroid = calculate_centroid(cities)
    edge_nodes = find_edge_nodes(cities, centroid)

    print("Edge Nodes:", edge_nodes)  # Print the indices of the edge nodes

    starting_city = 0  # Fixed starting city
    last_direction = 0  # Initialize direction based on first segment or default

    generated_path = generate_path(starting_city, cities, angle_matrix, distance_matrix, edge_nodes, last_direction, centroid)

    print("Generated Path:", generated_path)
    print("Generated Path Length:", sum(distance_matrix[generated_path[i]][generated_path[i + 1]] for i in range(len(generated_path) - 1)))
    print("Optimal Tour:", optimal_tour)
    print("Optimal Tour Length:", sum(distance_matrix[optimal_tour[i]][optimal_tour[i + 1]] for i in range(len(optimal_tour) - 1)))

    # Plot the generated path
    print("Plotting Generated Path")
    plot_tour(cities, generated_path, centroid, edge_nodes, title='Generated TSP Tour')
    # Plot the optimal path
    print("Plotting Optimal Path")
    plot_tour(cities, optimal_tour, centroid, edge_nodes, title='Optimal TSP Tour')

if __name__ == '__main__':
    main()


# def main():
#     tsp_filename = 'berlin52.tsp'
#     opt_tour_filename = 'berlin52.opt.tour'
#
#     cities = load_tsp_data_from_file(tsp_filename)
#     optimal_tour = load_optimal_tour(opt_tour_filename)
#     distance_matrix, angle_matrix = compute_distance_and_angle_matrices(cities)
#     centroid = calculate_centroid(cities)
#     edge_nodes = find_edge_nodes(cities, centroid)
#
#     starting_city = 0
#     last_direction = 0
#
#     generated_path = generate_path(starting_city, cities, angle_matrix, distance_matrix, edge_nodes, last_direction,
#                                     centroid)
#
#     print("Generated Path:", generated_path)
#     print("Generated Path Length:",
#             sum(distance_matrix[generated_path[i]][generated_path[i + 1]] for i in range(len(generated_path) - 1)))
#     print("Optimal Tour:", optimal_tour)
#     print("Optimal Tour Length:",
#             sum(distance_matrix[optimal_tour[i]][optimal_tour[i + 1]] for i in range(len(optimal_tour) - 1)))
#
#     # Plot the generated path
#     plot_tour(cities, generated_path, centroid, edge_nodes, title='Generated TSP Tour')
#     # Plot the optimal path
#     plot_tour(cities, optimal_tour, centroid, edge_nodes, title='Optimal TSP Tour')
#
#
# if __name__ == '__main__':
#     main()