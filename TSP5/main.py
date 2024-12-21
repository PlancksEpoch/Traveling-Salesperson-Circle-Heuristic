import numpy as np
import matplotlib.pyplot as plt

def load_tsp_data_from_file(filename):
    """Load TSP data from a plain text file containing coordinates."""
    cities = []
    try:
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
                    if len(parts) >= 3:
                        _, x, y = parts[:3]
                        cities.append((float(x), float(y)))
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return []
    except ValueError as e:
        print(f"Error processing line '{line.strip()}': {e}")
        return []
    return cities

def load_optimal_tour(filename):
    """Load the optimal tour from a plain text file."""
    tour = []
    try:
        with open(filename, 'r') as file:
            start_reading = False
            for line in file:
                if 'TOUR_SECTION' in line:
                    start_reading = True
                    continue
                if start_reading:
                    line = line.strip()
                    if line == '-1' or line == 'EOF':
                        break
                    if line.isdigit():
                        city = int(line)
                        tour.append(city - 1)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return []
    except ValueError as e:
        print(f"Error processing line '{line.strip()}': {e}")
        return []
    if not tour:
        print("Warning: Optimal tour is empty.")
    return tour


def compute_distance_and_angle_matrices(cities):
    """Compute both distance and angle matrices using vectorized operations."""
    coordinates = np.array(cities)
    x = coordinates[:, 0]
    y = coordinates[:, 1]

    # Calculate differences
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]

    # Compute distance matrix
    distance_matrix = np.hypot(dx, dy)

    # Compute angle matrix
    angle_matrix = (np.degrees(np.arctan2(dy, dx)) + 360) % 360

    # Set diagonals to zero (distance and angle from a city to itself)
    np.fill_diagonal(distance_matrix, 0)
    np.fill_diagonal(angle_matrix, 0)

    return distance_matrix, angle_matrix

def plot_tour(cities, tour, title='Traveling Salesman Path'):
    if not tour:
        print(f"Warning: No tour data to plot for {title}. Check the tour data.")
        return

    try:
        # Ensure all tour indices are within the range of cities
        if max(tour) >= len(cities) or min(tour) < 0:
            raise ValueError("Tour indices exceed the number of cities or are negative.")

        # Unpack city coordinates for the tour
        x_coords, y_coords = zip(*[cities[i] for i in tour])

        # Close the loop by adding the first city at the end if it's not already there
        if tour[0] != tour[-1]:
            x_coords += (x_coords[0],)
            y_coords += (y_coords[0],)

        plt.figure(figsize=(12, 8))
        plt.plot(x_coords, y_coords, 'o-', label='Tour Path')  # Plot the tour path
        plt.scatter(x_coords[0], y_coords[0], color='red', s=200, label='Starting City')  # Starting city

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

def find_edge_nodes(cities, percentage=0.2):
    num_edges = max(int(len(cities) * percentage), 1)

    # Calculate centroid of all cities
    coordinates = np.array(cities)
    centroid = np.mean(coordinates, axis=0)

    # Calculate distances from the centroid
    distances_from_centroid = [(index, np.hypot(city[0] - centroid[0], city[1] - centroid[1]))
                               for index, city in enumerate(cities)]

    # Sort by distances
    distances_from_centroid.sort(key=lambda x: x[1], reverse=True)

    # Select edge nodes based on distance
    edge_nodes = [index for index, _ in distances_from_centroid[:num_edges]]

    return edge_nodes

def find_next_city(current_index, path, visited, cities, angle_matrix, distance_matrix, last_direction, edge_nodes, is_first_move,
                   distance_weight=0.8, angle_weight=0.2, score_threshold=10):
    candidates = []

    for i in range(len(cities)):
        if i not in visited:
            current_angle = angle_matrix[current_index][i]
            distance_score = distance_matrix[current_index][i]

            if is_first_move:
                base_score = distance_score
                angular_deviation = 0
            else:
                angular_deviation = min(abs(current_angle - last_direction), 360 - abs(current_angle - last_direction))
                base_score = angle_weight * angular_deviation + distance_weight * distance_score

            candidates.append((base_score, i, current_angle, distance_score, angular_deviation))

    candidates.sort(key=lambda x: x[0])

    # Debug print all candidates
    # (Optional: comment out or remove if not needed)
    # for candidate in candidates:
    #     print(f"City: {candidate[1]}, Base Score: {candidate[0]}, Distance Score: {candidate[3]}, Angular Deviation: {candidate[4]}")

    top_candidates = candidates[:3]
    best_candidate = top_candidates[0]

    if len(top_candidates) > 1:
        score_discrepancy = top_candidates[1][0] - top_candidates[0][0]
    else:
        score_discrepancy = float('inf')

    if score_discrepancy < score_threshold:
        for candidate in top_candidates:
            if candidate[1] in edge_nodes:
                best_candidate = candidate
                break

    next_city = best_candidate[1]
    last_direction = best_candidate[2]

    return next_city, last_direction

def two_opt(route, distance_matrix):
    """Improve the route using the 2-opt algorithm."""
    best = route
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best)):
                if j - i == 1:
                    continue  # Skip adjacent edges
                new_route = best[:i] + best[i:j][::-1] + best[j:]
                if calculate_route_length(new_route, distance_matrix) < calculate_route_length(best, distance_matrix):
                    best = new_route
                    improved = True
        if improved:
            route = best
    return best

def calculate_route_length(route, distance_matrix):
    """Calculate the total length of the route."""
    total_distance = sum(distance_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))
    # Add distance from last to first to complete the loop
    total_distance += distance_matrix[route[-1], route[0]]
    return total_distance

def generate_path(starting_city, cities, angle_matrix, distance_matrix, edge_nodes,
                  distance_weight=0.8, angle_weight=0.2, score_threshold=10):
    """Generate a path starting from the given city, avoiding crossovers, considering edge nodes."""
    path = [starting_city]
    visited = set(path)
    is_first_move = True
    last_direction = 0

    while len(visited) < len(cities):
        current_city = path[-1]
        next_city, last_direction = find_next_city(
            current_city, path, visited, cities, angle_matrix, distance_matrix,
            last_direction, edge_nodes, is_first_move,
            distance_weight=distance_weight, angle_weight=angle_weight, score_threshold=score_threshold
        )
        is_first_move = False

        if next_city is not None:
            path.append(next_city)
            visited.add(next_city)
        else:
            print("No valid next city found.")
            break

    return path

def main():
    tsp_filename = 'berlin52.tsp'
    opt_tour_filename = 'berlin52.opt.tour'

    cities = load_tsp_data_from_file(tsp_filename)
    if not cities:
        print("No cities loaded. Exiting.")
        return

    optimal_tour = load_optimal_tour(opt_tour_filename)
    if not optimal_tour:
        print("No optimal tour loaded.")

    print(f"Number of cities loaded: {len(cities)}")
    if optimal_tour:
        print(f"Optimal tour length: {len(optimal_tour)}")

    distance_matrix, angle_matrix = compute_distance_and_angle_matrices(cities)
    edge_nodes = find_edge_nodes(cities)

    print("Edge Nodes:", edge_nodes)

    starting_city = 0  # You can change this to test different starting points

    # Set heuristic parameters
    distance_weight = 0.75
    angle_weight = 0.25
    score_threshold = 10

    generated_path = generate_path(
        starting_city, cities, angle_matrix, distance_matrix, edge_nodes,
        distance_weight=distance_weight, angle_weight=angle_weight, score_threshold=score_threshold
    )

    if not generated_path:
        print("No path generated. Exiting.")
        return

    print("Generated Path:", generated_path)
    generated_path_length = calculate_route_length(generated_path, distance_matrix)
    print("Generated Path Length:", generated_path_length)

    if optimal_tour:
        optimal_tour_length = calculate_route_length(optimal_tour, distance_matrix)
        print("Optimal Tour Length:", optimal_tour_length)

    # Plot the generated path
    print("Plotting Generated Path")
    plot_tour(cities, generated_path, title='Generated TSP Tour')

    # Optimize the path using 2-opt
    print("Optimizing path using 2-opt...")
    optimized_path = two_opt(generated_path, distance_matrix)
    optimized_path_length = calculate_route_length(optimized_path, distance_matrix)
    print("Optimized Path Length:", optimized_path_length)

    # Plot the optimized path
    print("Plotting Optimized Path")
    plot_tour(cities, optimized_path, title='Optimized TSP Tour')

    if optimal_tour:
        # Plot the optimal path
        print("Plotting Optimal Path")
        plot_tour(cities, optimal_tour, title='Optimal TSP Tour')

if __name__ == '__main__':
    main()