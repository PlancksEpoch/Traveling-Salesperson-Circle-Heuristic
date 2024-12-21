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

    # Calculate distances from the centroid
    centroid = np.mean(cities, axis=0)
    distances_from_centroid = [(index, np.hypot(city[0] - centroid[0], city[1] - centroid[1])) for index, city in enumerate(cities)]

    # Sort by distances
    distances_from_centroid.sort(key=lambda x: x[1], reverse=True)

    # Select edge nodes based on distance
    edge_nodes = [index for index, _ in distances_from_centroid[:num_edges]]

    return edge_nodes

def find_next_city(current_index, path, visited, cities, angle_matrix, distance_matrix, last_direction, edge_nodes, is_first_move):
    candidates = []

    for i in range(len(cities)):
        if i not in visited:
            current_angle = angle_matrix[current_index][i]
            distance_score = distance_matrix[current_index][i]

            if is_first_move:
                # For the first move, only consider distance
                base_score = distance_score
                angular_deviation = 0  # No angle deviation on the first move
            else:
                angular_deviation = min(abs(current_angle - last_direction), 360 - abs(current_angle - last_direction))
                base_score = 0.2 * angular_deviation + 0.8 * distance_score

            # Add the candidate to the list
            candidates.append((base_score, i, current_angle, distance_score, angular_deviation))

    # Sort candidates by base score (ascending)
    candidates.sort(key=lambda x: x[0])

    # Debug print all candidates
    for candidate in candidates:
        print(f"City: {candidate[1]}, Base Score: {candidate[0]}, Distance Score: {candidate[3]}, Angular Deviation: {candidate[4]}")

    # Check the top three candidates and apply edge priority if scores are close
    top_candidates = candidates[:3]
    best_candidate = top_candidates[0]

    score_threshold = 10  # Define a threshold for score discrepancy to consider edge weight
    score_discrepancy = top_candidates[1][0] - top_candidates[0][0] if len(top_candidates) > 1 else float('inf')

    if score_discrepancy < score_threshold:
        for candidate in top_candidates:
            if candidate[1] in edge_nodes:
                best_candidate = candidate
                break

    next_city = best_candidate[1]
    last_direction = best_candidate[2]

    # Debug print selected candidate
    print(f"Selected next city: {next_city} with base score: {best_candidate[0]}")

    return next_city, last_direction

def generate_path(starting_city, cities, angle_matrix, distance_matrix, edge_nodes):
    """Generate a path starting from the given city, avoiding crossovers, considering edge nodes."""
    path = [starting_city - 1]  # Start the path with the starting city
    visited = set(path)  # Track visited cities to avoid revisiting them

    is_first_move = True  # Flag to indicate if it's the first move
    last_direction = 0  # Initialize direction based on first segment or default

    while len(visited) < len(cities):
        current_city = path[-1]
        next_city, last_direction = find_next_city(current_city, path, visited, cities, angle_matrix, distance_matrix, last_direction, edge_nodes, is_first_move)
        is_first_move = False  # After the first move, reset the flag

        # Debug print
        print(f"Current city: {current_city}, Next city: {next_city}, Path: {path}")

        if next_city is not None:
            path.append(next_city)
            visited.add(next_city)
        else:
            print("No valid next city found.")
            break  # Break if no valid next city is found

    return path

def main():
    tsp_filename = 'berlin52.tsp'
    opt_tour_filename = 'berlin52.opt.tour'

    cities = load_tsp_data_from_file(tsp_filename)
    optimal_tour = load_optimal_tour(opt_tour_filename)

    print(f"Number of cities loaded: {len(cities)}")
    print(f"Optimal tour length: {len(optimal_tour)}")

    distance_matrix, angle_matrix = compute_distance_and_angle_matrices(cities)
    edge_nodes = find_edge_nodes(cities)

    print("Edge Nodes:", edge_nodes)  # Print the indices of the edge nodes

    starting_city = 1  # Fixed starting city

    generated_path = generate_path(starting_city, cities, angle_matrix, distance_matrix, edge_nodes)

    print("Generated Path:", generated_path)
    print("Generated Path Length:",
          sum(distance_matrix[generated_path[i]][generated_path[i + 1]] for i in range(len(generated_path) - 1)))
    print("Optimal Tour:", optimal_tour)
    print("Optimal Tour Length:",
          sum(distance_matrix[optimal_tour[i]][optimal_tour[i + 1]] for i in range(len(optimal_tour) - 1)))

    # Plot the generated path
    print("Plotting Generated Path")
    plot_tour(cities, generated_path, title='Generated TSP Tour')
    # Plot the optimal path
    print("Plotting Optimal Path")
    plot_tour(cities, optimal_tour, title='Optimal TSP Tour')

if __name__ == '__main__':
    main()