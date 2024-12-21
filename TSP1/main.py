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
        # Unpack city coordinates for the tour
        x_coords, y_coords = zip(*[cities[i] for i in tour])
        x_coords += (x_coords[0],)  # Close the loop
        y_coords += (y_coords[0],)

        plt.figure(figsize=(12, 8))
        plt.plot(x_coords, y_coords, 'o-', label='Tour Path')  # Plot the tour path
        plt.scatter(x_coords[0], y_coords[0], color='red', label='Starting City')  # Starting city

        # Highlight the centroid
        plt.plot(centroid[0], centroid[1], 'x', color='gold', markersize=10, label='Centroid')

        # Highlight edge nodes
        edge_x, edge_y = zip(*[cities[i] for i in edge_nodes])
        plt.scatter(edge_x, edge_y, color='green', s=100, edgecolor='black', label='Edge Nodes')

        # Label each edge node
        for i, (x, y) in enumerate(zip(edge_x, edge_y)):
            plt.text(x, y, f'Edge {edge_nodes[i] + 1}', color='darkgreen', fontsize=12)

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

def do_lines_intersect(p1, p2, q1, q2):
    """
    Return True if line segment p1p2 and q1q2 intersect, ignoring intersections at shared endpoints.

    Args:
    p1 (tuple): The first point of the first line segment.
    p2 (tuple): The second point of the first line segment.
    q1 (tuple): The first point of the second line segment.
    q2 (tuple): The second point of the second line segment.

    Returns:
    bool: True if the line segments intersect, False otherwise.
    """
    def orientation(p, q, r):
        """
        Return orientation type (0 = collinear, 1 = clockwise, 2 = counterclockwise).

        Args:
        p (tuple): The first point.
        q (tuple): The second point.
        r (tuple): The third point.

        Returns:
        int: The orientation type.
        """
        val = (float(q[1] - p[1]) * (r[0] - q[0])) - (float(q[0] - p[0]) * (r[1] - q[1]))
        if val > 0:
            return 1
        elif val < 0:
            return 2
        else:
            return 0

    def on_segment(p, q, r):
        """
        Check if point q lies on line segment pr.

        Args:
        p (tuple): The first point of the line segment.
        q (tuple): The point to check.
        r (tuple): The second point of the line segment.

        Returns:
        bool: True if q lies on line segment pr, False otherwise.
        """
        if min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1]):
            return True
        return False

    # Exclude cases where endpoints are shared
    if p1 == q1 or p1 == q2 or p2 == q1 or p2 == q2:
        return False

    # General case
    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    if o1 != o2 and o3 != o4:
        print(f"Intersection detected: Line {p1}-{p2} intersects with line {q1}-{q2}")
        return True

    # Special cases: Check if the points are collinear and on segment
    if o1 == 0 and on_segment(p1, q1, p2):
        return False
    if o2 == 0 and on_segment(p1, q2, p2):
        return False
    if o3 == 0 and on_segment(q1, p1, q2):
        return False
    if o4 == 0 and on_segment(q1, p2, q2):
        return False

    return False


def find_edge_nodes(cities, centroid, percentage=0.3):
    num_edges = max(int(len(cities) * percentage), 1)
    sorted_cities = sorted(cities, key=lambda city: -np.hypot(city[0] - centroid[0], city[1] - centroid[1]))
    edge_nodes = sorted_cities[:num_edges]
    return [cities.index(node) for node in edge_nodes]

def calculate_centroid(cities):
    x_coords = [city[0] for city in cities]
    y_coords = [city[1] for city in cities]
    centroid = (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))
    return centroid

def is_edge_node(node_index, edge_nodes):
    """Check if a node is an edge node."""
    return node_index in edge_nodes

def find_next_city(current_index, path, visited, cities, angle_matrix, distance_matrix, last_direction, edge_nodes, centroid, decision_points):
    best_score = float('inf')
    next_city = None
    current_centroid_angle = np.degrees(np.arctan2(cities[current_index][1] - centroid[1], cities[current_index][0] - centroid[0]))

    for i in range(len(cities)):
        if i not in visited and is_path_valid(path, current_index, i, cities):
            current_angle = angle_matrix[current_index][i]
            angular_deviation = min(abs(current_angle - last_direction), 360 - abs(current_angle - last_direction))
            distance_score = distance_matrix[current_index][i]
            is_edge = i in edge_nodes
            candidate_centroid_angle = np.degrees(np.arctan2(cities[i][1] - centroid[1], cities[i][0] - centroid[0]))
            angle_from_centroid_deviation = abs(current_centroid_angle - candidate_centroid_angle)

            score = (0.2 * angular_deviation + 0.8 * distance_score) * (0.9 if is_edge and angle_from_centroid_deviation < 10 else 1)

            if score < best_score:
                best_score = score
                next_city = i
                last_direction = current_angle  # Update the direction to the best found

            # Always store potential alternatives
            if decision_points and score > best_score and score - best_score < 0.1:
                decision_points[-1]['alternatives'].append({'index': current_index, 'score': score, 'city': i})
            elif not decision_points or not decision_points[-1]['alternatives']:
                decision_points.append({'index': current_index, 'score': best_score, 'city': next_city, 'alternatives': []})

    # Debug print
    print(f"Current index: {current_index}, Best score: {best_score}, Next city: {next_city}, Last direction: {last_direction}")

    return next_city, last_direction, decision_points

def is_path_valid(path, start_index, end_index, cities):
    """Check if adding an edge from start_index to end_index causes any intersection with existing path segments."""
    new_start_coords = cities[start_index]
    new_end_coords = cities[end_index]
    n = len(path)
    if n < 2:
        return True  # No possible intersections if less than two edges

    for i in range(n - 1):
        if do_lines_intersect(cities[path[i]], cities[path[i + 1]], new_start_coords, new_end_coords):
            print(f"Path invalid: Adding edge {new_start_coords}-{new_end_coords} intersects with existing path segment {cities[path[i]]}-{cities[path[i + 1]]}")
            return False
    return True

def generate_path(starting_city, cities, angle_matrix, distance_matrix, edge_nodes, last_direction, centroid):
    """Generate a path starting from the given city, avoiding crossovers, considering edge nodes, and managing decision points for possible backtracking."""
    path = [starting_city]  # Start the path with the starting city
    visited = set(path)  # Track visited cities to avoid revisiting them
    decision_points = []  # Initialize decision points list

    while len(visited) < len(cities):
        current_city = path[-1]
        next_city, last_direction, decision_points = find_next_city(current_city, path, visited, cities, angle_matrix,
                                                                    distance_matrix, last_direction, edge_nodes,
                                                                    centroid, decision_points)

        # Debug print
        print(f"Current city: {current_city}, Next city: {next_city}, Path: {path}, Decision points: {decision_points}")

        if next_city is not None:
            path.append(next_city)
            visited.add(next_city)
        else:
            if decision_points:  # If dead end and decisions were close, backtrack
                last_decision = decision_points.pop()
                print(f"Backtracking to decision point: {last_decision}")
                if last_decision['city'] is not None and last_decision['city'] in path:
                    path = path[:path.index(last_decision['city']) + 1]  # Rollback to last contentious decision
                    visited = set(path)  # Reset visited based on new path
                elif last_decision['alternatives']:
                    best_alternative = min(last_decision['alternatives'], key=lambda alt: alt['score'])
                    next_city = best_alternative['city']
                    path = path[:path.index(last_decision['index']) + 1]  # Rollback to alternative decision point
                    visited = set(path)  # Reset visited based on new path
                    path.append(next_city)
                    visited.add(next_city)
                else:
                    print(f"Error: Last decision city {last_decision['city']} not in current path {path}")
                    break
            else:
                print("No valid next city found and no decision points left to backtrack.")
                break  # Break if no valid next city is found and no decision points left

    return path

def main():
    tsp_filename = 'berlin52.tsp'
    opt_tour_filename = 'berlin52.opt.tour'

    cities = load_tsp_data_from_file(tsp_filename)
    optimal_tour = load_optimal_tour(opt_tour_filename)
    distance_matrix, angle_matrix = compute_distance_and_angle_matrices(cities)
    centroid = calculate_centroid(cities)
    edge_nodes = find_edge_nodes(cities, centroid)  # Assume this function is defined to find edge nodes

    print("Edge Nodes:", edge_nodes)  # Print the indices of the edge nodes

    starting_city = 0  # Fixed starting city
    last_direction = 0  # Initialize direction based on first segment or default

    generated_path = generate_path(starting_city, cities, angle_matrix, distance_matrix, edge_nodes, last_direction, centroid)

    print("Generated Path:", generated_path)
    print("Generated Path Length:", sum(distance_matrix[generated_path[i]][generated_path[i + 1]] for i in range(len(generated_path) - 1)))
    print("Optimal Tour:", optimal_tour)
    print("Optimal Tour Length:", sum(distance_matrix[optimal_tour[i]][optimal_tour[i + 1]] for i in range(len(optimal_tour) - 1)))

    # Plot the generated path
    plot_tour(cities, generated_path, centroid, edge_nodes, title='Generated TSP Tour')
    # Plot the optimal path
    plot_tour(cities, optimal_tour, centroid, edge_nodes, title='Optimal TSP Tour')

if __name__ == '__main__':
    main()