/**
 * Parallel VLSI Wire Routing via OpenMP
 * Devika Manohar (dmanohar), Barbara Zhuo (jiayizhu)
 */

#include "wireroute.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <omp.h>
#include <unistd.h>

void print_stats(const std::vector<std::vector<int>> &occupancy) {
  int max_occupancy = 0;
  long long total_cost = 0;

  for (const auto &row : occupancy) {
    for (const int count : row) {
      max_occupancy = std::max(max_occupancy, count);
      total_cost += count * count;
    }
  }

  std::cout << "Max occupancy: " << max_occupancy << '\n';
  std::cout << "Total cost: " << total_cost << '\n';
}

/* This function write the output into 2 files
(1) It write occupancy grids into a file
(2) It convert wires from Wire to validate_wire_t by to_validate_format
(2) It write wires into another file
*/
void write_output(
    const std::vector<Wire> &wires, const int num_wires,
    const std::vector<std::vector<int>> &occupancy, const int dim_x,
    const int dim_y,
    std::string wires_output_file_path = "outputs/wire_output.txt",
    std::string occupancy_output_file_path = "outputs/occ_output.txt") {

  std::ofstream out_occupancy(occupancy_output_file_path, std::fstream::out);
  if (!out_occupancy) {
    std::cerr << "Unable to open file: " << occupancy_output_file_path << '\n';
    exit(EXIT_FAILURE);
  }
  out_occupancy << dim_x << ' ' << dim_y << '\n';

  for (const auto &row : occupancy) {
    for (size_t i = 0; i < row.size(); ++i)
      out_occupancy << row[i] << (i == row.size() - 1 ? "" : " ");
    out_occupancy << '\n';
  }
  out_occupancy.close();

  std::ofstream out_wires(wires_output_file_path, std::fstream::out);
  if (!out_wires) {
    std::cerr << "Unable to open file: " << wires_output_file_path << '\n';
    exit(EXIT_FAILURE);
  }

  out_wires << dim_x << ' ' << dim_y << '\n';
  out_wires << num_wires << '\n';

  for (const auto &wire : wires) {
    // NOTICE: we convert to keypoint representation here, using
    // to_validate_format which need to be defined in the bottom of this file
    validate_wire_t keypoints = wire.to_validate_format();
    for (int i = 0; i < keypoints.num_pts; ++i) {
      out_wires << keypoints.p[i].x << ' ' << keypoints.p[i].y;
      if (i < keypoints.num_pts - 1)
        out_wires << ' ';
    }
    out_wires << '\n';
  }

  out_wires.close();
}

/** from a chosen wire, get the point coordinates of starting point (at index 0),
or a bend points (index 1, 2, 3) or end point (index 4)
*/
Point get_point_i(const Wire &wire, int i) {
  if (i ==0){
    return {wire.start_x, wire.start_y};
  } else if (i == wire.num_bends + 1) {
    return {wire.end_x, wire.end_y};
  } else {
    return wire.bends[i - 1];
  }
}

/** this updates the occupancy matrix by adding or subtracting 1 in the entire
 wire's path on the matrix. to add a wire, use delta = 1 and to subtract 
 a wire, use delta = -1. 
*/
void update_occupancy_with_wire(std::vector<std::vector<int>> &occupancy,
                               const Wire &wire, int delta) {
  Point curr_pt = get_point_i(wire, 0);
  occupancy[curr_pt.y][curr_pt.x] += delta;

  // loop through all line segments and add the delta (+1 or -1) on all points you traverse
  const int num_segments = wire.num_bends + 1;
  for (int s = 0; s < num_segments; s++) {
    // get the end point where this current segment will end
    Point next_pt = get_point_i(wire, s + 1);
    // figure out if we are adding from the x coordinate or subutracting
    int step_x = 0;
    if (next_pt.x > curr_pt.x) {
      step_x = 1;
    } else if (next_pt.x < curr_pt.x) {
      step_x = -1;
    }
    // figure out if we are adding from the y coordinate or subutracting
    int step_y = 0;
    if (next_pt.y > curr_pt.y) {
      step_y = 1;
    } else if (next_pt.y < curr_pt.y) {
      step_y = -1;
    }
    
    // move through from curr_pt to next_pt
    while (curr_pt.x != next_pt.x || curr_pt.y != next_pt.y) {
      curr_pt.x += step_x;
      curr_pt.y += step_y;
      occupancy[curr_pt.y][curr_pt.x] += delta;
    }
  }
}

int calculate_occupancy_cost(const std::vector<std::vector<int>> &occupancy) {
  int total = 0;
  for (const auto &row : occupancy) {
    for (std::size_t col = 0; col < row.size(); col ++) {
      int count = row[col];
      total += count * count;
    }
  }
  return total;
}

int main(int argc, char *argv[]) {
  const auto init_start = std::chrono::steady_clock::now();

  std::string input_filename;
  int num_threads = 0;
  double SA_prob = 0.1;
  int SA_iters = 5;
  char parallel_mode = '\0';
  int batch_size = 1;

  int opt;
  while ((opt = getopt(argc, argv, "f:n:p:i:m:b:")) != -1) {
    switch (opt) {
    case 'f':
      input_filename = optarg;
      break;
    case 'n':
      num_threads = atoi(optarg);
      break;
    case 'p':
      SA_prob = atof(optarg);
      break;
    case 'i':
      SA_iters = atoi(optarg);
      break;
    case 'm':
      parallel_mode = *optarg;
      break;
    case 'b':
      batch_size = atoi(optarg);
      break;
    default:
      std::cerr << "Usage: " << argv[0]
                << " -f input_filename -n num_threads [-p SA_prob] [-i "
                   "SA_iters] -m parallel_mode -b batch_size\n";
      exit(EXIT_FAILURE);
    }
  }

  // Check if required options are provided
  if (empty(input_filename) || num_threads <= 0 || SA_iters <= 0 ||
      (parallel_mode != 'A' && parallel_mode != 'W') || batch_size <= 0) {
    std::cerr << "Usage: " << argv[0]
              << " -f input_filename -n num_threads [-p SA_prob] [-i SA_iters] "
                 "-m parallel_mode -b batch_size\n";
    exit(EXIT_FAILURE);
  }

  std::cout << "Number of threads: " << num_threads << '\n';
  std::cout << "Simulated annealing probability parameter: " << SA_prob << '\n';
  std::cout << "Simulated annealing iterations: " << SA_iters << '\n';
  std::cout << "Input file: " << input_filename << '\n';
  std::cout << "Parallel mode: " << parallel_mode << '\n';
  std::cout << "Batch size: " << batch_size << '\n';

  std::ifstream fin(input_filename);

  if (!fin) {
    std::cerr << "Unable to open file: " << input_filename << ".\n";
    exit(EXIT_FAILURE);
  }

  int dim_x, dim_y;
  int num_wires;

  /* Read the grid dimension and wire information from file */
  fin >> dim_x >> dim_y >> num_wires;

  std::vector<Wire> wires(num_wires);
  // occupancy matrix
  std::vector occupancy(dim_y, std::vector<int>(dim_x));
  std::cout << "Question Spec: dim_x=" << dim_x << ", dim_y=" << dim_y
            << ", number of wires=" << num_wires << '\n';

  // TODO (student code start): Read the wire information from file, 
  // you may need to change this if you define the wire structure differently.

  // uniform distribution for set of all possible routes
  std::bernoulli_distribution choose_random_route(SA_prob);

  for (auto &wire : wires) {
    fin >> wire.start_x >> wire.start_y >> wire.end_x >> wire.end_y;
    wire.move_x_start = true;
    wire.move_x_end = false;
    wire.mid_x = wire.end_x;
    wire.mid_y = wire.start_y;
  }

  /* Initialize any additional data structures needed in the algorithm */

  // Student code end

  const double init_time =
      std::chrono::duration_cast<std::chrono::duration<double>>(
          std::chrono::steady_clock::now() - init_start)
          .count();
  std::cout << "Initialization time (sec): " << std::fixed
            << std::setprecision(10) << init_time << '\n';

  const auto compute_start = std::chrono::steady_clock::now();

  /* TODO (student code start): Implement the wire routing algorithm here and
    feel free to structure the algorithm into different functions.
    Don't use global variables.
    Use OpenMP to parallelize the algorithm.
  */
    
  // place initial route using greedy heuristic
  for (const auto &wire : wires){ // OPENMP PARALLELIZE: each wire is independent for this initial laying
    // straight line conditions
    if (wire.start_x == wire.end_x) {
      wire.num_bends = 0;
      // draw straight vertical line
      update_occupancy_with_wire(occupancy,wire,1);
    } else if (wire.start_y == wire.end_y) {
      // draw straight horizontal line
      wire.num_bends = 0;
      update_occupancy_with_wire(occupancy,wire,1);
    } else {
      // naive so draw 1 bend line (just to lay smth down)
      wire.bend1_x = wire.start_x
      wire.bend1_y = wire.end_y
      wire.num_bends = 1
      update_occupancy_with_wire(occupancy, wire, 1);
    }
  }
  int curr_cost;
  int best_cost;
  Wire best_wire;
  // now iterate 
  for (int iters = 0; iters < SA_iters; iters++){
    for (auto &wire : wires){
      // straight line: no other paths for these wires
      if (wire.num_bends == 0){
        continue;
      }

      /*Iterate over all legal routes between the endpoints (including 1, 2, and 3 bends,
      and horizontal-first and vertical-first routes), calculating their cost by reading the corresponding
      elements of the occupancy matrix, squaring their values, and incrementally adding them to the cost
      of the route. */
      // measure current placement before removing wire
      curr_cost = calculate_occupancy_cost(occupancy);
      best_cost = curr_cost;
      best_wire = wire;
      // remove the old route
      update_occupancy_with_wire(occupancy, wire, -1);

      int dx = std::abs(wire.end_x - wire.start_x);
      int dy = std::abs(wire.end_y - wire.start_y);

      int x_direction = 1;
      if (wire.end_x < wire.start_x) {
        x_direction = -1;
      }

      int y_direction = 1;
      if (wire.end_y < wire.start_y) {
        y_direction = -1;
      }
      int num_routes = dx + dy + 2 * (dx - 1) * (dy - 1);
      for (int route_idx = 0; route_idx < num_routes; route_idx++) {
      }
    }
  }

  // initialize wires
  // Within wires
  if (parallel_mode == 'W') {
    // within wires
  } else {
    // across wires
  }

  // Student code end
  // DON'T CHANGE THE FOLLOWING CODE
  const double compute_time =
      std::chrono::duration_cast<std::chrono::duration<double>>(
          std::chrono::steady_clock::now() - compute_start)
          .count();
  std::cout << "Computation time (sec): " << compute_time << '\n';

  /* wire to run check on wires and occupancy */
  wr_checker checker(wires, occupancy);
  checker.validate();

  /* Write wires and occupancy matrix to files */
  print_stats(occupancy);
  write_output(wires, num_wires, occupancy, dim_x, dim_y);
}

/* TODO (student): implement to_validate_format to convert Wire to
  validate_wire_t keypoint representation in order to run checker and
  write output
*/
validate_wire_t Wire::to_validate_format(void) const {
  validate_wire_t w;
  
  return w;
}
