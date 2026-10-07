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
#include <limits>
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
  if (i == 0){
    return {wire.start_x, wire.start_y};
  } else if (i == wire.num_bends + 1) {
    return {wire.end_x, wire.end_y};
  } else {
    return wire.bends[i-1];
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


/** calculates the cost of laying a route on the occupancy matrix without 
 calculating all (NxN) individual square costs. for every point 
 the route passes through, we add the occupancy of that square. */
int calculate_route_cost(const std::vector<std::vector<int>> &occupancy,
                         const Wire &route) {
  Point curr_pt = get_point_i(route, 0);
  int new_count = occupancy[curr_pt.y][curr_pt.x] + 1;
  int route_cost = new_count * new_count;

  // loop through all segments and add the cost of every point
  const int num_segments = route.num_bends + 1;
  for (int s = 0; s < num_segments; s++) {
    // get end point for curr segment
    Point next_pt = get_point_i(route, s + 1);
    // figure out if we are adding to the x coordinate or subtracting
    int step_x = 0;
    if (next_pt.x > curr_pt.x) {
      step_x = 1;
    } else if (next_pt.x < curr_pt.x) {
      step_x = -1;
    }
    // figure out if we are adding to the y coordinate or subtracting
    int step_y = 0;
    if (next_pt.y > curr_pt.y) {
      step_y = 1;
    } else if (next_pt.y < curr_pt.y) {
      step_y = -1;
    }

    // move through from curr_pt to next_pt and add each new point's cost
    while (curr_pt.x != next_pt.x || curr_pt.y != next_pt.y) {
      curr_pt.x += step_x;
      curr_pt.y += step_y;
      new_count = occupancy[curr_pt.y][curr_pt.x] + 1;
      route_cost += new_count * new_count;
    }
  }
  return route_cost;
}

// int calculate_occupancy_cost(const std::vector<std::vector<int>> &occupancy) {
//   int total = 0;
//   for (const auto &row : occupancy) {
//     for (std::size_t col = 0; col < row.size(); col ++) {
//       int count = row[col];
//       total += count * count;
//     }
//   }
//   return total;
// }

// we are building the route based on the index, bounding box 
Wire build_route(const Wire &wire, int route_idx) {
  Wire route = wire;
  const int start_x = wire.start_x;
  const int start_y = wire.start_y;
  const int end_x = wire.end_x;
  const int end_y = wire.end_y;
  const int dx = std::abs(end_x - start_x);
  const int dy = std::abs(end_y - start_y);
  // direction the start pt points towards the end pt if we think about
  // it directionally
  const int x_dir = (end_x > start_x) ? 1 : -1;
  const int y_dir = (end_y > start_y) ? 1 : -1;
  const int routes_per_three_bend_shape = (dx - 1) * (dy - 1);

  // low id chooses horizontal first options: turn at column x1
  if (route_idx < dx) {
    int x1 = start_x + x_dir * (route_idx + 1);
    if (x1 == end_x) {
      route.num_bends = 1;
      route.bends[0] = {end_x, start_y};
    } else {
      route.num_bends = 2;
      route.bends[0] = {x1, start_y};
      route.bends[1] = {x1, end_y};
    }
  // go down vertical first options with v height = 1, 2 ... turn at row y1 into horizontal
  } else if (route_idx < dx + dy) {
    int y1 = start_y + y_dir * (route_idx - dx + 1);
    if (y1 == end_y) {
      route.num_bends = 1;
      route.bends[0] = {start_x, end_y};
    } else {
      route.num_bends = 2;
      route.bends[0] = {start_x, y1};
      route.bends[1] = {end_x, y1};
    }
  } else {
    // pick the middle bend location randomly within the "bounding box"
    // and then set the other two bends to make the route work
    // k = which 3 bend im on, subtract dx + dy because get rid of 1 bend and 2 bend

    // 3-bend routes
    // position within the 3 bend section
    int three_bend_offset = route_idx - (dx + dy);

    // first half of the section is horizontal first, second half is vertical first
    bool starts_horizontal = (three_bend_offset < routes_per_three_bend_shape);
    if (!starts_horizontal) {
      three_bend_offset -= routes_per_three_bend_shape;
    }

    // unflatten the offset
    int turn_column_choice = three_bend_offset / (dy - 1);
    int turn_row_choice    = three_bend_offset % (dy - 1);
    int turn_x = start_x + x_dir * (turn_column_choice + 1);
    int turn_y = start_y + y_dir * (turn_row_choice + 1);

    route.num_bends = 3;
    if (starts_horizontal) {
      route.bends[0] = {turn_x, start_y};
      route.bends[1] = {turn_x, turn_y};
      route.bends[2] = {end_x, turn_y};
    } else {
      route.bends[0] = {start_x, turn_y};
      route.bends[1] = {turn_x, turn_y};
      route.bends[2] = {turn_x, end_y};
    }
  }
  return route;
}

void route_within_wires(std::vector<Wire> &wires, std::vector<std::vector<int>> &occupancy,
  int SA_iters, double SA_prob, int num_threads) {
  // random number generator for simulated annealing
  std::mt19937 rng(676);
  std::bernoulli_distribution choose_random_route(SA_prob);

  // do iterative algorithm
  for (int iters = 0; iters < SA_iters; iters++) {
    for (auto &wire : wires) { // no parallelization across wires
      // straight line conditions
      if (wire.num_bends == 0) {
        continue;
      }
      // remove original route placed in greedy placement so it doesn't count the cost of that layout
      update_occupancy_with_wire(occupancy, wire, -1);

      int dx = std::abs(wire.end_x - wire.start_x);
      int dy = std::abs(wire.end_y - wire.start_y);
      int num_routes = dx + dy + 2 * (dx - 1) * (dy - 1);

      
      Wire best_wire = wire;
      
      // either pick a random route or do a full search for the best route
      if (choose_random_route(rng)) {
        std::uniform_int_distribution<int> pick_random_route(0, num_routes - 1);
        best_wire = build_route(wire, pick_random_route(rng));
      } 
      
      else {
        // CITE: initialize worst cost to highest number so better costs can compare if <
        int best_cost = std::numeric_limits<int>::max();
        int best_route_idx = num_routes;

        // parallelize the candidate search across threads (threads might have to deal with more than 1 route)
        // threads score every possible route in parallel, and each one updates its private best variables
        #pragma omp parallel num_threads(num_threads)
        {
          int local_best_cost = std::numeric_limits<int>::max();
          int local_best_idx = num_routes;

          #pragma omp for schedule(static) nowait
          for (int route_idx = 0; route_idx < num_routes; route_idx++) {

            // calculate a route given the enumeration strat we have with route_idx
            Wire new_route = build_route(wire, route_idx);
            int new_cost = calculate_route_cost(occupancy, new_route);

            if (new_cost < local_best_cost || (new_cost == local_best_cost &&
                route_idx < local_best_idx)) { // ties go to the lowest index
              local_best_cost = new_cost;
              local_best_idx = route_idx;
            }
          }

          // now we have best, best result
          // among all the thread's private local bests pick the lowest cost, so we have the best amongst all threads
          #pragma omp critical
          {
            if (local_best_cost < best_cost ||
                (local_best_cost == best_cost &&
                local_best_idx < best_route_idx)) {
              best_cost = local_best_cost;
              best_route_idx = local_best_idx;
            }
          }
        }
        best_wire = build_route(wire, best_route_idx);
      }
      // actually update the occupancy matrix with the new best wire
      wire = best_wire;
      update_occupancy_with_wire(occupancy, wire, 1);
    }
  }
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
    wire.num_bends = 0;
    fin >> wire.start_x >> wire.start_y >> wire.end_x >> wire.end_y;
    // wire.move_x_start = true;
    // wire.move_x_end = false;
    // wire.mid_x = wire.end_x;
    // wire.mid_y = wire.start_y;
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

  // TODO WE HAVE TO PARSE THE INPUT WIRE START AND END AND POPULATE THE
  // WIRES list, making sure we 0 initiate all parameters
    
  // place initial route using greedy heuristic
  for (auto &wire : wires){ // OPENMP PARALLELIZE: each wire is independent for this initial laying
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
      wire.bends[0] = {wire.start_x, wire.end_y};
      wire.num_bends = 1;
      update_occupancy_with_wire(occupancy, wire, 1);
    }
  }
  
  // initialize wires
  // Within wires
  if (parallel_mode == 'W') {
    // within wires
    route_within_wires(wires, occupancy, SA_iters, SA_prob, num_threads);
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
  // keypoints are start, every bend, then end
  w.num_pts = num_bends + 2;
  for (int i = 0; i < w.num_pts; i++) {
    Point pt = get_point_i(*this, i);
    w.p[i].x = pt.x;
    w.p[i].y = pt.y;
  }
  return w;
}
