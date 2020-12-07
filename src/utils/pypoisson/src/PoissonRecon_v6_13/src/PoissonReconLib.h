#pragma once

#include <vector>
extern std::vector<double> double_data;
extern std::vector<int> int_data;
extern std::vector<double> mem_data;
extern bool* is_verbose;


int PoissonReconLibMain(int argc, char* argv[]);
