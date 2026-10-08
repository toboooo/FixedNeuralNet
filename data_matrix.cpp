#include <fstream>
#include <ios>
#include <iostream>
#include <sstream>
#include <string>
#include "data_matrix.hpp"

DataMatrix::~DataMatrix() {
	delete[] m;
	delete[] y;
}

DataMatrix::DataMatrix(const std::string &data_filename, const int n_features,
const int sample_size) {
	std::ifstream data_file(data_filename);
	if (!data_file.is_open()) {
		std::cerr << "Error: Failed to open data file\n";
		std::exit(0);
	}
	rows = n_features;
	cols = sample_size;
	m = new double[rows*cols];
	y = new double[sample_size];
	std::string line;
	std::stringstream line_stream;
	std::getline(data_file, line);
	for (int i = 0; i < sample_size; ++i) {
		std::getline(data_file, line);
		line_stream.clear();
		line_stream.str(line);
		for (int j = 0; j < n_features; ++j) {
			line_stream >> m[j*cols+i];
			if (line_stream.fail() || line_stream.eof()) {
				std::cerr << "Error: Failed to read from data file\n";
				std::exit(0);
			}
			line_stream.ignore(1, ',');
		}
		line_stream >> y[i];
		if (line_stream.fail() || line_stream.eof()) {
			std::cerr << "Error: Failed to read from data file\n";
			std::exit(0);
		}
	}
	data_file.close();
}
