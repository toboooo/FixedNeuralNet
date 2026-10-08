#ifndef DATA_MATRIX_HPP
#define DATA_MATRIX_HPP
#include "network.hpp"

class DataMatrix : public Matrix {
	public:
	double *y = nullptr;
	DataMatrix() {}
	DataMatrix(const DataMatrix &other) : Matrix(other) {}
	DataMatrix(const int n_row, const int n_col) : Matrix(n_row, n_col) {}
	DataMatrix(DataMatrix &&other) noexcept : Matrix(other) {}
	DataMatrix(const std::string &csv_filename, const int n_features,
	const int sample_size);
	~DataMatrix();
};

#endif
