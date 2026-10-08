#include <cmath>
#include <fstream>
#include <ios>
#include <iostream>
#include <string>
#include "network.hpp"

Matrix::Matrix(const Matrix &other) {
	int n_elem = other.rows * other.cols;
	m = new double[n_elem];
	rows = other.rows;
	cols = other.cols;
	std::copy(other.m, other.m + n_elem, m);
}

Matrix::Matrix(const int n_row, const int n_col) {
	m = new double[n_row*n_col];
	rows = n_row;
	cols = n_col;
}

Matrix::Matrix(Matrix &&other) noexcept {
	m = other.m;
	rows = other.rows;
	cols = other.cols;
	other.m = nullptr;
	other.rows = 0;
	other.cols = 0;
}

Matrix::~Matrix() {
	delete[] m;
}

void Matrix::mult_matrix(const Matrix &other, Matrix &output) const {
	if (cols != other.rows) {
		std::cerr << "mult_weights: dimension mismatch with other\n";
		std::exit(1);
	}
	else if (output.rows != rows || output.cols != other.cols) {
		std::cerr << "mult_weights: dimension mismatch with output\n";
		std::exit(1);
	}
	double *temp_m = new double[other.cols*other.rows];
	for (int i = 0; i < other.rows; ++i) {
		for (int j = 0; j < other.cols; ++j) {
			temp_m[j*other.rows+i] = other.m[i*other.cols+j];
		}
	}
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < other.cols; ++j) {
			double elem = 0.0;
			for (int k = 0; k < cols; ++k) {
				elem += m[i*cols+k] * temp_m[j*other.rows+k];
			}
			output.m[i*output.cols+j] = elem;
		}
	}
	delete[] temp_m;
}

double Matrix::trace() const {
	double tr = 0.0;
	for (int i = 0; i < rows && i < cols; ++i) {
		tr += m[i*cols+i];
	}
	return tr;
}

double Matrix::frobenius_norm() const {
	double norm = 0.0;
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			norm += m[i*cols+j] * m[i*cols+j];
		}
	}
	return std::sqrt(norm);
}

Weights::Weights(const std::string &weights_filename) {
	std::ifstream weights_file(weights_filename, std::ios::binary);
	if (!weights_file.is_open()) {
		std::exit(1);
	}
	weights_file.read(reinterpret_cast<char*>(&rows), sizeof(int));
	weights_file.read(reinterpret_cast<char*>(&cols), sizeof(int));
	const int n_elem = rows * cols;
	m = new double[n_elem];
	weights_file.read(reinterpret_cast<char*>(m), n_elem * sizeof(double));
	weights_file.close();
}

Biases::Biases(const std::string &biases_filename) {
	std::ifstream biases_file(biases_filename, std::ios::binary);
	if (!biases_file.is_open()) {
		std::exit(1);
	}
	biases_file.read(reinterpret_cast<char*>(&rows), sizeof(int));
	cols = 1;
	m = new double[rows];
	biases_file.read(reinterpret_cast<char*>(m), rows * sizeof(double));
	biases_file.close();
}

void Network::forward_pass(Matrix &short_output, Matrix &residual,
Matrix &wide_output, Matrix &output) {
	for (const std::pair<int,int> &inout_indices : weight_indices) {
		int n_elem = short_output.rows * short_output.cols;
		std::copy(short_output.m, short_output.m + n_elem, residual.m);
		int in_index = inout_indices.first;
		int out_index = inout_indices.second;
		const Weights &in_weight = weights[in_index];
		const Weights &out_weight = weights[out_index];
		const Biases &in_bias = biases[in_index];
		const Biases &out_bias = biases[out_index];
		in_weight.mult_matrix(short_output, wide_output);
		add_bias(wide_output, in_bias);
		relu_activation(wide_output);
		out_weight.mult_matrix(wide_output, short_output);
		add_bias(short_output, out_bias);
		add_residual(short_output, residual);
	}
	const Weights &last_weight = weights[last];
	const Biases &last_bias = biases[last];
	last_weight.mult_matrix(short_output, output);
	add_bias(output, last_bias);
}

void Network::add_bias(Matrix &output, const Biases &bias) {
	if (bias.rows != output.rows) {
		std::cerr << "add_bias: bias dimension mismatch\n";
		std::exit(0);
	}
	for (int i = 0; i < output.rows; ++i) {
		for (int j = 0; j < output.cols; ++j) {
			output.m[i*output.cols+j] += bias.m[i];
		}
	}
}

void Network::relu_activation(Matrix &output) {
	for (int i = 0; i < output.rows; ++i) {
		for (int j = 0; j < output.cols; ++j) {
			double value = output.m[i*output.cols+j];
			value = value > 0.0 ? value : 0.0;
			output.m[i*output.cols+j] = value;
		}
	}
}

void Network::add_residual(Matrix &output, const Matrix &residual) {
	if (output.rows != residual.rows || output.cols != residual.cols) {
		std::cerr << "add_residual: dimension mismatch\n";
		std::exit(0);
	}
	for (int i = 0; i < output.rows; ++i) {
		for (int j = 0; j < output.cols; ++j) {
			output.m[i*output.cols+j] += residual.m[i*residual.cols+j];
		}
	}
}
