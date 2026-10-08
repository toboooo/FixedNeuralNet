#ifndef NETWORK_HPP
#define NETWORK_HPP
#include <string>
#include <utility>
#include <vector>

constexpr int LO_DIM = 48;
constexpr int HI_DIM = 96;

class Matrix {
	public:
	int rows = 0;
	int cols = 0;
	double *m = nullptr;
	Matrix() {};
	Matrix(const Matrix &other);
	Matrix(const int n_row, const int n_col);
	Matrix(Matrix &&other) noexcept;
	~Matrix();
	void mult_matrix(const Matrix &other, Matrix &output) const;
	double trace() const;
	double frobenius_norm() const;
};

class Weights : public Matrix {
	public:
	Weights() {}
	Weights(const Weights &other) : Matrix(other) {}
	Weights(const int n_row, const int n_col) : Matrix(n_row, n_col) {}
	Weights(Weights &&other) noexcept : Matrix(other) {}
	Weights(const std::string &weights_filename);
};

class Biases : public Matrix {
	public:
	Biases() {}
	Biases(const Biases &other) : Matrix(other) {}
	Biases(const int n_row, const int n_col) : Matrix(n_row, n_col) {}
	Biases(Biases &&other) noexcept : Matrix(other) {}
	Biases(const std::string &biases_filename);
};

class Network {
	public:
	std::vector<Weights> &weights;
	std::vector<Biases> &biases;
	std::vector<std::pair<int,int> > weight_indices;
	int last;
	Network(std::vector<Weights> &raw_weights, std::vector<Biases> &raw_biases,
	std::vector<std::pair<int,int> > &paired_weights, int last_index)
	: weights(raw_weights), biases(raw_biases), weight_indices(paired_weights),
	last(last_index) {}
	void forward_pass(Matrix &short_output, Matrix &residual,
	Matrix &wide_output, Matrix &output);
	void add_bias(Matrix &output, const Biases &bias);
	void relu_activation(Matrix &output);
	void add_residual(Matrix &output, const Matrix &residual);
};

#endif
