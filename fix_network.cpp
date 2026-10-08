#include <algorithm>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#include <dirent.h>
#include <sys/types.h>
#include "data_matrix.hpp"
#include "network.hpp"

constexpr int BATCH_SIZE = 1000;

bool string_end_compare(const std::string &a, const std::string &end) {
	int i = a.size() - 1;
	int j = end.size() - 1;
	while (i >= 0 && j >= 0) {
		if (a[i--] != end[j--]) {
			return false;
		}
	}
	if (i < 0 && j >= 0) {
		return false;
	}
	return true;
}

std::vector<Weights> load_network_weights(const std::string &path) {
	std::vector<Weights> weights;
	int n_weights = 0;
	DIR *directory_p = opendir(path.data());
	struct dirent *dirent_p;
	while (dirent_p = readdir(directory_p)) {
		std::string filename = std::string(dirent_p->d_name);
		if (!string_end_compare(filename, "w.dat")) {
			continue;
		}
		++n_weights;
	}
	for (int i = 0; i < n_weights; ++i) {
		std::string filename = "pieces/piece_" + std::to_string(i) + "_w.dat";
		weights.push_back(Weights(filename));
	}
	return weights;
}

std::vector<Biases> load_network_biases(const std::string &path) {
	std::vector<Biases> biases;
	int n_biases = 0;
	DIR *directory_p = opendir(path.data());
	struct dirent *dirent_p;
	while (dirent_p = readdir(directory_p)) {
		std::string filename = std::string(dirent_p->d_name);
		if (!string_end_compare(filename, "b.dat")) {
			continue;
		}
		++n_biases;
	}
	for (int i = 0; i < n_biases; ++i) {
		std::string filename = "pieces/piece_" + std::to_string(i) + "_b.dat";
		biases.push_back(Biases(filename));
	}
	return biases;
}

std::vector<int> get_matching_weights(std::vector<Weights> &weights, int in_dim,
int out_dim) {
	std::vector<int> matching_indices;
	for (int i = 0; i < weights.size(); ++i) {
		if (weights[i].rows == in_dim && weights[i].cols == out_dim) {
			matching_indices.push_back(i);
		}
	}
	return matching_indices;
}

std::vector<std::pair<int,int> > pair_weights(std::vector<Weights> &weights,
std::vector<int> &in_indices, std::vector<int> &out_indices) {
	Weights output(LO_DIM, LO_DIM);
	std::vector<std::pair<int,int> > paired;
	std::vector<bool> used(in_indices.size());
	for (int i = 0; i < out_indices.size(); ++i) {
		int out_index = out_indices[i];
		const Weights &out_weight = weights[out_index];
		double max_ratio = std::numeric_limits<double>::min();
		int match_in_index = -1;
		for (int j = 0; j < in_indices.size(); ++j) {
			int in_index = in_indices[j];
			if (used[in_index]) {
				continue;
			}
			const Weights &in_weight = weights[in_index];
			out_weight.mult_matrix(in_weight, output);
			double tr = output.trace();
			double frob_norm = output.frobenius_norm();
			double ratio = std::abs(tr) / frob_norm;
			if (ratio > max_ratio) {
				max_ratio = ratio;
				match_in_index = in_index;
			}
		}
		if (match_in_index == -1) {
			std::cerr << "Error: Did not find matching index\n";
			std::exit(1);
		}
		else {
			std::cout << "Matched " << match_in_index << " against "
				<< out_index << '\n';
		}
		used[match_in_index] = true;
		paired.push_back(std::make_pair(match_in_index, out_index));
	}
	return paired;
}

double mean_squared_error(const DataMatrix &data, const Matrix &output) {
	if (data.cols != output.cols) {
		std::cerr << "Error: Mean square error calculation row mismatch\n";
		std::exit(0);
	}
	double error = 0.0;
	for (int i = 0; i < data.cols; ++i) {
		double d = data.y[i] - output.m[i];
		error += d * d;
	}
	error /= data.cols;
	return error;
}

void bubble_repair_network(Network &network, const DataMatrix &data,
Matrix &short_output, Matrix &residual, Matrix &wide_output, Matrix &output) {
	network.forward_pass(short_output, residual, wide_output, output);
	double curr_error = mean_squared_error(data, output);
	int n_swaps;
	int total_swaps = 0;
	do {
		n_swaps = 0;
		for (int i = 1; i < network.weight_indices.size(); ++i) {
			std::swap(network.weight_indices[i-1], network.weight_indices[i]);
			int n_elem = data.rows * data.cols;
			std::copy(data.m, data.m + n_elem, short_output.m);
			network.forward_pass(short_output, residual, wide_output, output);
			double error = mean_squared_error(data, output);
			if (error < curr_error) {
				++n_swaps;
				curr_error = error;
				std::cout << "New lowest error: " << curr_error << '\n';
			}
			else {
				std::swap(network.weight_indices[i-1],
					network.weight_indices[i]);
			}
		}
		total_swaps += n_swaps;
		std::cout << "Swaps: " << n_swaps << " MSE: " << curr_error
			<< " Total swaps: " << total_swaps << '\n';
	} while (n_swaps > 0);
}

int main() {
	std::vector<Weights> weights = load_network_weights("pieces");
	std::vector<Biases> biases = load_network_biases("pieces");
	std::vector<int> in_weights = get_matching_weights(weights, HI_DIM, LO_DIM);
	std::vector<int> out_weights = get_matching_weights(weights, LO_DIM,
		HI_DIM);
	int last = get_matching_weights(weights, 1, 48)[0];
	std::vector<std::pair<int,int> > paired_weights = pair_weights(weights,
		in_weights, out_weights);
	std::vector<double> out_frob_norms(weights.size());
	for (const std::pair<int,int> pairing : paired_weights) {
		const Weights &out_weight = weights[pairing.second];
		out_frob_norms[pairing.second] = out_weight.frobenius_norm();
	}
	std::function<bool(const std::pair<int,int>&,const std::pair<int,int>&)>
	cmp_out_frob_norm = [&out_frob_norms](const std::pair<int,int> &a,
	const std::pair<int,int> &b) {
		return out_frob_norms[a.second] < out_frob_norms[b.second];
	};
	std::sort(paired_weights.begin(), paired_weights.end(), cmp_out_frob_norm);
	Network network(weights, biases, paired_weights, last);
	DataMatrix data("historical_data.csv", LO_DIM, BATCH_SIZE);
	Matrix short_output(data);
	Matrix residual(short_output.rows, short_output.cols);
	Matrix wide_output(HI_DIM, BATCH_SIZE);
	Matrix output(1, BATCH_SIZE);
	bubble_repair_network(network, data, short_output, residual, wide_output,
		output);
	for (const std::pair<int,int> &indices : network.weight_indices) {
		std::cout << indices.first << ',' << indices.second << ',';
	}
	std::cout << network.last << '\n';
	return 0;
}
