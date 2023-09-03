#pragma once

#include <torch/torch.h>
#include <torch/script.h>

#include <opencv2/core.hpp>

class Generator : public torch::jit::Module {
private:
	torch::jit::Module module;
public:
	Generator(const std::string& path);

	torch::Tensor forward(std::vector<torch::jit::IValue> x);

	torch::Tensor generate();
};
