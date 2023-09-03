#pragma once

#include <torch/torch.h>

class Discriminator : public torch::nn::Module {
public:
	Discriminator();
	torch::Tensor forward(torch::Tensor x);
	torch::Tensor predict(torch::Tensor x);
private:
	torch::nn::Conv2d conv1;
	torch::nn::Conv2d conv2;
	torch::nn::Dropout2d conv2_drop;
	torch::nn::Linear fc1;
	torch::nn::Linear fc2;
};
