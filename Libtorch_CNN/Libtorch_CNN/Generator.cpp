#include "Generator.h"
#include <opencv2/highgui.hpp>

Generator::Generator(const std::string& path) {
	try {
		module = torch::jit::load(path);
	}
	catch (const c10::Error& e) {
		std::cerr << e.what() << std::endl;
	}
}

torch::Tensor Generator::forward(std::vector<torch::jit::IValue> x) {
	return module.forward({ x }).toTensor();
}

torch::Tensor Generator::generate() {
	std::vector<torch::jit::IValue> inputs;
	torch::Tensor t = torch::randn({ 1, 100 });
	inputs.push_back(t);
	return module.forward(inputs).toTensor();
}