// ---- PYTORCH ----
#include <torch/torch.h>
#include <torch/script.h>

// ---- OPENCV ----
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include "Discriminator.h"
#include "Generator.h"

using namespace std;
using namespace cv;

const string pathToMNISTDataset = "../../mnist/";

void testDIS() {
	try {
		string path = "models/discriminator.pt";
		torch::serialize::InputArchive inputArchive;
		inputArchive.load_from(path);
		Discriminator model;
		model.load(inputArchive);

		std::vector<torch::jit::IValue> inputs;
		cv::Mat img = cv::imread(pathToMNISTDataset + "8/188.png", cv::IMREAD_GRAYSCALE);
		auto a = torch::from_blob(img.data, { 1, 1, 28, 28 }, torch::kByte).toType(torch::kFloat).div(255);
		torch::Tensor output = model.predict(a);
		auto max_value = output.max(1, true);
		auto max_index = std::get<1>(max_value).item<int64_t>();
		std::cout << "Predicted: " << max_index << std::endl;
	}
	catch (const c10::Error& e) {
		std::cerr << e.what() << std::endl;
	}
}

void testGEN() {
	try{
		// DCGAN_generateur ou GAN_generateur
		Generator generator("models/DCGAN_generateur.pt");
		while (true) {
			torch::Tensor output = generator.generate();
			cv::Mat img = cv::Mat(28, 28, CV_32FC1, output.data_ptr());
			cv::threshold(img, img, 0.5, 1.0, cv::THRESH_BINARY);
			cv::resize(img, img, cv::Size(280, 280));
			cv::imshow("img", img);
			cv::waitKey(0);
		}
	}
	catch (const c10::Error& e) {
		std::cerr << e.what() << std::endl;
	}
}

void gan(Generator& generator, Discriminator& discriminator, int nbImagesToGenerate) {
	int nb = 0;
	while (nb < nbImagesToGenerate) {
		torch::Tensor outputGEN = generator.generate();
		outputGEN = outputGEN.mul(255).clamp(0, 255).to(torch::kU8);
		cv::Mat img = cv::Mat(28, 28, CV_8U, outputGEN.data_ptr());
		auto a = torch::from_blob(img.data, { 1, 1, 28, 28 }, torch::kByte).toType(torch::kFloat).div(255);
		torch::Tensor outputDIS = discriminator.predict(a);
		auto max_value = outputDIS.max(1, true);
		auto predict = get<1>(max_value).item<int64_t>();
		cout << "Predicted: " << predict << endl;
		switch (predict) {
		case 0:
			imwrite("output/0/" + to_string(rand()) + ".png", img);
			nb++; break;
		case 1:
			imwrite("output/1/" + to_string(rand()) + ".png", img);
			nb++; break;
		case 2:
			imwrite("output/2/" + to_string(rand()) + ".png", img);
			nb++; break;
		case 3:
			imwrite("output/3/" + to_string(rand()) + ".png", img);
			nb++; break;
		case 4:
			imwrite("output/4/" + to_string(rand()) + ".png", img);
			nb++; break;
		case 5:
			imwrite("output/5/" + to_string(rand()) + ".png", img);
			nb++; break;
		case 6:
			imwrite("output/6/" + to_string(rand()) + ".png", img);
			nb++; break;
		case 7:
			imwrite("output/7/" + to_string(rand()) + ".png", img);
			nb++; break;
		case 8:
			imwrite("output/8/" + to_string(rand()) + ".png", img);
			nb++; break;
		case 9:
			imwrite("output/9/" + to_string(rand()) + ".png", img);
			nb++; break;
		}
	}
}

void testGAN() {
	try {
		Generator generator("models/GAN_generateur.pt");

		string path = "models/discriminator.pt";
		torch::serialize::InputArchive inputArchive;
		inputArchive.load_from(path);
		Discriminator model;
		model.load(inputArchive);

		gan(generator, model, 3);
	}
	catch (const c10::Error& e) {
		std::cerr << e.what() << std::endl;
	}
}

void testCNNOnAllMNIST() {
	string path = "models/discriminator.pt";
	torch::serialize::InputArchive inputArchive;
	inputArchive.load_from(path);
	Discriminator model;
	model.load(inputArchive);

	cout << "Testing for CNN model on all MNIST images" << endl;
	ofstream file("tests/cnn.csv");
	file << "Classe" << ";" << "Total" << ";" << "Number of good predictions" << ";" << "Number of bad predictions" << ";" << "Percentage of good predictions" << endl;
	vector<string> files;
	for (int j = 0; j < 10; ++j) {
		glob(pathToMNISTDataset + to_string(j) + "/*.png", files);
		int nb = 0;
		for (int i = 0; i < files.size(); i++) {
			cout << "Testing... : " << files[i] << endl;
			Mat image = imread(files[i], cv::IMREAD_GRAYSCALE);
			std::vector<torch::jit::IValue> inputs;
			auto a = torch::from_blob(image.data, { 1, 1, 28, 28 }, torch::kByte).toType(torch::kFloat).div(255);
			torch::Tensor output = model.predict(a);
			auto max_value = output.max(1, true);
			auto predict = std::get<1>(max_value).item<int64_t>();
			if (predict == j) nb++;
		}
		file << j << ";" << files.size() << ";" << nb << ";" << files.size() - nb << ";" << (float)nb / (float)files.size() * 100 << "%" << endl;
	}
	file.close();
}

int main() {
	testGAN();
	
	return 0;
}