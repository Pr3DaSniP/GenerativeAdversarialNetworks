#include "Utils.h"

vector<string> loadTrainedDataFiles() {
	vector<string> traineddataFiles;
	cv::glob("traineddata/*.traineddata", traineddataFiles);
	for (int i = 0; i < traineddataFiles.size(); i++) {
		// Remove .traineddata from the name
		string model = traineddataFiles[i].substr(0, traineddataFiles[i].size() - 12);
		// Remove the path
		model = model.substr(model.find_last_of("\\") + 1, model.size());
		traineddataFiles[i] = model;
	}
	return traineddataFiles;
}

namespace test {

	void testingAllTraineddataFileOnMNIST() {
		auto files = loadTrainedDataFiles();

		for (int i = 0; i < files.size(); i++) {
			Discriminator d("traineddata", files[i], "0123456789");
			testing(d);
		}
	}

	void testing(Discriminator& dis) {
		cout << "Testing for " << dis.getModel() << " model on all MNIST images" << endl;
		ofstream file("tests/" + dis.getModel() + ".traineddata.csv");
		file << "Classe" << ";" << "Total" << ";" << "Number of good predictions" << ";" << "Number of bad predictions" << ";" << "Percentage of good predictions" << endl;
		vector<string> files;
		for (int j = 0; j < 10; ++j) {
			glob(pathToMNISTDataset + to_string(j) + "/*.png", files);
			int nb = 0;
			for (int i = 0; i < files.size(); i++) {
				//cout << "Testing... : " << files[i] << endl;
				Mat image = imread(files[i]);
				string predic = dis.predict(image);
				if (predic[0] == to_string(j)[0]) nb++;
			}
			file << j << ";" << files.size() << ";" << nb << ";" << files.size() - nb << ";" << (float)nb / (float)files.size() * 100 << "%" << endl;
		}
		file.close();
	}
	
}

void gan(Generator& gen, Discriminator& dis, int nbImagesToGenerate) {
	int nb = 0;
	while (nb < nbImagesToGenerate) {
		Mat image = gen.generateWithCurves();
		string prediction = dis.predict(image);
		float proba = dis.predictProbability(image);
		if (prediction != "" && proba > 90) {
			bitwise_not(image, image);
			resize(image, image, Size(taille, taille));
			cout << "Predict : " << prediction[0] << " ------ Proba :" << proba << endl;
			switch (prediction[0]) {
			case '0':
				imwrite("output/0/" + to_string(rand()) + ".png", image);
				nb++; break;
			case '1':
				imwrite("output/1/" + to_string(rand()) + ".png", image);
				nb++; break;
			case '2':
				imwrite("output/2/" + to_string(rand()) + ".png", image);
				nb++; break;
			case '3':
				imwrite("output/3/" + to_string(rand()) + ".png", image);
				nb++; break;
			case '4':
				imwrite("output/4/" + to_string(rand()) + ".png", image);
				nb++; break;
			case '5':
				imwrite("output/5/" + to_string(rand()) + ".png", image);
				nb++; break;
			case '6':
				imwrite("output/6/" + to_string(rand()) + ".png", image);
				nb++; break;
			case '7':
				imwrite("output/7/" + to_string(rand()) + ".png", image);
				nb++; break;
			case '8':
				imwrite("output/8/" + to_string(rand()) + ".png", image);
				nb++; break;
			case '9':
				imwrite("output/9/" + to_string(rand()) + ".png", image);
				nb++; break;
			}
		}
		else {
			cout << "No prediction but it remains " << nbImagesToGenerate - nb << " images to generate." << endl;
		}
	}
}