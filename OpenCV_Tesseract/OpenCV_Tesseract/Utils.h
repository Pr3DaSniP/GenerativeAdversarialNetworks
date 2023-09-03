#pragma once

#include <vector>
#include <iostream>
#include <string>
#include <fstream>

#include <opencv2/opencv.hpp>

#include "Discriminator.h"
#include "Generator.h"

using namespace std;

const int taille = 28;

const std::string pathToMNISTDataset = "../../mnist/";

vector<string> loadTrainedDataFiles();

namespace test {
	void testingAllTraineddataFileOnMNIST();

	void testing(Discriminator& dis);
}

void gan(Generator& gen, Discriminator& dis, int nbImagesToGenerate);