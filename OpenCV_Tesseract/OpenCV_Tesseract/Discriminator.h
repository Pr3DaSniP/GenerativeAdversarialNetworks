#pragma once

#include <leptonica/allheaders.h>
#include <tesseract/baseapi.h>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class Discriminator {
private:
	tesseract::TessBaseAPI* d_api;
	string d_model;
public:
	Discriminator(const string& pathToTrainedFiles, const string& model, const string& whitelist);

	// Retourne ce que le discriminateur voit
	string predict(const Mat& input);

	// Retourne la probabilité que le discriminateur voit ce qu'il voit
	float predictProbability(const Mat& input);
	
	// Retourne le nom du model choisit
	string getModel();
};