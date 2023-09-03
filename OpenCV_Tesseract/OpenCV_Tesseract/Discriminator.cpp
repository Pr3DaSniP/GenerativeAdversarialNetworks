#include "Discriminator.h"

Discriminator::Discriminator(const string& pathToTrainedFiles, const string& model, const string& whitelist) : d_model{ model } {
	d_api = new tesseract::TessBaseAPI;
	if (d_api->Init(pathToTrainedFiles.c_str(), model.c_str(), tesseract::OEM_DEFAULT)) {
		fprintf(stderr, "Could not initialize tesseract.\n");
		exit(1);
	}
	d_api->SetPageSegMode(tesseract::PSM_SINGLE_CHAR);
	d_api->SetVariable("tessedit_char_whitelist", whitelist.c_str());
	d_api->SetVariable("debug_file", "/dev/null");
};

string Discriminator::predict(const Mat& input) {
	d_api->SetImage(input.data, input.cols, input.rows, input.channels(), input.step1());
	d_api->Recognize(0);
	char* outText = d_api->GetUTF8Text();
	string outputText(outText);
	return outputText;
}

float Discriminator::predictProbability(const Mat& input) {
	d_api->SetImage(input.data, input.cols, input.rows, input.channels(), input.step1());
	d_api->Recognize(0);
	return d_api->MeanTextConf();
}

string Discriminator::getModel() {
	return d_model;
}