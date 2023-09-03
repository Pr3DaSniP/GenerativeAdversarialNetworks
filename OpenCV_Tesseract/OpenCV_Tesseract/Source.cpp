// ---- OPENCV ----
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

// ---- TESSERACT ----
#include <leptonica/allheaders.h>
#include <tesseract/baseapi.h>

#include <iostream>

#include "Generator.h"
#include "Discriminator.h"
#include "Utils.h"

using namespace std;

int main() {

	// Générateur
	Generator gen(taille);
	
	// Discriminateur
	/*
	* 
	* "mnist2" est le nom du modele pour l'OCR (voir le dossier "traineddata")
	* 
	* Modele dispo : digits, digits_comma, digits1, eng, equ, mnist1, mnist2
	* 
	*/
	Discriminator dis("traineddata", "mnist2", "0123456789");

	// GAN
	gan(gen, dis, 5);

	return 0;
}