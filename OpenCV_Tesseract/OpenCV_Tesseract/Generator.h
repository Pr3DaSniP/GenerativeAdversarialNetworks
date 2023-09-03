#pragma once

#include <iostream>
#include <vector>
#include <time.h>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class Generator {
private:
	int d_taille;

	// Génère un nombre aléatoire entre min et max
	double randomNumber(double min, double max);

	// Génère un entier aléatoire entre min et max
	int randomNumber(int min, int max);
	
public:
	Generator(int taille);

	// Génère une image avec des points aléatoires
	Mat generateWithPoints();

	// Génère une image avec des courbes aléatoires
	Mat generateWithCurves();
};
