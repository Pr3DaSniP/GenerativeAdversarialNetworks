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

	// G�n�re un nombre al�atoire entre min et max
	double randomNumber(double min, double max);

	// G�n�re un entier al�atoire entre min et max
	int randomNumber(int min, int max);
	
public:
	Generator(int taille);

	// G�n�re une image avec des points al�atoires
	Mat generateWithPoints();

	// G�n�re une image avec des courbes al�atoires
	Mat generateWithCurves();
};
