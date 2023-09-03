#include "Generator.h"

Generator::Generator(int taille) : d_taille{ taille } { srand(time(NULL)); }

double Generator::randomNumber(double min, double max) {
	return min + (rand() / (RAND_MAX / (max - min)));
}

int Generator::randomNumber(int min, int max) {
	return min + (rand() / (RAND_MAX / (max - min)));
}

Mat Generator::generateWithPoints() {
	Mat image = Mat::zeros(d_taille, d_taille, CV_8UC1);
	int nbr = 0.05 * d_taille * d_taille;
	for (int i = 0; i < nbr; i++) {
		int x = randomNumber(0, d_taille);
		int y = randomNumber(0, d_taille);
		image.at<uchar>(x, y) = 255;
	}
	bitwise_not(image, image);
	Mat image300dpi;
	resize(image, image300dpi, Size(), 10, 10, INTER_NEAREST);
	return image300dpi;
}

Mat Generator::generateWithCurves() {
	Mat image = Mat::zeros(d_taille, d_taille, CV_8UC1);
	// Ajout de courbes 
	int nbrCurves = 1 + rand() % 5;
	vector<Point> points;
	for (int i = 0; i < nbrCurves; i++) {
		Point p1(
			randomNumber(0, d_taille),
			randomNumber(0, d_taille)
		);
		Point p2(
			randomNumber(0, d_taille),
			randomNumber(0, d_taille)
		);
		points.push_back(p1);
		points.push_back(p2);
	}
	for (int i = 0; i < nbrCurves - 1; i++) {
		line(image, points[i], points[i + 1], Scalar(255), 1, 8);
	}
	bitwise_not(image, image);
	Mat image300dpi;
	resize(image, image300dpi, Size(), 10, 10, INTER_NEAREST);
	return image;
}