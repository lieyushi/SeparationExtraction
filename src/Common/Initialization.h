#ifndef _INITIALIZATION_H_
#define _INITIALIZATION_H_

#include <eigen3/Eigen/Dense>
#include <vector>
#include <ctime>
#include <cassert>
#include <iostream>

using namespace std;
using namespace Eigen;


class Initialization
{
public:
	static void generateRandomPos(MatrixXd& clusterCenter,
								  const int& column,
								  const MatrixXd& cArray,
								  const int& Cluster);

	static void generateFromSamples(MatrixXd& clusterCenter,
								    const int& column,
								    const MatrixXd& cArray,
								    const int& Cluster);

};


#endif
