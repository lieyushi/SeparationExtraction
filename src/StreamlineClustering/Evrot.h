/*
 * evrot.h
 *
 *  https://github.com/pthimon/clustering
 *  Class to compute the gradient of the eigenvectors
 *  alignment quality
 *
 *  Lihi Zelnik (Caltech) March.2005
 *
 *  Created on: 02-Mar-2009
 *      Author: sbutler
 *
 */

#ifndef EVROT_H_
#define EVROT_H_

#include <iostream>
#include <vector>
#include <math.h>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <exception>

using namespace Eigen;
using namespace std;

#define EPS 2.2204e-8

class Evrot {

public:
	Evrot(const Eigen::MatrixXd& X, int method);
	virtual ~Evrot();
	double getQuality() { return mQuality; }
	std::vector<std::vector<int> > getClusters() { return mClusters; }
	Eigen::MatrixXd& getRotatedEigenVectors() { return mXrot; }

protected:
	void evrot();
	void cluster_assign();
	double evqual(const Eigen::MatrixXd& X);
	double evqualitygrad(const Eigen::VectorXd& theta, const int& angle_index);
	Eigen::MatrixXd rotate_givens(const Eigen::VectorXd& theta);
	Eigen::MatrixXd build_Uab(const Eigen::VectorXd& theta, const int& a, const int& b);
	Eigen::MatrixXd gradU(const Eigen::VectorXd& theta, const int& k);

	//constants

	int mMethod;
	const int mNumDims;
	const int mNumData;
	int mNumAngles;
	Eigen::VectorXi ik;
	Eigen::VectorXi jk;

	//current iteration
	Eigen::MatrixXd mX;
	Eigen::MatrixXd mXrot;
	double mQuality;

	std::vector<std::vector<int> > mClusters;
};

#endif /* EVROT_H_ */
