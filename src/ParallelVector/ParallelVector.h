/*
 * ParallelVector.h
 *
 *  Created on: Apr 27, 2019
 *      Author: lieyu
 */

#ifndef SRC_PARALLELVECTOR_PARALLELVECTOR_H_
#define SRC_PARALLELVECTOR_PARALLELVECTOR_H_

#include <iostream>
#include <vector>
#include <math.h>
#include <fstream>
#include <stdlib.h>
#include <float.h>
#include <string>
#include <time.h>
#include <tuple>
#include <sstream>
#include <algorithm>
#include <queue>
#include <cassert>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SVD>
#include <unordered_map>
#include "ReadData.h"

using namespace std;

class ParallelVector {
public:

	virtual ~ParallelVector();

	/*
	 * calculate the PV operator for input vector field
	 */
	ParallelVector(const string& fileName, std::vector<Vertex>& vertexVec, CoordinateLimits li[3],
			const int& x_resolution, const int& y_resolution, const int& z_resolution,
			const double& x_step, const double& y_step, const double& z_step);

	// perform parallel vector operation
	void performPVOperation();

	// parallel operator solution
	std::vector<Eigen::Vector3d> pvPointVec;

	// get the divergence from Jacobian
	void getDivergence();

	// divergence vector
	std::vector<double> divergenceVec;

	// get Jacobian information
	Eigen::Matrix3d getJacobian(const Eigen::Vector3d& pos);

	// calculate the Jacobian values on the grid points based on finite difference method
	void getJacobianOnGridPoints();


	// coordinates limit
	CoordinateLimits limits[3];

	// original data set name
	string dataset_name;

	// rectilinear grid information
	int X_RESOLUTION = -1, Y_RESOLUTION = -1, Z_RESOLUTION = -1;
	double X_STEP = -1.0, Y_STEP = -1.0, Z_STEP = 1.0;

private:

	/* parallel vector option, 1.eigenvector decomposition, 2.max eigenvector method */
	int pvMethodOption;

	std::vector<Vertex>& vertexVec;

	//store the Jacobian ahead of time
	std::vector<Eigen::Matrix3d> JacobianGridVec;

	// get solution point for 2D grid points
	void getSolutionFor2D(const int& x_index, const int& y_index, std::vector<Eigen::Vector3d>& solutionPoint);

	// get solution point for 3D grid points
	void getSolutionFor3D(const int& x_index, const int& y_index, const int& z_index,
			std::vector<Eigen::Vector3d>& solutionPoint);

	// check whether points are within range or not
	bool stayInDomain(const Eigen::Vector3d& point);

	/* ---------------------------------------------------------------------------------------------------------------
	 * -------------------------------------- Eigen-decomposition ----------------------------------------------------
	 * ---------------------------------------------------------------------------------------------------------------
	 */
	// get PV solution point within the triangle w.r.t. eigen-decomposition method
	void getEigenDecompositionSolution(const std::tuple<int,int,int>& triangleIndex,
			std::vector<Eigen::Vector3d>& solutionPoint);

	// get the velocity of the point w.r.t. pixel information
	Eigen::Vector3d getVelocityInVoxel(const Eigen::Vector3d& pos);

	// use eigen-vector to get the [s,t,1] solution
	std::vector<std::pair<double,double> > getParallelEigenVector(const Eigen::Matrix3d& matrix);

	/* ---------------------------------------------------------------------------------------------------------------
	 * -------------------------------------- Max eigen-vector method ------------------------------------------------
	 * ---------------------------------------------------------------------------------------------------------------
	 */
	// get PV solution point within the triangle w.r.t. max eigen-vector method
	void getMaxEigenvectorSolution(const std::tuple<int,int,int>& triangleIndex,
			std::vector<Eigen::Vector3d>& solutionPoint);

	/* ---------------------------------------------------------------------------------------------------------------
	 * -------------------------------------- with 2D Jacobian matrix ------------------------------------------------
	 * ---------------------------------------------------------------------------------------------------------------
	 */

};

#endif /* SRC_PARALLELVECTOR_PARALLELVECTOR_H_ */
