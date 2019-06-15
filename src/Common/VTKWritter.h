/*
 * VTKWritter.h
 *
 *  Created on: Mar 12, 2019
 *      Author: lieyu
 */

#ifndef SRC_COMMON_VTKWRITTER_H_
#define SRC_COMMON_VTKWRITTER_H_

#include <iostream>
#include <vector>
#include <math.h>
#include <fstream>
#include <stdlib.h>
#include <float.h>
#include <sstream>
#include <time.h>
#include <sstream>
#include <algorithm>
#include <cassert>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SVD>
#include "DataSet.h"

using namespace std;
using namespace Eigen;

class VTKWritter {
public:
	/* print streamlines as segment */
	static void printStreamlineSegments(const std::vector<Eigen::VectorXd>& coordinates, const string& datasetName,
			const std::vector<int>& pointToSegment, const int& streamlineVertexCount);

	/* print streamlines with scalars on segments */
	static void printStreamlineScalarsOnSegments(const std::vector<Eigen::VectorXd>& coordinates,
			const string& datasetName, const int& streamlineVertexCount, const std::vector<Eigen::VectorXd>& lineSegments,
			const std::vector<double>& segmentScalars);

	/* print streamlines with scalars on segments */
	static void printStreamlineScalarsOnSegments(const std::vector<Eigen::VectorXd>& coordinates,
			const string& datasetName, const int& streamlineVertexCount, const std::vector<double>& segmentScalars);

	/* print 3d vector field with regular grid */
	static void printVectorField(const string& fileName, const std::vector<Vertex>& vertexVec,
			CoordinateLimits limits[3], const int& x_resolution, const int& y_resolution, const int& z_resolution,
			const double& x_step, const double& y_step, const double& z_step);

	/* print 3d point cloud */
	static void printPoints(const string& fileName, const std::vector<Eigen::Vector3d>& pointArray);

};

#endif /* SRC_COMMON_VTKWRITTER_H_ */
