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

};

#endif /* SRC_COMMON_VTKWRITTER_H_ */
