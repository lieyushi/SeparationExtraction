/*
 * SimilarityDistance.h
 *
 *  Created on: Feb 4, 2019
 *      Author: lieyu
 */

#ifndef SRC_COMMON_SIMILARITYDISTANCE_H_
#define SRC_COMMON_SIMILARITYDISTANCE_H_

#include <iostream>
#include <vector>
#include <math.h>
#include <fstream>
#include <stdlib.h>
#include <float.h>
#include <time.h>
#include <sstream>
#include <algorithm>
#include <queue>
#include <cassert>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SVD>

using namespace std;
using namespace Eigen;

/* curvature with index */
struct CurvatureObject
{
	double curvature;
	int index;

	CurvatureObject(const double& curvature, const int& i): curvature(curvature), index(i)
	{}

	CurvatureObject()
	{}
};

/* compare function */
class CompareFunc
{
public:
	bool operator()(const CurvatureObject& first, const CurvatureObject& second)
	{
		return abs(first.curvature) > abs(second.curvature);
	}
};


class SimilarityDistance {
public:
	SimilarityDistance();
	virtual ~SimilarityDistance();

	/* a point-based distance for computing mcp and minimal distance */
	static const double getClosestPair(const Eigen::VectorXd& first, const Eigen::VectorXd& second,
				                       std::pair<int,int>& closest, double& mcp_dist);

	/* a chi-test for discrete curvatures with each attribute on each vertex */
	static const double getChiTestOfCurvatures(const Eigen::VectorXd& first, const Eigen::VectorXd& second,
				const std::pair<int,int>& closest, const bool& useSegments);

private:
	/* a chi-test with two equal-size attribute arrays */
	static const double getChiTest(const Eigen::VectorXd& first, const Eigen::VectorXd& second);

	/* chi-test with two attribute arrays that have discrete curvatures over each vertex */
	static const double getChiTestOfCurvaturesOnVertex(const Eigen::VectorXd& first, const Eigen::VectorXd& second,
			     const int& secondHalf, const std::pair<int,int>& closest, const string& sign);

	/* chi-test with two attribute arrays that have discrete curvatures over segments */
	static const double getChiTestOfCurvaturesOnSegments(const Eigen::VectorXd& first,
			const Eigen::VectorXd& second, const int& firstHalf, const std::pair<int,int>& closest, const string& sign);
};

#endif /* SRC_COMMON_SIMILARITYDISTANCE_H_ */
