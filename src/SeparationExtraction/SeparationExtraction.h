/*
 * SeparationExtraction.h
 *
 *  Created on: Jan 9, 2019
 *      Author: lieyu
 */

#ifndef SRC_SEPARATIONEXTRACTION_SEPARATIONEXTRACTION_H_
#define SRC_SEPARATIONEXTRACTION_SEPARATIONEXTRACTION_H_

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
#include "SimilarityDistance.h"

using namespace std;
using namespace Eigen;

#define SingularityDist 1.0E-3
#define BiggerRatio 1.02
#define CloseToSelected 5.0E-3

// a struct object to record k-smallest distance w.r.t. a given reference streamline
struct DistRecord
{
	double distance;
	int index, streamlineID;

	DistRecord(): distance(-1.0), index(-1), streamlineID(-1)
	{}

	DistRecord(const double& dist, const int& index): distance(dist), index(index), streamlineID(-1)
	{}

	DistRecord(const double& dist, const int& index, const int& fID): distance(dist), index(index), streamlineID(fID)
	{}
};

// a compare struct for min heap to find k-smallest distance
struct CompareDistRecord
{
	bool operator()(const DistRecord& d1, const DistRecord& d2)
	{
		return d1.distance<d2.distance;
	}
};


class SeparationExtraction {
public:
	SeparationExtraction();
	virtual ~SeparationExtraction();

	/* extract separation lines with expansion by point-based distance or discrete curvature */
	void extractSeparationLinesByExpansion(const std::vector<Eigen::VectorXd>& streamlineVector);

	/* extract separation lines with possibly largest chi-test-based distance of discrete curvatures */
	void extractSeparationLinesByChiTest(const std::vector<Eigen::VectorXd>& streamlineVector);

	// judge enough expansion by discrete curvature of segments
	void extractSeparationLinesBySegments(const std::vector<Eigen::VectorXd>& streamlineVector);

	/* update the separation ratio parameter */
	void setSeparationRatio(const double& separationValue);

	/* update the search percentage for minimal class number */
	void setSearchPercentage(const double& percentage);

	/* flag for marking the index of separation */
	std::vector<int> separationVector;

	/* whether use pair-wise distance or discrete curvatures */
	bool useCurvature;

private:
	// closest distance point
	Eigen::MatrixXd distanceMatrix;

	// MCP distance matrix
	Eigen::MatrixXd MCP_distance;

	// closest point pair
	std::vector<std::vector<std::pair<int,int> > > closestPair;

	// check whether separation checking has been performed between two streamlines
	std::vector<std::vector<bool> > computationFlag;

	// separation ratio
	double SeparationRatio = 0.3;

	// search percentage of the total number
	double SearchPercentage = 0.5;

	// judge whether two given streamlines satisfy the separation criterion
	bool satisfySeparation(const int& firstIndex, const int& secondIndex,
			const std::vector<Eigen::VectorXd>& streamlineVector, double& ratio);

	// judge whether two streamlines are close to selected separations
	bool stayCloseToSelected(const int& first, const int& second, const std::vector<std::pair<int,int> >& selected);

	// judge enough expansion by pair-wise distance of corresponding points
	bool ExpansionByDistance(const Eigen::VectorXd& first_streamline, const Eigen::VectorXd& second_streamline,
			const int& closest_first, const int& closest_second, const int& forward_length,
			const int& backward_length, const double& ClosestDist, double& ratio);

	// judge enough expansion by discrete curvature measurement
	bool ExpansionByCurvature(const Eigen::VectorXd& first_streamline,
			const Eigen::VectorXd& second_streamline, const int& closest_first, const int& closest_second,
			const int& forward_length, const int& backward_length, const double& ClosestDist, double& ratio);

	// with given parameter, we can get the maximal-of-maximal-of-minimal method
	void getMaxCurvatureWithMimimalMCP(const std::vector<Eigen::VectorXd>& lineSegments,
			const std::vector<Eigen::VectorXd>& lineCurvatures,
			const std::vector<std::vector<int> >& segmentOfStreamlines,
			const Eigen::MatrixXd& chi_distanceMatrix, const double& MCP_threshold, const int& maximalPairs);

	// linear combination of chi-test curvature, MCP and closest point distance
	void getMaxCombinedDistance(const std::vector<Eigen::VectorXd>& lineSegments,
			const std::vector<Eigen::VectorXd>& lineCurvatures,
			const std::vector<std::vector<int> >& segmentOfStreamlines,
			const Eigen::MatrixXd& chi_distanceMatrix, const std::pair<double, double>& parameterPair,
			const int& maximalPairs);

	// judge whether partial of the segments satisfy the point expansion criterion
	void getMeasurementByPointExpansion(const std::vector<Eigen::VectorXd>& lineSegments,
			const std::vector<std::vector<int> >& segmentOfStreamlines, const int& maximalPairs,
			const double& expansionRatio, const double& ratioThreshold);

	// judge enough expansion by pair-wise distance of corresponding points
	bool measureSegmentExpansion(const Eigen::VectorXd& first_streamline,
			const Eigen::VectorXd& second_streamline, const int& closest_first, const int& closest_second,
			const int& forward_length, const int& backward_length, const double& ClosestDist,
			const double& ratioThreshold);

	/*
	 * A composite class to assimilate max heap.
	 */
	struct distPair
	{
		int first, second, firstID, secondID;
		double distance;
		distPair(const int& first, const int& second, const double& distance): first(first),
				second(second), firstID(-1), secondID(-1), distance(distance)
		{
		}

		distPair(const int& first, const int& second, const int& fID, const int& sID, const double& distance):
			first(first), second(second), firstID(fID), secondID(sID), distance(distance)
		{
		}

		distPair(): first(-1), second(-1), firstID(-1), secondID(-1), distance(-1.0)
		{
		}
	};

	/*
	 * A object compare class function
	 */
	struct distPairCompare
	{
		bool operator()(const distPair& d1, const distPair& d2)
		{
			return d1.distance<d2.distance;
		}
	};

};

#endif /* SRC_SEPARATIONEXTRACTION_SEPARATIONEXTRACTION_H_ */
