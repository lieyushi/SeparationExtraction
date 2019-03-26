/*
 * SeparationExtraction.cpp
 *
 *  Created on: Jan 9, 2019
 *      Author: lieyu
 */

#include "SeparationExtraction.h"

SeparationExtraction::SeparationExtraction() {
	// TODO Auto-generated constructor stub

}

SeparationExtraction::~SeparationExtraction() {
	// TODO Auto-generated destructor stub
}


/* extract separation lines for 2D streamlines */
void SeparationExtraction::extractSeparationLinesByExpansion(const std::vector<Eigen::VectorXd>& streamlineVector)
{
	const int& streamlineSize = streamlineVector.size();
	closestPair = std::vector<std::vector<std::pair<int,int> > >(streamlineSize,
			std::vector<std::pair<int,int> >(streamlineSize, std::make_pair(-1,-1)));
	distanceMatrix = Eigen::MatrixXd::Zero(streamlineSize, streamlineSize);	// closest point distance
	MCP_distance = Eigen::MatrixXd::Zero(streamlineSize, streamlineSize);

#pragma omp parallel for schedule(static) num_threads(8)
	for(int i=0; i<streamlineSize; ++i)
	{
		for(int j=0; j<streamlineSize; ++j)
		{
			if(i==j)
				continue;
			double mcp_dist;
			std::pair<int,int> closest;
			distanceMatrix(i,j) = SimilarityDistance::getClosestPair(streamlineVector[i], streamlineVector[j],
														             closest, mcp_dist);
			closestPair[i][j] = closest;
			MCP_distance(i,j) = mcp_dist;
		}
	}

	const int& max_heap_size = int(streamlineSize*SearchPercentage);

	// whether use pair-wise distance or discrete-curvature to measure expansion
	std::cout << "Whether use discrete curvatures for expansion measuring? 1.Yes, 2.No." << std::endl;
	int curvatureChoice;
	std::cin >> curvatureChoice;
	assert(curvatureChoice==1 || curvatureChoice==2);
	useCurvature = (curvatureChoice==1);

	// whether streamline i has been marked as separation. If yes then no need for computation
	std::vector<bool> separationFlag(streamlineSize, false);
	// whether separation checking has been performed on i and j. If yes, then no need for check again
	computationFlag = std::vector<std::vector<bool> >(streamlineSize, std::vector<bool>(streamlineSize, false));
	separationVector = std::vector<int>(streamlineSize, -1);

	std::vector<std::pair<int,int> > alreadyMarked;
	int separationNumber = 0;
	for(int i=0; i<streamlineSize-1; ++i)
	{
		if(!separationFlag[i])
		{
			std::priority_queue<DistRecord, std::vector<DistRecord>, CompareDistRecord> max_heap;
			for(int j=i+1; j<streamlineSize; ++j)
			{
				max_heap.push(DistRecord(distanceMatrix(i,j), j));
				if(max_heap.size()>max_heap_size)
					max_heap.pop();
			}

			// store the qualified candidates into a sorted list
			DistRecord top;
			std::vector<DistRecord> sorted_record;
			while(!max_heap.empty())
			{
				top = max_heap.top();
				//if(!separationFlag[top.index])
				sorted_record.push_back(top);
				max_heap.pop();
			}
			//  std::sort(sorted_record.begin(), sorted_record.end(), [](const DistRecord& d1, const DistRecord& d2)
			//	{	return d1.distance<d2.distance;	});
			std::reverse(sorted_record.begin(), sorted_record.end());

			int correspond = -1, another;
			double ratio = -1.0;
			for(int j=0; j<sorted_record.size(); ++j)
			{
				double temp_ratio = -1.0;
				another = sorted_record[j].index;
				if(!computationFlag[i][another] && !computationFlag[another][i]
						&& satisfySeparation(i, another, streamlineVector, temp_ratio))
				{
					// found, but already used for others
					if(!separationFlag[another])
					{
						correspond = another;
						ratio = temp_ratio;
						break;
					}
				}
			}

			if(correspond!=-1)
			{
				separationFlag[i] = true;
				separationFlag[correspond] = true;
				// already close to selected separation patterns
				if(!stayCloseToSelected(i, correspond, alreadyMarked))
				{
					std::cout << "Find separation pair: [" << i << "," << correspond << "]: " << ratio << std::endl;
					separationVector[i] = separationNumber;
					separationVector[correspond] = separationNumber;
					alreadyMarked.push_back(std::make_pair(i, correspond));
					++separationNumber;
				}
			}
		}
	}
}

// judge whether two given streamlines satisfy the separation criterion
bool SeparationExtraction::satisfySeparation(const int& firstIndex, const int& secondIndex,
		const std::vector<Eigen::VectorXd>& streamlineVector, double& ratio)
{
	computationFlag[firstIndex][secondIndex] = true;
	computationFlag[secondIndex][firstIndex] = true;

	const std::pair<int,int>& closest = closestPair[firstIndex][secondIndex];
	const int& closest_first = closest.first;
	const int& closest_second = closest.second;

	const Eigen::VectorXd& first_streamline = streamlineVector[firstIndex];
	const Eigen::VectorXd& second_streamline = streamlineVector[secondIndex];
	const int& first_size = first_streamline.size()/3;
	const int& second_size = second_streamline.size()/3;
	/*
	 * First criterion: both index should be on the middle
	 */
	if((closest_first<=2||closest_first>=first_size-3)||(closest_second<=2||closest_second>=second_size-3))
		return false;

	/*
	 * Second criterion: minimal distance should not be too small
	 */
	Eigen::Vector3d v1(first_streamline(3*closest_first), first_streamline(3*closest_first+1),
			first_streamline(3*closest_first+2));
	Eigen::Vector3d v2(second_streamline(3*closest_second), second_streamline(3*closest_second+1),
			second_streamline(3*closest_second+2));
	const double& ClosestDist = (v1-v2).norm();
	if(ClosestDist<SingularityDist || MCP_distance(firstIndex, secondIndex)<SingularityDist)
		return false;

	/*
	 * Third criterion: from forward and backward direction, the total length of monotonic inrement
	 * should be larger enough
	 */

	int forward_length = std::min(first_size-closest_first-1, second_size-closest_second-1);
	int backward_length = std::min(closest_first, closest_second);

	// use pair-wise distance to measure expansion
	if(!useCurvature)
		return ExpansionByDistance(first_streamline, second_streamline, closest_first, closest_second,
				forward_length, backward_length, ClosestDist, ratio);
	else
		return ExpansionByCurvature(first_streamline, second_streamline, closest_first, closest_second,
				forward_length, backward_length, ClosestDist, ratio);
}


// judge whether two streamlines are close to selected separations
bool SeparationExtraction::stayCloseToSelected(const int& first, const int& second,
									const std::vector<std::pair<int,int> >& selected)
{
	for(int i=0; i<selected.size(); ++i)
	{
		if(MCP_distance(first, selected[i].first)<CloseToSelected)
			return true;
		if(MCP_distance(first, selected[i].second)<CloseToSelected)
			return true;
		if(MCP_distance(second, selected[i].first)<CloseToSelected)
			return true;
		if(MCP_distance(second, selected[i].second)<CloseToSelected)
			return true;
	}
	return false;
}


// judge enough expansion by pair-wise distance of corresponding points
bool SeparationExtraction::ExpansionByDistance(const Eigen::VectorXd& first_streamline,
		const Eigen::VectorXd& second_streamline, const int& closest_first, const int& closest_second,
		const int& forward_length, const int& backward_length, const double& ClosestDist, double& ratio)
{
	// forward direction
	int forward_ratio = 0, x, y;
	Eigen::Vector3d v1, v2;
	double dist, minDist = ClosestDist;
	for(int i=1; i<=forward_length; i+=2)
	{
		x = closest_first+i;
		v1 = Eigen::Vector3d(first_streamline(3*x), first_streamline(3*x+1), first_streamline(3*x+2));
		y = closest_second+i;
		v2 = Eigen::Vector3d(second_streamline(3*y), second_streamline(3*y+1), second_streamline(3*y+2));
		dist = (v1-v2).norm();
		if(dist>BiggerRatio*minDist)
		{
			forward_ratio+=2;
			minDist = dist;
		}
		else
		{
			break;
		}
	}

	int backward_ratio = 0;
	minDist = ClosestDist;
	for(int i=1; i<=backward_length; i+=2)
	{
		x = closest_first-i;
		v1 = Eigen::Vector3d(first_streamline(3*x), first_streamline(3*x+1), first_streamline(3*x+2));
		y = closest_second-i;
		v2 = Eigen::Vector3d(second_streamline(3*y), second_streamline(3*y+1), second_streamline(3*y+2));
		dist = (v1-v2).norm();
		if(dist>BiggerRatio*minDist)
		{
			backward_ratio+=2;
			minDist = dist;
		}
		else
		{
			break;
		}
	}
	ratio = double(forward_ratio+backward_ratio)/double(backward_length+forward_length);
	if(/*forward_ratio>0 && backward_ratio>0 &&*/ ratio>=SeparationRatio)
		return true;
	else
		return false;
}


// judge enough expansion by discrete curvature measurement
/*
 * The experiment shows that expansion check by curvature didn't work as predicted
 */
bool SeparationExtraction::ExpansionByCurvature(const Eigen::VectorXd& first_streamline,
		const Eigen::VectorXd& second_streamline, const int& closest_first, const int& closest_second,
		const int& forward_length, const int& backward_length, const double& ClosestDist, double& ratio)
{
	// forward direction
	int forward_ratio = 0, x, y;
	Eigen::Vector3d v1, v2, u1, u2;
	// 0 for negative, 1 for positive
	bool isPositive[2] = {false, false};
	bool start = false;
	for(int i=1; i<=forward_length; ++i)
	{
		x = closest_first+i;
		v1 = Eigen::Vector3d(first_streamline(3*x)-first_streamline(3*x-3),
							 first_streamline(3*x+1)-first_streamline(3*x-2),
							 first_streamline(3*x+2)-first_streamline(3*x-1));
		v2 = Eigen::Vector3d(first_streamline(3*x+3)-first_streamline(3*x),
							 first_streamline(3*x+4)-first_streamline(3*x+1),
							 first_streamline(3*x+5)-first_streamline(3*x+2));
		v1 = v1.cross(v2);
		y = closest_second+i;
		u1 = Eigen::Vector3d(second_streamline(3*y)-second_streamline(3*y-3),
							 second_streamline(3*y+1)-second_streamline(3*y-2),
							 second_streamline(3*y+2)-second_streamline(3*y-1));
		u2 = Eigen::Vector3d(second_streamline(3*y+3)-second_streamline(3*y),
							 second_streamline(3*y+4)-second_streamline(3*y+1),
							 second_streamline(3*y+5)-second_streamline(3*y+2));
		u1 = u1.cross(u2);

		// first time, decide the starting line segments to be positive or negative
		if(!start)
		{
			isPositive[0]=(v1(2)>0);
			isPositive[1]=(u1(2)>0);
			start = true;
		}
		else if(start && ((v1(2)>0)==isPositive[0])&&((u1(2)>0)==isPositive[1]))
		{
			++forward_ratio;
		}
		else
			break;

	}

	int backward_ratio = 0;
	start = false;
	for(int i=1; i<=backward_length; ++i)
	{
		x = closest_first-i;
		v1 = Eigen::Vector3d(first_streamline(3*x)-first_streamline(3*x-3),
							 first_streamline(3*x+1)-first_streamline(3*x-2),
							 first_streamline(3*x+2)-first_streamline(3*x-1));
		v2 = Eigen::Vector3d(first_streamline(3*x+3)-first_streamline(3*x),
							 first_streamline(3*x+4)-first_streamline(3*x+1),
							 first_streamline(3*x+5)-first_streamline(3*x+2));
		v1 = v1.cross(v2);
		y = closest_second-i;
		u1 = Eigen::Vector3d(second_streamline(3*y)-second_streamline(3*y-3),
							 second_streamline(3*y+1)-second_streamline(3*y-2),
							 second_streamline(3*y+2)-second_streamline(3*y-1));
		u2 = Eigen::Vector3d(second_streamline(3*y+3)-second_streamline(3*y),
							 second_streamline(3*y+4)-second_streamline(3*y+1),
							 second_streamline(3*y+5)-second_streamline(3*y+2));
		u1 = u1.cross(u2);

		// first time, decide the starting line segments to be positive or negative
		if(!start)
		{
			isPositive[0]=(v1(2)>0);
			isPositive[1]=(u1(2)>0);
			start = true;
		}
		else if(start && ((v1(2)>0)==isPositive[0])&&((u1(2)>0)==isPositive[1]))
		{
			++backward_ratio;
		}
		else
			break;
	}
	ratio = double(forward_ratio+backward_ratio)/double(backward_length+forward_length);
	if(forward_ratio>0 && backward_ratio>0 && ratio>=SeparationRatio)
		return true;
	else
		return false;
}


/* extract separation lines with possibly largest chi-test-based distance of discrete curvatures */
void SeparationExtraction::extractSeparationLinesByChiTest(const std::vector<Eigen::VectorXd>& streamlineVector)
{
	const int& streamlineSize = streamlineVector.size();
	closestPair = std::vector<std::vector<std::pair<int,int> > >(streamlineSize,
			std::vector<std::pair<int,int> >(streamlineSize, std::make_pair(-1,-1)));
	distanceMatrix = Eigen::MatrixXd::Zero(streamlineSize, streamlineSize);
	MCP_distance = Eigen::MatrixXd::Zero(streamlineSize, streamlineSize);

#pragma omp parallel for schedule(static) num_threads(8)
	for(int i=0; i<streamlineSize; ++i)
	{
		for(int j=0; j<streamlineSize; ++j)
		{
			if(i==j)
				continue;
			double mcp_dist;
			std::pair<int,int> closest;
			distanceMatrix(i,j) = SimilarityDistance::getClosestPair(streamlineVector[i],
					streamlineVector[j], closest, mcp_dist);
			closestPair[i][j] = closest;
			MCP_distance(i,j) = mcp_dist;
		}
	}

	std::cout << "Use curvatures on segments or purely individual vertex? 1. Segments, 2. vertex. " << std::endl;
	int segmentInput;
	std::cin >> segmentInput;
	assert(segmentInput==1 || segmentInput==2);
	bool useSegments = (segmentInput==1);

	std::cout << "Set percentage for MCP: " << std::endl;
	double mcp_percentage;
	std::cin >> mcp_percentage;
	assert(mcp_percentage>=0 && mcp_percentage<1.0);

	std::cout << "Set percentage for closest point distance: " << std::endl;
	double closest_percentage;
	std::cin >> closest_percentage;
	assert(closest_percentage>=0 && closest_percentage<1.0);

	Eigen::MatrixXd chi_test_distance = Eigen::MatrixXd::Zero(streamlineSize, streamlineSize);

	std::cout << "Input how many pairs of separation lines are wanted? " << std::endl;
	int pairsWanted;
	std::cin >> pairsWanted;
	assert(pairsWanted>0 && pairsWanted<streamlineSize);

	/* all normalized to [0, 1]. Closest distance, MCP distance and chi-test distance */
	double distRange[3][3] =
	{
			{DBL_MAX, -DBL_MAX, -1},
			{DBL_MAX, -DBL_MAX, -1},
			{DBL_MAX, -DBL_MAX, -1}
	};
//#pragma omp parallel for schedule(static) num_threads(8)
	std::priority_queue<distPair, std::vector<distPair>, distPairCompare> distPairPQueue;
	for(int i=0; i<streamlineSize-1; ++i)
	{
		for(int j=i+1; j<streamlineSize; ++j)
		{
			double chi_test_value = SimilarityDistance::getChiTestOfCurvatures(streamlineVector[i],
					streamlineVector[j], closestPair[i][j], useSegments);
			chi_test_distance(i,j) = chi_test_value;
			chi_test_distance(j,i) = chi_test_value;
			/* get the min and max of closest point distance */
			distRange[0][0] = std::min(distRange[0][0], distanceMatrix(i,j));
			distRange[0][1] = std::max(distRange[0][1], distanceMatrix(i,j));

			/* get the min and max of MCP distance */
			distRange[1][0] = std::min(distRange[1][0], MCP_distance(i,j));
			distRange[1][1] = std::max(distRange[1][1], MCP_distance(i,j));

			/* get the min and max of chi-test distance */
			distRange[2][0] = std::min(distRange[2][0], chi_test_value);
			distRange[2][1] = std::max(distRange[2][1], chi_test_value);
		}
	}

	for(int i=0; i<3; ++i)
		distRange[i][2] = distRange[i][1]-distRange[i][0];

	/* normalization for the distance and get the linearly combined values */
	for(int i=0; i<streamlineSize-1; ++i)
	{
		for(int j=i+1; j<streamlineSize; ++j)
		{
			distPairPQueue.push(distPair(i,j,(1.0-mcp_percentage-closest_percentage)*
					(chi_test_distance(i,j)-distRange[2][0])/distRange[2][2]
					+mcp_percentage*(MCP_distance(i,j)-distRange[1][0])/distRange[1][2]
					-closest_percentage*(distanceMatrix(i,j)-distRange[1][0])/distRange[0][2]));
		}
	}

	/* just in case of collision of separation values assigend to streamlines */
	std::vector<bool> isSelected(streamlineSize, false);

	separationVector = std::vector<int>(streamlineSize, -1);
	std::vector<distPair> distPairVec;
	distPair dTop;
	while(!distPairPQueue.empty() && distPairVec.size()<pairsWanted)
	{
		dTop = distPairPQueue.top();
		if(!isSelected[dTop.first] && !isSelected[dTop.second])
		{
			distPairVec.push_back(dTop);
			isSelected[dTop.first] = true;
			isSelected[dTop.second] = true;
		}
		distPairPQueue.pop();
	}
	std::reverse(distPairVec.begin(), distPairVec.end());

	/* assign separation values */
	int separationValue = 0;
	for(auto &d:distPairVec)
	{
		separationVector[d.first] = separationValue;
		separationVector[d.second] = separationValue;
		std::cout << d.first << " " << d.second << " " << d.distance << std::endl;
		++separationValue;
	}

	std::cout << chi_test_distance(0,12) << " " << MCP_distance(0,12) << " " << distanceMatrix(0,12) << std::endl;
}


/* update the separation ratio parameter */
void SeparationExtraction::setSeparationRatio(const double& separationValue)
{
	assert(separationValue>=0 && separationValue<=1);
	SeparationRatio = separationValue;
}


/* update the search percentage for minimal class number */
void SeparationExtraction::setSearchPercentage(const double& percentage)
{
	assert(percentage>=0 && percentage<=1);
	SearchPercentage = percentage;
}


// judge enough expansion by discrete curvature of segments
void SeparationExtraction::extractSeparationLinesBySegments(const std::vector<Eigen::VectorXd>& streamlineVector)
{
	const int& streamlineSize = streamlineVector.size();

	/*
	 * each streamline might be decomposed into log2(vertexCount) segments and we would like to segment it base
	 * on vertices w.r.t. maximal discrete curvatures
	 */
	std::vector<Eigen::VectorXd> lineSegments, lineCurvatures;
	std::vector<std::vector<int> > segmentOfStreamlines(streamlineSize);

	int numberOfSegments = 0;
	for(int i = 0; i<streamlineSize; ++i)
	{
		numberOfSegments += int(log2(streamlineVector.at(i).size()/3))+1;
	}

	lineSegments = std::vector<Eigen::VectorXd>(numberOfSegments);
	lineCurvatures = std::vector<Eigen::VectorXd>(numberOfSegments);

	/*
	 * make segmentation based on signature for all streamlines. Similar to signature-based sampling
	 */
	int accumulated = 0;
	SimilarityDistance::computeSegments(lineSegments, lineCurvatures, segmentOfStreamlines,
					accumulated, streamlineVector);

	/*
	 * compute distance values among all streamline segment pairs
	 */
	distanceMatrix = Eigen::MatrixXd::Zero(numberOfSegments, numberOfSegments);
	MCP_distance = Eigen::MatrixXd::Zero(numberOfSegments, numberOfSegments);
	Eigen::MatrixXd chi_distanceMatrix = Eigen::MatrixXd::Zero(numberOfSegments, numberOfSegments);
	closestPair = std::vector<std::vector<std::pair<int,int> > >(numberOfSegments,
				std::vector<std::pair<int,int> >(numberOfSegments, std::make_pair(-1,-1)));

	/*
	 * compute MCP, closest-point-distance and curvature-based Chi-test distance for all streamline segment pairs
	 */
#pragma omp parallel for schedule(static) num_threads(8)
	for(int i=0; i<numberOfSegments; ++i)
	{
		for(int j=0; j<numberOfSegments; ++j)
		{
			if(i==j)
				continue;
			double mcp_dist;
			std::pair<int,int> closest;
			distanceMatrix(i,j) = SimilarityDistance::getClosestPair(lineSegments[i],
					lineSegments[j], closest, mcp_dist);
			closestPair[i][j] = closest;
			chi_distanceMatrix(i,j) = SimilarityDistance::getChiTestPair(lineCurvatures[i], lineCurvatures[j]);
			MCP_distance(i,j) = mcp_dist;
		}
	}

	/*
	 * what to do with all the dissimilarity measures among them?
	 * Two options might be practical: 1.use linear combinations to get maximal,
	 * 2.set the minimal percentage of MCP distance. Maximal of maximal chi-test
	 */

	std::cout << "Input a post-processing method, 1.Maximal of maximal curvature chi-test, "
			"2.linear combination of MCP and chi-test of curvatures, "
			"3.maximal point-distance expansion..." << std::endl;
	int postProcessing;
	std::cin >> postProcessing;
	assert(postProcessing==1 || postProcessing==2);

	std::cout << "Choose how many maximal pairs need to be selected between (0," << streamlineSize << ")."
			  << std::endl;
	int maximalPairs;
	std::cin >> maximalPairs;
	assert(maximalPairs>0 && maximalPairs<streamlineSize);

	if(postProcessing==1)
	{
		/*
		 * Input a threshold such that I would only choose threshold*numberOfSegments smallest MCP distance
		 * to judge the maximal
		 */
		std::cout << "Input a threshold between [0,1.0] to find smallest MCP distance pair..." << std::endl;
		double MCP_threshold;
		std::cin >> MCP_threshold;
		assert(0.0 <= MCP_threshold && MCP_threshold <= 1.0);

		/*
		 * For example, get 0.05% of the neighbor, and input the pair with maximal curvatures.
		 * Once get all curvatures pairs, should find the maximal numbers of pairs again.
		 * So-called maximal-of-maximal-of-minimal detection algorithm
		 */
		getMaxCurvatureWithMimimalMCP(lineSegments, lineCurvatures, segmentOfStreamlines, chi_distanceMatrix,
				MCP_threshold, maximalPairs);
	}

	else if(postProcessing==2)
	{
		/*
		 * Input the linear combination letter of curvature + MCP + closest_point_dist
		 * and find the several maximal pairs as highlighted
		 */
		std::cout << "Input the ratio of MCP in filtering between [0,1.0]." << std::endl;
		double MCP_percentage;
		std::cin >> MCP_percentage;
		assert(MCP_percentage>=0.1 && MCP_percentage<=1.0);

		std::cout << "Input the ratio of closest point distance in filtering between [0,1.0]." << std::endl;
		double cpd_percentage;
		std::cin >> cpd_percentage;
		assert(cpd_percentage>=0.1 && cpd_percentage<=1.0);

		getMaxCombinedDistance(lineSegments, lineCurvatures, segmentOfStreamlines, chi_distanceMatrix,
				std::make_pair(MCP_percentage, cpd_percentage), maximalPairs);
	}
	else if(postProcessing==3)
	{
		/*
		 * set the expansion ratio and search among the neighboring segments starting from minimal MCP.
		 * For segments, MCP and closest point distance didn't make too much difference, so only consider MCP
		 * should be totally enough
		 */
		std::cout << "Input the ratio of expansion point percentage between [0, 1.0]... " << std::endl;
		double expansionRatio;
		std::cin >> expansionRatio;
		assert(expansionRatio>0 && expansionRatio<1.0);

		std::cout << "Input threshold ratio to say two lines are expanding between [0, 1.0]... " << std::endl;
		double ratioThreshold;
		std::cin >> ratioThreshold;
		assert(ratioThreshold>0 && ratioThreshold<=1.0);
		getMeasurementByPointExpansion(lineSegments, segmentOfStreamlines, maximalPairs,
				expansionRatio, ratioThreshold);
	}
}


// with given parameter, we can get the maximal-of-maximal-of-minimal method
void SeparationExtraction::getMaxCurvatureWithMimimalMCP(const std::vector<Eigen::VectorXd>& lineSegments,
		const std::vector<Eigen::VectorXd>& lineCurvatures,
		const std::vector<std::vector<int> >& segmentOfStreamlines,
		const Eigen::MatrixXd& chi_distanceMatrix, const double& MCP_threshold, const int& maximalPairs)
{
	const int& numberOfSegments = lineSegments.size();
	const int& MCP_minimal_number = int(MCP_threshold*numberOfSegments);

	DistRecord top;
	double max_chi_for_i;
	int targetSegment, targetSegmentStreamlineID;
	std::priority_queue<distPair, std::vector<distPair>, distPairCompare> distPairPQueue;
	for(int i=0; i<numberOfSegments; ++i)
	{
		// get for example, 0.05 minimal segment pairs with minimal MCP distance
		std::priority_queue<DistRecord, std::vector<DistRecord>, CompareDistRecord> max_mcp_heap;

		// exclude segments on the same streamline
		int fID, sID;
		fID = SimilarityDistance::getStreamlineID(i,segmentOfStreamlines);
		for(int j=0; j<numberOfSegments; ++j)
		{
			if(j!=i)
			{
				sID = SimilarityDistance::getStreamlineID(j,segmentOfStreamlines);

				if(fID!=sID)
				{
					max_mcp_heap.push(DistRecord(MCP_distance(i,j), j, sID));
					if(max_mcp_heap.size()>MCP_minimal_number)
					{
						max_mcp_heap.pop();
					}
				}
			}
		}

		// find the maximal chi-test example inside that are also not on one streamline
		max_chi_for_i = -DBL_MAX;
		while(!max_mcp_heap.empty())
		{
			top = max_mcp_heap.top();
			if(max_chi_for_i<chi_distanceMatrix(i,top.index))
			{
				max_chi_for_i = chi_distanceMatrix(i,top.index);
				targetSegment = top.index;
				targetSegmentStreamlineID = top.streamlineID;
			}
			max_mcp_heap.pop();
		}
		distPairPQueue.push(distPair(i,targetSegment,fID,targetSegmentStreamlineID,max_chi_for_i));
	}

	// get MCP_minimal_number pairs of maximal curvature-based chi-test distance value
	separationVector = std::vector<int>(segmentOfStreamlines.size(), -1);
	// a boolean vector to judge whether some streamline has been selected or not
	std::vector<bool> hasBeenChosen(segmentOfStreamlines.size(), false);
	int count = 0;

	distPair dpTop;
	while(count<maximalPairs)
	{
		dpTop = distPairPQueue.top();
		if(!hasBeenChosen[dpTop.firstID] && !hasBeenChosen[dpTop.secondID])
		{
			separationVector[dpTop.firstID] = separationVector[dpTop.secondID] = count;
			hasBeenChosen[dpTop.firstID] = true;
			hasBeenChosen[dpTop.secondID] = true;
			++count;
		}
		distPairPQueue.pop();
	}
}

// linear combination of chi-test curvature, MCP and closest point distance
void SeparationExtraction::getMaxCombinedDistance(const std::vector<Eigen::VectorXd>& lineSegments,
		const std::vector<Eigen::VectorXd>& lineCurvatures,
		const std::vector<std::vector<int> >& segmentOfStreamlines,
		const Eigen::MatrixXd& chi_distanceMatrix, const std::pair<double, double>& parameterPair,
		const int& maximalPairs)
{
	const int& numberOfSegments = lineSegments.size();
	/* all normalized to [0, 1]. Closest distance, MCP distance and chi-test distance */
	double distRange[3][3] =
	{
		{DBL_MAX, -DBL_MAX, -1},
		{DBL_MAX, -DBL_MAX, -1},
		{DBL_MAX, -DBL_MAX, -1}
	};

	for(int i=0; i<numberOfSegments-1; ++i)
	{
		for(int j=i+1; j<numberOfSegments; ++j)
		{
			/* update the min and max for closest point distance */
			distRange[0][0] = std::min(distRange[0][0], distanceMatrix(i,j));
			distRange[0][1] = std::max(distRange[0][1], distanceMatrix(i,j));

			/* update the min and max for MCP distance */
			distRange[1][0] = std::min(distRange[1][0], MCP_distance(i,j));
			distRange[1][1] = std::max(distRange[1][1], MCP_distance(i,j));

			/* update the min and max for chi_test distance */
			distRange[2][0] = std::min(distRange[2][0], chi_distanceMatrix(i,j));
			distRange[2][1] = std::max(distRange[2][1], chi_distanceMatrix(i,j));
		}
	}

	for(int i=0; i<3; ++i)
		distRange[i][2] = distRange[i][1]-distRange[i][0];

	/* push all linearly-combined distance of segment pairs into max heap */
	std::priority_queue<distPair, std::vector<distPair>, distPairCompare> distPairPQueue;

	int fID, sID;
	for(int i=0; i<numberOfSegments-1; ++i)
	{
		fID = SimilarityDistance::getStreamlineID(i,segmentOfStreamlines);
		for(int j=i+1; j<numberOfSegments; ++j)
		{
			sID = SimilarityDistance::getStreamlineID(j,segmentOfStreamlines);
			if(fID!=sID)
			{
				distPairPQueue.push(distPair(i,j,fID,sID,
						(1.0-parameterPair.first-parameterPair.second)*
						(chi_distanceMatrix(i,j)-distRange[2][0])/distRange[2][2]
						-parameterPair.first*(MCP_distance(i,j)-distRange[1][0])/distRange[1][2]
						-parameterPair.second*(distanceMatrix(i,j)-distRange[0][0])/distRange[0][2]));
			}
		}
	}

	/* select given number of pairs as highlighted */
	separationVector = std::vector<int>(segmentOfStreamlines.size(), -1);
	// a boolean vector to judge whether some streamline has been selected or not
	std::vector<bool> hasBeenChosen(segmentOfStreamlines.size(), false);
	int count = 0;
	distPair top;
	while(!distPairPQueue.empty() && count<maximalPairs)
	{
		top = distPairPQueue.top();
		if(!hasBeenChosen[top.firstID] && !hasBeenChosen[top.secondID])
		{
			separationVector[top.firstID] = separationVector[top.secondID] = count;
			hasBeenChosen[top.firstID] = true;
			hasBeenChosen[top.secondID] = true;
			++count;
		}
		distPairPQueue.pop();
	}
}


// judge whether partial of the segments satisfy the point expansion criterion
void SeparationExtraction::getMeasurementByPointExpansion(const std::vector<Eigen::VectorXd>& lineSegments,
		const std::vector<std::vector<int> >& segmentOfStreamlines, const int& maximalPairs,
		const double& expansionRatio, const double& ratioThreshold)
{
	const int& numberOfSegments = lineSegments.size();
	const int& neighborSize = int(expansionRatio*numberOfSegments);

	std::vector<bool> isChecked(segmentOfStreamlines.size(), false);
	/* select given number of pairs as highlighted */
	separationVector = std::vector<int>(segmentOfStreamlines.size(), -1);

	std::vector<int> segmentsOfOneLine, secondLine;
	int count = 0;
	for(int i=0; i<segmentOfStreamlines.size()-1; ++i)
	{
		segmentsOfOneLine = segmentOfStreamlines[i];
		for(int j=0; j<segmentsOfOneLine.size() && !isChecked[i]; ++j)
		{
			std::priority_queue<distPair, std::vector<distPair>, distPairCompare> distPairPQueue;
			for(int k=i+1; k<segmentOfStreamlines.size() && !isChecked[i]; ++k)
			{
				if(!isChecked[k])
				{
					secondLine = segmentOfStreamlines[k];
					for(int l=0; l<secondLine.size(); ++l)
					{
						distPairPQueue.push(distPair(segmentsOfOneLine[j],secondLine[l],
								MCP_distance(segmentsOfOneLine[j],secondLine[l])));
						if(distPairPQueue.size()>neighborSize)
							distPairPQueue.pop();
					}

					std::vector<int> candidateVec;
					while(!distPairPQueue.empty())
					{
						candidateVec.push_back(distPairPQueue.top().second);
						distPairPQueue.pop();
					}
					std::reverse(candidateVec.begin(), candidateVec.end());

					for(int l=0; l<candidateVec.size() && !isChecked[k]; ++l)
					{
						auto closest = closestPair[segmentsOfOneLine[j]][candidateVec[l]];
						int left = closest.first, right = closest.second;

						Eigen::VectorXd first = lineSegments[segmentsOfOneLine[j]];
						Eigen::VectorXd second = lineSegments[candidateVec[l]];

						int first_size = first.size()/3, second_size = second.size()/3;

						double closestPDistance = distanceMatrix(segmentsOfOneLine[j], candidateVec[l]);

						int forward_length = std::min(first_size-left-1, second_size-right-1);
						int backward_length = std::min(left, right);

						// judge enough expansion by pair-wise distance of corresponding points
						if(measureSegmentExpansion(first,second,left,right,forward_length, backward_length,
								closestPDistance,ratioThreshold) && !isChecked[j] && !isChecked[k])
						{
							isChecked[j] = isChecked[k] = true;
							separationVector[j] = separationVector[k] = count;
							++count;
						}
					}
				}
			}
		}
	}

	std::cout << "Generated " << count << " pairs." << std::endl;
}


// judge enough expansion by pair-wise distance of corresponding points
bool SeparationExtraction::measureSegmentExpansion(const Eigen::VectorXd& first_streamline,
		const Eigen::VectorXd& second_streamline, const int& closest_first, const int& closest_second,
		const int& forward_length, const int& backward_length, const double& ClosestDist, const double& ratioThreshold)
{
	// forward direction
	int forward_ratio = 0, x, y;
	Eigen::Vector3d v1, v2;
	double dist, minDist = ClosestDist;
	for(int i=1; i<=forward_length; i+=1)
	{
		x = closest_first+i;
		v1 = Eigen::Vector3d(first_streamline(3*x), first_streamline(3*x+1), first_streamline(3*x+2));
		y = closest_second+i;
		v2 = Eigen::Vector3d(second_streamline(3*y), second_streamline(3*y+1), second_streamline(3*y+2));
		dist = (v1-v2).norm();
		if(dist>=BiggerRatio*minDist)
		{
			forward_ratio+=1;
			minDist = dist;
		}
		else
			break;
	}

	int backward_ratio = 0;
	minDist = ClosestDist;
	for(int i=1; i<=backward_length; i+=1)
	{
		x = closest_first-i;
		v1 = Eigen::Vector3d(first_streamline(3*x), first_streamline(3*x+1), first_streamline(3*x+2));
		y = closest_second-i;
		v2 = Eigen::Vector3d(second_streamline(3*y), second_streamline(3*y+1), second_streamline(3*y+2));
		dist = (v1-v2).norm();
		if(dist>=BiggerRatio*minDist)
		{
			backward_ratio+=1;
			minDist = dist;
		}
		else
		{
			break;
		}
	}
	double ratio = double(forward_ratio+backward_ratio)/double(backward_length+forward_length);
	if(forward_ratio>0 && backward_ratio>0 && ratio>=ratioThreshold)
		return true;
	else
		return false;
}


