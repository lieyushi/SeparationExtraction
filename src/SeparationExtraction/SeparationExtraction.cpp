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
			distanceMatrix(i,j) = SimilarityDistance::getClosestPair(streamlineVector[i], streamlineVector[j],
														             closest, mcp_dist);
			closestPair[i][j] = closest;
			MCP_distance(i,j) = mcp_dist;
		}
	}

	// find k for k-th smallest distance so that distance will be searched along [0,k]
	std::cout << "Input a percentage to determine k-th smallest search among (0, 1.0). " << std::endl;
	double percentage;
	std::cin >> percentage;
	assert(percentage>0.0 && percentage<1.0);
	const int& max_heap_size = int(streamlineSize*percentage);

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
	for(int i=0; i<streamlineSize; ++i)
	{
		if(!separationFlag[i])
		{
			std::priority_queue<DistRecord, std::vector<DistRecord>, CompareDistRecord> max_heap;
			for(int j=0; j<streamlineSize; ++j)
			{
				if(i==j)
					continue;
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
			double ratio = -1;
			for(int j=0; j<sorted_record.size(); ++j)
			{
				another = sorted_record[j].index;
				if(!computationFlag[i][another] && !computationFlag[another][i]
						&& satisfySeparation(i, another, streamlineVector, ratio))
				{
					// found, but already used for others
					if(!separationFlag[another])
					{
						correspond = another;
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
	for(int i=1; i<forward_length; i+=2)
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
			break;
	}

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
			forward_ratio+=2;
			minDist = dist;
		}
		else
		{
			break;
		}
	}
	ratio = double(forward_ratio)/double(backward_length+forward_length);
	if(ratio>=SeparationRatio)
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
	for(int i=1; i<forward_length; ++i)
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
			++forward_ratio;
		}
		else
			break;
	}
	ratio = double(forward_ratio)/double(backward_length+forward_length);
	if(ratio>=SeparationRatio)
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

//#pragma omp parallel for schedule(static) num_threads(8)
	std::priority_queue<distPair, std::vector<distPair>, distPairCompare> distPairPQueue;
	for(int i=0; i<streamlineSize; ++i)
	{
		for(int j=0; j<streamlineSize; ++j)
		{
			if(i==j)
				continue;
			double chi_test_value = SimilarityDistance::getChiTestOfCurvatures(streamlineVector[i],
					streamlineVector[j], closestPair[i][j], useSegments);
			chi_test_distance(i,j) = (1.0-mcp_percentage-closest_percentage)*chi_test_value
					+mcp_percentage*MCP_distance(i,j)-closest_percentage*distanceMatrix(i,j);
			distPairPQueue.push(distPair(i,j,chi_test_distance(i,j)));
			if(distPairPQueue.size()>pairsWanted)
				distPairPQueue.pop();
		}
	}
	separationVector = std::vector<int>(streamlineSize, -1);
	std::vector<distPair> distPairVec;
	while(!distPairPQueue.empty())
	{
		distPairVec.push_back(distPairPQueue.top());
		distPairPQueue.pop();
	}
	std::reverse(distPairVec.begin(), distPairVec.end());

	/* assign separation values */
	int separationValue = 0;
	for(auto &d:distPairVec)
	{
		separationVector[d.first] = separationValue;
		separationVector[d.second] = separationValue;
		++separationValue;
	}
}

