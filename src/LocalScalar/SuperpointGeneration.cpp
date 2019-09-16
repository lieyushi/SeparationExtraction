/*
 * SuperpointGeneration.cpp
 *
 *  Created on: Sep 15, 2019
 *      Author: lieyu
 */

#include "SuperpointGeneration.h"

SuperpointGeneration::SuperpointGeneration() {
	// TODO Auto-generated constructor stub

}

SuperpointGeneration::~SuperpointGeneration() {
	// TODO Auto-generated destructor stub
}


/*
 * use k-means clustering to cluster the spatial points into several clusters
 */
void SuperpointGeneration::get_superpoint_bandwidth(const std::vector<Eigen::Vector3d>& coordinates, int& numOfClusters,
		std::vector<double>& bandwidth, double& maxBandwidth, bool use_kmeans_plus_plus)
{
	std::cout << use_kmeans_plus_plus << std::endl;

	// initial samples for k-means clustering
	std::vector<Eigen::Vector3d> centroids;

	if(use_kmeans_plus_plus)
		generateSamplesByKmeans_plus_plus(coordinates, numOfClusters, centroids);
	else
		generateRandomSamples(coordinates, numOfClusters, centroids);

	perform_kmeans(coordinates, numOfClusters, centroids);

	calculate_bandwidth(coordinates, numOfClusters, centroids, bandwidth, maxBandwidth);
}


// use random sample strategy to generate intial samples for k-means clustering
void SuperpointGeneration::generateRandomSamples(const std::vector<Eigen::Vector3d>& coordinates, int& numOfClusters,
		std::vector<Eigen::Vector3d>& centroids)
{
	std::cout << "k-means initialization start..." << std::endl;

	// get the number of points
	const int& numOfVertices = coordinates.size();

	centroids.resize(numOfClusters);

	std::unordered_set<int> chosenPoint;	// use a set to record the points already chosen
	srand(time(NULL));

	int count = 0;
	do
	{
		// generate random index
		int sample = rand()%numOfVertices;

		if(!chosenPoint.count(sample))
		{
			centroids[count] = coordinates[sample];
			chosenPoint.insert(sample);
			++count;
		}
	}while(count < numOfClusters);
}


// use k-means++ to generate intial samples for k-means clustering
void SuperpointGeneration::generateSamplesByKmeans_plus_plus(const std::vector<Eigen::Vector3d>& coordinates,
		int& numOfClusters, std::vector<Eigen::Vector3d>& centroids)
{
	std::cout << "k-means++ initialization start..." << std::endl;

	// get the number of points
	const int& numOfVertices = coordinates.size();

	centroids.resize(numOfClusters);

	std::unordered_set<int> chosenPoint;	// use a hash set to record the points already chosen
	srand(time(NULL));

	Eigen::VectorXd distance;	// use to store the vector of distance
	int count = 0, sample;
	do
	{
		if(chosenPoint.empty())	// for first iteration
		{
			sample = rand()%numOfVertices;
			chosenPoint.insert(sample);
			centroids[count] = coordinates[sample];
			++count;
		}
		else
		{
			distance = Eigen::VectorXd::Zero(numOfVertices);

			// calculate the distance to the closest candidates, and their squared summation
			double minDist;
			long double summation = 0.0;
			for(int i=0; i<numOfVertices; ++i)
			{
				if(!chosenPoint.count(i))	// the point has not been chosen
				{
					minDist = DBL_MAX;

					// find the minimal distance to the chosen centers
					for(auto &iter:chosenPoint)
					{
						minDist = std::min(minDist, (coordinates[i]-coordinates[iter]).norm());
					}
					distance(i) = minDist*minDist;
					summation += distance(i);
				}
			}

			// perform normalization
			distance/=summation;

			// choose the element based on the uniform probability
			bool found = false;
			do
			{
				// generate the elements by the uniform distribution
				double percentage = rand()/RAND_MAX;

				double left = 0.0, right = 0.0;
				for (int i = 0; i < numOfVertices; ++i)
				{
					left = right;
					right += distance(i);
					// find the lower_bound of the probability
					if(left < percentage && percentage <= right)
					{
						if(!chosenPoint.count(i))	// fortunately the chosen point index is not included before
						{
							found = true;
							chosenPoint.insert(i);
							centroids[count] = coordinates[i];
							++count;
						}
						break;
					}
				}
			}while(!found);
		}
	}while(count<numOfClusters);
}


// perform k-means clustering
void SuperpointGeneration::perform_kmeans(const std::vector<Eigen::Vector3d>& coordinates, int& numOfClusters,
		std::vector<Eigen::Vector3d>& centroids)
{
	const int& numOfVertices = coordinates.size();

	double moving=1000, tempMoving, before;

	neighborCandidates.resize(numOfClusters);
	groupTag.resize(numOfVertices);

	std::vector<Eigen::Vector3d> tempCentroid(numOfClusters);

	std::cout << "...k-means started!" << std::endl;

	struct timeval start, end;
	gettimeofday(&start, NULL);

	int iteration = 0;
	do
	{
		before = moving;

	#pragma omp parallel for schedule(dynamic) num_threads(8)
		for (int i = 0; i < numOfClusters; ++i)
		{
			neighborCandidates[i].clear();
			tempCentroid[i] = Eigen::VectorXd::Zero(3);
		}

	#pragma omp parallel num_threads(8)
		{
		#pragma omp for nowait
			for (int i = 0; i < numOfVertices; ++i)
			{
				double dist = DBL_MAX, temp;
				int clusTemp = -1;
				for (int j = 0; j < numOfClusters; ++j)
				{
					temp = (coordinates[i]-centroids[j]).norm();
					if(temp<dist)
					{
						dist = temp;
						clusTemp = j;
					}
				}

			#pragma omp critical
				{
					neighborCandidates[clusTemp].push_back(i);
					groupTag[i] = clusTemp;
					tempCentroid[clusTemp]+=coordinates[i];
				}
			}
		}

		// check the maximal moving threshold of centroids
		moving = 0.0;
	#pragma omp parallel for reduction(max:moving) num_threads(8)
		for (int i = 0; i < numOfClusters; ++i)
		{
			if(neighborCandidates[i].size()>0)
			{
				tempCentroid[i]/=neighborCandidates[i].size();
				tempMoving = (tempCentroid[i]-centroids[i]).norm();
				if(moving<tempMoving)
					moving = tempMoving;
			}
		}

		// update the centroids
		centroids = tempCentroid;
		std::cout << "K-means iteration " << (++iteration) << " completed, and maximal centroid moving is "
				  << moving << "!" << std::endl;
	}while(abs(moving-before)/before >= MOVE_THRESHOLD && iteration < 50);

	gettimeofday(&end, NULL);
	double timeTemp = ((end.tv_sec-start.tv_sec)*1000000u+end.tv_usec-start.tv_usec)/1.e6;

	std::cout << "k-means takes " << timeTemp << " seconds!" << std::endl;

 }


// calculate the bandwidth from clustering result
void SuperpointGeneration::calculate_bandwidth(const std::vector<Eigen::Vector3d>& coordinates, int& numOfClusters,
		std::vector<Eigen::Vector3d>& centroids, std::vector<double>& bandwidth, double& maxBandwidth)
{
	const int& numOfVertices = coordinates.size();
	bandwidth.resize(numOfVertices);

	maxBandwidth = 0.0;

	// assign the same 1.0/bandwidth to the candidates inside the same cluster
	for(int i=0; i<numOfClusters; ++i)
	{
		long double summation = 0.0;

		auto& each = neighborCandidates[i];
		if(each.empty())
			continue;

		for(int j=0; j<each.size(); ++j)
		{
			summation += (coordinates[each[j]]-centroids[i]).norm();
		}
		double bwidth = double(each.size())/summation;

		// get the max band width
		maxBandwidth = std::max(maxBandwidth, 1.0/bwidth);

		for(int j=0; j<each.size(); ++j)
		{
			bandwidth[each[j]] = bwidth;	// it's actually storing the 1.0/h
		}
	}

	std::cout << "The max band width is " << maxBandwidth << std::endl;
}
