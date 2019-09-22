/*
 * SuperpointGeneration.h
 *
 *  Created on: Sep 15, 2019
 *      Author: lieyu
 */

#ifndef SRC_LOCALSCALAR_SUPERPOINTGENERATION_H_
#define SRC_LOCALSCALAR_SUPERPOINTGENERATION_H_

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
#include <sys/time.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SVD>
#include <unordered_map>
#include <unordered_set>
#include <ctime>

#define MOVE_THRESHOLD 1.0E-5


using namespace std;
using namespace Eigen;


class SuperpointGeneration {
public:
	SuperpointGeneration();
	virtual ~SuperpointGeneration();

	// cluster tag for all the candidates
	std::vector<int> groupTag;

	// the neighboring candidates
	std::vector<std::vector<int> > neighborCandidates;

	/*
	 * use k-means clustering to cluster the spatial points into several clusters
	 */
	void get_superpoint_bandwidth(const std::vector<Eigen::Vector3d>& coordinates, int& numOfClusters,
			std::vector<double>& bandwidth, bool use_kmeans_plus_plus = false);

private:

	// use random sample strategy to generate intial samples for k-means clustering
	void generateRandomSamples(const std::vector<Eigen::Vector3d>& coordinates, int& numOfClusters,
			std::vector<Eigen::Vector3d>& centroids);

	// use random sample strategy to generate intial samples for k-means clustering
	void generateSamplesByKmeans_plus_plus(const std::vector<Eigen::Vector3d>& coordinates, int& numOfClusters,
			std::vector<Eigen::Vector3d>& centroids);

	// perform k-means clustering
	void perform_kmeans(const std::vector<Eigen::Vector3d>& coordinates, int& numOfClusters,
			std::vector<Eigen::Vector3d>& centroids);

	// calculate the bandwidth from clustering result
	void calculate_bandwidth(const std::vector<Eigen::Vector3d>& coordinates, int& numOfClusters,
			std::vector<Eigen::Vector3d>& centroids, std::vector<double>& bandwidth);

};

#endif /* SRC_LOCALSCALAR_SUPERPOINTGENERATION_H_ */
