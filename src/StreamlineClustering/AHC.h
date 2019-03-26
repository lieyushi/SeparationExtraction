/*
 * AHC.h
 *
 *  Created on: Mar 6, 2019
 *      Author: lieyu
 */

#ifndef SRC_STREAMLINECLUSTERING_AHC_H_
#define SRC_STREAMLINECLUSTERING_AHC_H_

#include "LineClustering.h"
#include "DetermClusterNum.h"

// define a treeNode structure to store AHC clustering tree
struct Ensemble
{
	int index = -1;

	/* to alleviate the computational cost to traverse all node elements */
	std::vector<int> element;

	Ensemble(const int& index): index(index)
	{}

	Ensemble()
	{}
};

// remove two elements in template vector
template <class T>
void deleteVecElements(std::vector<T>& origine, const T& first, const T& second);


/* we will use a min-heap to perserve sorted distance for hirarchical clustering */
struct DistNode
{
	int first = -1, second = -1;
	double distance = -1.0;

	DistNode(const int& first, const int& second, const double& dist):first(first), second(second), distance(dist)
	{}

	DistNode()
	{}
};


class AHC: public LineClustering
{
public:
	// AHC();

	AHC(Eigen::MatrixXd& distanceMatrix, const string& name, const int& vertexCount);

	void performClustering();

	virtual ~AHC();

private:

	/* linkage choice */
	int linkageOption;

	/* whether used L-method to detect optimal number of clusters */
	bool lMethod;

	/* cluster numbers */
	int numberOfClusters;

	/* compute distance between two clusters based on likage type */
	const double getDistAtNodes(const vector<int>& firstList, const vector<int>& secondList, const int& Linkage);

	/* perform AHC merging by given a distance threshold */
	void hierarchicalMerging(std::unordered_map<int, Ensemble>& node_map, std::vector<DistNode>& dNodeVec,
			std::vector<Ensemble>& nodeVec);

	/* set a vector for min-heap */
	void setValue_merge(std::vector<DistNode>& dNodeVec, std::unordered_map<int, Ensemble>& node_map);

	/* set input parameters for AHC clustering */
	void setInputParameters();

	/* perform group-labeling information */
	void setLabel(const std::vector<Ensemble>& nodeVec);
};

#endif /* SRC_STREAMLINECLUSTERING_AHC_H_ */
