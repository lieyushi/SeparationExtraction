/*
 * LineClustering.h
 *
 *  Created on: Mar 6, 2019
 *      Author: lieyu
 */

#ifndef SRC_STREAMLineClustering_LineClustering_H_
#define SRC_STREAMLineClustering_LineClustering_H_

#include <unordered_set>
#include <unordered_map>
#include <map>
#include <string>
#include <algorithm>
#include <fstream>
#include <vector>
#include <iostream>
#include <cstring>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <climits>
#include <cassert>
#include <float.h>
#include <unordered_map>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SVD>

using namespace std;
using namespace Eigen;

struct Cluster
{
	int index;
	std::vector<int> element;

	Cluster(): index(-1)
	{}

	Cluster(const int& index, const std::vector<int>& element): index(index), element(element)
	{}
};

class LineClustering {
public:
	//LineClustering();

	LineClustering(Eigen::MatrixXd& distanceMatrix, const string& name, const int& vertexCount);

	virtual ~LineClustering();

	/* this is a pure virtual member function */
	virtual void performClustering() = 0;

	/* reassign cluster numbers w.r.t. cluster number */
	void reassignClusterAscending();

	// write different cluster labels into vtk file
	void writeLabelsIntoVTK(const std::vector<Eigen::VectorXd>& coordinates, const string& labelName);

	// print vtk for streamlines
	void printStreamlinesVTK(const std::vector<Eigen::VectorXd>& coordinates);

	/* cluster of each streamlines */
	std::vector<int> groupID;

	/* storage of each group */
	std::vector<std::vector<int> > storage;

	/* distance matrix */
	Eigen::MatrixXd& distanceMatrix;

protected:

	/* How many streamlines? */
	int streamlineCount = -1;

	/* data set name */
	string datasetName;

	/* vertex total number */
	int streamlineVertexCount = 0;

private:

	// get storage
	void getStorageFromGroup();
};

#endif /* SRC_STREAMLineClustering_LineClustering_H_ */
