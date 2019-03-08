/*
 * LineClustering.cpp
 *
 *  Created on: Mar 6, 2019
 *      Author: lieyu
 */

#include "LineClustering.h"

//LineClustering::LineClustering(): coordinates(NULL), distanceMatrix(NULL) {
//	// TODO Auto-generated constructor stub
//
//}

LineClustering::LineClustering(std::vector<Eigen::VectorXd>& coordinates, Eigen::MatrixXd& distanceMatrix,
		const string& name, const int& vertexCount): coordinates(coordinates), distanceMatrix(distanceMatrix),
				datasetName(name), streamlineVertexCount(vertexCount), streamlineCount(coordinates.size())
{
}


LineClustering::~LineClustering() {
	// TODO Auto-generated destructor stub
}



/* reassign cluster numbers w.r.t. cluster number */
void LineClustering::reassignClusterAscending()
{
	std::vector<Cluster> originalClusters(storage.size());

	if(groupID.empty())
	{
		groupID.resize(streamlineCount);
	}

	for(int i=0; i<storage.size(); ++i)
	{
		originalClusters[i].index = i;
		originalClusters[i].element = storage[i];
	}

	/* reorder the cluster id by its size in ascending order */
	std::sort(originalClusters.begin(), originalClusters.end(), [](const Cluster& first, const Cluster& second){
		return first.element.size()<second.element.size();});

	std::vector<int> element;
	for(int i=0; i<originalClusters.size(); ++i)
	{
		element = originalClusters[i].element;
	#pragma omp parallel for schedule(static) num_threads(8)
		for(int j=0; j<element.size(); ++j)
		{
			groupID[element[j]] = i;
		}
		storage[i] = element;
	}
}


// print vtk for streamlines
void LineClustering::printStreamlinesVTK()
{
	if(coordinates.empty())
		return;

	stringstream ss;
	ss << datasetName << "_streamline.vtk";

	/* in case of file existing */
	std::ifstream fin(ss.str().c_str(), ios::in);
	if(!fin.fail())
		return;

	fin.close();
	std::ofstream fout(ss.str().c_str(), ios::out);
	if(!fout)
	{
		std::cout << "Error creating a new file!" << std::endl;
		exit(1);
	}
	fout << "# vtk DataFile Version 3.0" << std::endl << "streamline" << std::endl
		 << "ASCII" << std::endl << "DATASET POLYDATA" << std::endl;
	fout << "POINTS " << streamlineVertexCount << " double" << std::endl;

	int subSize, arraySize;
	Eigen::VectorXd tempRow;
	for (int i = 0; i < streamlineCount; ++i)
	{
		tempRow = coordinates[i];
		subSize = tempRow.size()/3;
		for (int j = 0; j < subSize; ++j)
		{
			for (int k = 0; k < 3; ++k)
			{
				fout << tempRow(j*3+k) << " ";
			}
			fout << endl;
		}
	}

	fout << "LINES " << streamlineCount << " " << (streamlineVertexCount+streamlineCount) << std::endl;

	subSize = 0;
	for (int i = 0; i < streamlineCount; ++i)
	{
		arraySize = coordinates[i].size()/3;
		fout << arraySize << " ";
		for (int j = 0; j < arraySize; ++j)
		{
			fout << subSize+j << " ";
		}
		subSize+=arraySize;
		fout << std::endl;
	}
	fout << "POINT_DATA" << " " << streamlineVertexCount << std::endl;
	fout << "SCALARS streamID int 1" << std::endl;
	fout << "LOOKUP_TABLE group_table" << std::endl;

	for (int i = 0; i < streamlineCount; ++i)
	{
		arraySize = coordinates[i].size()/3;
		for (int j = 0; j < arraySize; ++j)
		{
			fout << i << std::endl;
		}
	}

	fout.close();
}


// write different cluster labels into vtk file
void LineClustering::writeLabelsIntoVTK(const string& labelName)
{
	if(coordinates.empty())
		return;

	stringstream ss;
	ss << datasetName << "_streamline.vtk";

	std::ofstream fout(ss.str().c_str(), ios::out | ios::app );
	if(!fout)
	{
		std::cout << "Error opening the file!" << std::endl;
		exit(1);
	}

	fout << "SCALARS " << labelName << " int 1" << std::endl;
	fout << "LOOKUP_TABLE " << labelName+string("_table") << std::endl;

	int arraySize;
	for (int i = 0; i < streamlineCount; ++i)
	{
		arraySize = coordinates[i].size()/3;
		for (int j = 0; j < arraySize; ++j)
		{
			fout << groupID[i] << std::endl;
		}
	}
	fout.close();
}


// get storage
void LineClustering::getStorageFromGroup()
{
	assert(!storage.empty());
	for(int i=0; i<streamlineCount; ++i)
	{
		storage[groupID[i]].push_back(i);
	}
}
