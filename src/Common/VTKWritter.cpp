/*
 * VTKWritter.cpp
 *
 *  Created on: Mar 12, 2019
 *      Author: lieyu
 */

#include "VTKWritter.h"

/* print streamlines as segment */
void VTKWritter::printStreamlineSegments(const std::vector<Eigen::VectorXd>& coordinates, const string& datasetName,
		const std::vector<int>& pointToSegment, const int& streamlineVertexCount)
{
	if(coordinates.empty())
		return;

	const int& streamlineCount = coordinates.size();
	stringstream ss;
	ss << datasetName << "_streamline.vtk";

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

	fout << "SCALARS segmentID int 1" << std::endl;
	fout << "LOOKUP_TABLE segmentID_table" << std::endl;
	for (int i = 0; i < streamlineVertexCount; ++i)
	{
		fout << pointToSegment[i] << std::endl;
	}

	fout.close();

	std::cout << "vtk printed!" << std::endl;
}


/* print streamlines with scalars on segments */
void VTKWritter::printStreamlineScalarsOnSegments(const std::vector<Eigen::VectorXd>& coordinates,
		const string& datasetName, const int& streamlineVertexCount, const std::vector<Eigen::VectorXd>& lineSegments,
		const std::vector<double>& segmentScalars)
{
	if(coordinates.empty())
		return;

	const int& streamlineCount = coordinates.size();
	const int& numberOfSegments = lineSegments.size();

	stringstream ss;
	ss << datasetName << "_streamline.vtk";

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

	fout << "SCALARS segmentScalar double 1" << std::endl;
	fout << "LOOKUP_TABLE segmentScalar_table" << std::endl;

	int segmentSize;
	for(int i=0; i<numberOfSegments; ++i)
	{
		segmentSize = lineSegments[i].size()/3;
		for(int j=0; j<segmentSize; ++j)
			fout << segmentScalars[i] << std::endl;
	}
	fout.close();

	std::cout << "vtk printed!" << std::endl;
}


/* print streamlines with scalars on segments */
void VTKWritter::printStreamlineScalarsOnSegments(const std::vector<Eigen::VectorXd>& coordinates,
		const string& datasetName, const int& streamlineVertexCount, const std::vector<double>& segmentScalars)
{
	if(coordinates.empty())
		return;

	const int& streamlineCount = coordinates.size();

	stringstream ss;
	ss << datasetName << "_streamline.vtk";

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

	fout << "SCALARS pointScalar double 1" << std::endl;
	fout << "LOOKUP_TABLE pointScalar_table" << std::endl;

	for(int i=0; i<streamlineVertexCount; ++i)
	{
		fout << segmentScalars[i] << std::endl;
	}
	fout.close();

	std::cout << "vtk printed!" << std::endl;
}
