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


/* print 3d vector field with regular grid */
void VTKWritter::printVectorField(const string& fileName, const std::vector<Vertex>& vertexVec,
		CoordinateLimits limits[3], const int& x_resolution, const int& y_resolution, const int& z_resolution,
		const double& x_step, const double& y_step, const double& z_step)
{

	// create vtk file
	stringstream ss;
	ss << fileName << "_VectorField.vtk";
	std::ofstream fout(ss.str().c_str(), ios::out);
	if(fout.fail())
	{
		std::cout << "Error for creating vector field vtk file!" << std::endl;
		exit(1);
	}

	// writing out the vector field vtk information
    fout << "# vtk DataFile Version 3.0" << endl;
    fout << "Volume example" << endl;
    fout << "ASCII" << endl;
    fout << "DATASET STRUCTURED_POINTS" << endl;
    fout << "DIMENSIONS " << x_resolution << " " << y_resolution << " " << z_resolution << endl;
    fout << "ASPECT_RATIO " << x_step << " " << y_step << " " << z_step << endl;
    fout << "ORIGIN " << limits[0].inf << " " << limits[1].inf << " " << limits[2].inf << endl;
    fout << "POINT_DATA " << x_resolution*y_resolution*z_resolution << endl;

    const int& SLICE_NUMBER = x_resolution*y_resolution;

	fout << "SCALARS velocity_magnitude double 1" << endl;
	fout << "LOOKUP_TABLE velo_table" << endl;

	for (int i = 0; i < z_resolution; ++i)
	{
		for (int j = 0; j < y_resolution; ++j)
		{
			for (int k = 0; k < x_resolution; ++k)
			{
				fout << vertexVec[SLICE_NUMBER*i+x_resolution*j+k].v_magnitude << endl;
			}
		}
	}

    fout << "VECTORS velocityDirection double" << endl;
    Vertex vertex;
    for (int i = 0; i < z_resolution; ++i)
    {
		for (int j = 0; j < y_resolution; ++j)
		{
			for (int k = 0; k < x_resolution; ++k)
			{
				vertex = vertexVec[SLICE_NUMBER*i+x_resolution*j+k];
				fout << vertex.vx << " " << vertex.vy << " " << vertex.vz << endl;
			}
		}
    }
	fout.close();
}


/* print 3d point cloud */
void VTKWritter::printPoints(const string& fileName, const std::vector<Eigen::Vector3d>& pointArray)
{
	// create vtk file
	stringstream ss;
	ss << fileName << "_pvsolution.vtk";
	std::ofstream fout(ss.str().c_str(), ios::out);
	if(fout.fail())
	{
		std::cout << "Error for creating vector field vtk file!" << std::endl;
		exit(1);
	}

	const int& pointSize = pointArray.size();

	// writing out the vector field vtk information
    fout << "# vtk DataFile Version 3.0" << endl;
    fout << "point example" << endl;
    fout << "ASCII" << endl;
    fout << "DATASET UNSTRUCTURED_GRID" << endl;
    fout << "POINTS " << pointSize << " double" << std::endl;
    for(int i=0; i<pointSize; ++i)
    {
    	fout << pointArray[i](0) << " " << pointArray[i](1) << " " << pointArray[i](2) << std::endl;
    }
    fout << "CELLS " << pointSize << " " << 2*pointSize << endl;
	for (int i = 0; i < pointSize; i++)
	{
		fout << 1 << " " << i << endl;
	}
	fout << "CELL_TYPES " << pointSize << endl;
	for (int i = 0; i < pointSize; i++)
	{
		fout << 1 << endl;
	}
    fout.close();
}
