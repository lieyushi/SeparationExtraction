#ifndef READ_DATA_H_
#define READ_DATA_H_

#include <iostream>
#include <vector>
#include <math.h>
#include <fstream>
#include <stdlib.h>
#include <float.h>
#include <time.h>
#include <sstream>
#include <algorithm>
#include <cassert>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SVD>
#include <bits/stdc++.h>
#include "DataSet.h"
#include "VTKWritter.h"

using namespace std;
using namespace Eigen;


#define MINIMAL_VELOCITY 1.0E-6

// a struct to showcase a point on the streamline with a given id
struct StreamlinePoint
{
	Eigen::Vector3d coordinate;	// point coordinate
	int id;	// streamline id

	StreamlinePoint(const Eigen::Vector3d& point, const int& streamlineID): coordinate(point), id(streamlineID)
	{}
};

// a three dimension vector field storage
class VectorField
{
public:

	// default constructor
	VectorField();

	// destructor
	~VectorField();

	// read vector field data
	void readVectorField(const string& fileName);

	// print vector field for certification
	void printVectorFieldVTK();

	// trace streamlines w.r.t. some parameters and seeding strategy
	void traceStreamlines(const double& integrationStepRatio, const int& maxLength, int& maxSeeding);

	// print vtk for streamlines
	void printStreamlinesVTK();

	// write vector values to the streamline tracer
	void writeSeparationToStreamlines(const std::vector<int>& separationFlag, const string& flagName);

	// read 3D streamlines from streamline data set
	void readStreamlineFromFile(const string& fileName);

	// print streamline data sets into clustering framework
	void printStreamlineTXT();

	// elimiate short length of streamlines below numSize points
	void filterShortStreamlines(const double& numRatio);

	// storage of vertex information (coordinates and velocity components)
	std::vector<Vertex> vertexVec;

	// number of vertex count
	int vertexCount;

	// streamline holder
	std::vector<Eigen::VectorXd> streamlineVector;

	// velocity magnitude on the streamline points
	std::vector<std::vector<double> > lineVelocity;

	// the length of streamlines to be filtered for smaller lengths
	std::vector<double> lineLength;

	// coordinates limit
	CoordinateLimits limits[3];

	// original data set name
	string dataset_name;

	// rectilinear grid information, for 2D, Z_RESOLUTION = 1
	int X_RESOLUTION = -1, Y_RESOLUTION = -1, Z_RESOLUTION = -1;
	double X_STEP = -1.0, Y_STEP = -1.0, Z_STEP = 1.0;

	// the ratio to the diagonal distance for the distance threshold for generating streamlines far away enough
	double minimal_ratio = 0.2;
	double MINIMAL;

	// whether the data set is 3D or not. It decides whether to call 2D or 3D interpolation
	bool is3D = false;

private:
	// read vector field from ply file
	void readPlyFile(const string& fileName);

	// read vector field from plain file
	void readPlainFile(const string& fileName);

	// read vector field from specific format
	void readVectorFieldFromRaw(const string& fileName);

	// use uniform sampling for streamline tracing
	void uniformSeeding(std::vector<Vertex>& seeds, int& maxSeeding);

	// use entropy-based sampling for streamline tracing
	// entropy-based computing: http://web.cse.ohio-state.edu/~shen.94/papers/xu_vis10.pdf
	// seeding strategy: http://vis.cs.ucdavis.edu/papers/pg2011paper.pdf
	void entropySeeding(std::vector<Vertex>& seeds, int& maxSeeding);

	// fourth-order Runge-Kutta integration method for streamline tracing
	bool getIntegrationValue(const double& step, const Eigen::Vector3d& position,
							 const Eigen::Vector3d& velocity, Eigen::Vector3d& nextPos);

	// get the velocity of temporary position
	bool getInterpolatedVelocity(const Eigen::Vector3d& position, Eigen::Vector3d& velocity);

	// get the velocity of temporary position for bilinear interpolation
	bool getInterpolatedVelocity2D(const Eigen::Vector3d& position, Eigen::Vector3d& velocity);

	// get the velocity of temporary position for trilinear interpolation
	bool getInterpolatedVelocity3D(const Eigen::Vector3d& position, Eigen::Vector3d& velocity);

	// trace streamlines given seeding vertex position
	void traceStreamlinesBySeeds(const std::vector<Vertex>& seeds, const double& step, const int& maxLength);

	// stay in the range of the domain
	bool stayInDomain(const Eigen::Vector3d& pos);

	// whether stay away from existing point in the streamline far enough
	bool isFarToStreamlines(const Eigen::Vector3d& nextPos, const int& id);

	// spatial binning for checking whether current streamline points are close to existing points or not
	std::vector<std::vector<StreamlinePoint> > spatialBins;
	bool useSpatial = false;
};

#endif
