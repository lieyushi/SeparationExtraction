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

using namespace std;
using namespace Eigen;

#define MINIMAL 1.0E-6

// a struct of vertex to record vector field values at vertex
struct Vertex
{
	double x, y, z;
	double vx, vy, vz;
	double v_magnitude;

	// default constructor
	Vertex(): x(-1), y(-1), z(-1), vx(-1), vy(-1), vz(-1)
	{
		v_magnitude = 0.0;
	}

	// argument constructor I
	Vertex(const double& x, const double& y, const double& z,
		   const double& vx, const double& vy, const double& vz)
		   : x(x), y(y), z(z), vx(vx), vy(vy), vz(vz)
	{
		v_magnitude = std::sqrt(vx*vx+vy*vy+vz*vz);
	}

	// argument constructor II
	Vertex(const std::vector<double>& value): x(value[0]), y(value[1]), z(value[2]),
			 vx(value[3]), vy(value[4]), vz(value[5])
	{
		v_magnitude = std::sqrt(vx*vx+vy*vy+vz*vz);
	}

	// argument constructor III
	Vertex(const Eigen::Vector3d& pos, const Eigen::Vector3d& vel): x(pos(0)), y(pos(1)), z(pos(2)),
			 vx(vel(0)), vy(vel(1)), vz(vel(2))
	{
		v_magnitude = std::sqrt(vx*vx+vy*vy+vz*vz);
	}

	// assignment operator
	void assignValue(const std::vector<double>& value)
	{
		x = value[0];
		y = value[1];
		z = value[2];
		vx = value[3];
		vy = value[4];
		vz = value[5];
		v_magnitude = std::sqrt(vx*vx+vy*vy+vz*vz);
	}
};

struct CoordinateLimits
{
	double inf, sup;
	CoordinateLimits(): inf(DBL_MAX), sup(-DBL_MAX)
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
	void traceStreamlines(const double& integrationStep, const int& maxLength, const int& maxSeeding);

	// print vtk for streamlines
	void printStreamlinesVTK();

	// write vector values to the streamline tracer
	void writeSeparationToStreamlines(const std::vector<int>& separationFlag, const string& flagName);

	// storage of vertex information (coordinates and velocity components)
	std::vector<Vertex> vertexVec;

	// number of vertex count
	int vertexCount;

	// streamline holder
	std::vector<Eigen::VectorXd> streamlineVector;

private:
	// read vector field from ply file
	void readPlyFile(const string& fileName);

	// read vector field from plain file
	void readPlainFile(const string& fileName);

	// use uniform sampling for streamline tracing
	void uniformSeeding(std::vector<Vertex>& seeds, const int& maxSeeding);

	// use entropy-based sampling for streamline tracing
	// entropy-based computing: http://web.cse.ohio-state.edu/~shen.94/papers/xu_vis10.pdf
	// seeding strategy: http://vis.cs.ucdavis.edu/papers/pg2011paper.pdf
	void entropySeeding(std::vector<Vertex>& seeds, const int& maxSeeding);

	// fourth-order Runge-Kutta integration method for streamline tracing
	bool getIntegrationValue(const double& step, const Eigen::Vector3d& position,
							 const Eigen::Vector3d& velocity, Eigen::Vector3d& nextPos);

	// get the velocity of temporary position
	bool getInterpolatedVelocity(const Eigen::Vector3d& position, Eigen::Vector3d& velocity);

	// trace streamlines given seeding vertex position
	void traceStreamlinesBySeeds(const std::vector<Vertex>& seeds, const double& step, const int& maxLength);

	// stay in the range of the domain
	bool stayInDomain(const Eigen::Vector3d& pos);

	// coordinates limit
	CoordinateLimits limits[3];

	// original data set name
	string dataset_name;

	// rectilinear grid information
	int SIZE_SQRT;
	double X_STEP, Y_STEP;


};

#endif
