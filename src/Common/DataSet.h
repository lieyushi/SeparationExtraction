/*
 * DataSet.h
 *
 *  Created on: Feb 21, 2019
 *      Author: lieyu
 */

#ifndef SRC_COMMON_DATASET_H_
#define SRC_COMMON_DATASET_H_

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

class DataSet {
public:
	DataSet();
	virtual ~DataSet();

	/*
	 * read 3D vector field from some given customized 3D data sets
	 */
	static void read3DVectorField(const string& fileName, std::vector<Vertex>& vertexVec,
			CoordinateLimits limits[3]);
};

#endif /* SRC_COMMON_DATASET_H_ */
