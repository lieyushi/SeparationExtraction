/*
 * LocalScalar.h
 *
 *  Created on: Mar 11, 2019
 *      Author: lieyu
 */

#ifndef SRC_LOCALSCALAR_LOCALSCALAR_H_
#define SRC_LOCALSCALAR_LOCALSCALAR_H_

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
#include "SimilarityDistance.h"
#include "VTKWritter.h"

using namespace std;
using namespace Eigen;

// parameter set-up for Newton Iteration
#define Relaxation 0.8
#define NewtonIteration 200
#define ErrorThreshold 1.0E-4

// a struct to showcase a point on the streamline with a given id
struct streamPoint
{
	int streamID;	// streamline id
	int vertexID;

	streamPoint(const int& streamlineID, const int& vertexID): streamID(streamlineID), vertexID(vertexID)
	{}
};

struct coordinateLimit
{
	double inf, sup;
	coordinateLimit(): inf(DBL_MAX), sup(-DBL_MAX)
	{}
};

enum DirectionSearch
{
	KNN = 1,
	LeftAndRight,
	EightDirections,
	NineVoxelDirection
};

struct MinimalDist
{
	double dist;
	int index;

	MinimalDist(): dist(-1.0), index(-1)
	{}

	MinimalDist(const double& dist, const int& index): dist(dist), index(index)
	{}
};


// a compare struct for min heap to find k-smallest distance
struct CompareDistRecord
{
	bool operator()(const MinimalDist& d1, const MinimalDist& d2)
	{
		return d1.dist<d2.dist;
	}
};


class LocalScalar {
public:
	// LocalScalar();
	LocalScalar(std::vector<Eigen::VectorXd>& coordinates, const string& name);

	/* compute local scalar values on streamline segments */
	void getLocalScalar();

	virtual ~LocalScalar();

private:

	/* How many streamlines? */
	int streamlineCount = -1;

	/* data set name */
	string datasetName;

	/* vertex total number */
	int streamlineVertexCount = 0;

	/* coordinates of streamlines */
	std::vector<Eigen::VectorXd>& coordinates;

	/* separation scalar measure, 1.point-based, 2.discrete curvature, 3.line vector direction entropy */
	int scalarMeasurement;

	/* segment option, 1.log2(vertexSize) sampling, 2.3-1-3 sampling, 3.curvature threshold sampling */
	int segmentOption;

	/* k value in KNN */
	int kNeighborNumber;

	/* encoding is enabled or not */
	bool encodingEnabled;

	/* neighbor difference measurement method: 1. point distance ratio, 2. standard deviation */
	int separationMeasureOption;

	/* whether the deviation is normalized or not */
	bool useNormalizedDevi;

	/* to judge whether search direction is close to some point or not */
	double searchThreshold;

	/* whether use two of max value or not */
	bool useMaxRatio;

	/* vertex coordinates system */
	std::vector<Eigen::Vector3d> vertexVec;

	/* vertex to streamline mapping */
	std::vector<int> vertexToLine;

	/* streamline to vertex mapping */
	std::vector<std::vector<int> > streamlineToVertex;

	/* how many directions are used */
	int directionNumbers;

	/* get user-input option */
	void getUserInputParameter();

	/* -----------------------------------------------------------------------------------------------------------
	 * ---------------------------------Segments based on signature sampling--------------------------------------
	 * -----------------------------------------------------------------------------------------------------------
	 */
	/* segment option as 1 with signature-based segmentation */
	void processSegmentByCurvature();

	/* get scalar value based on KNN */
	void getScalarsOnSegments(const Eigen::MatrixXd& distanceMatrix, std::vector<double>& segmentScalars,
			const std::vector<Eigen::VectorXd>& lineCurvatures, std::vector<int>& segmentToStreamline);

	/* segment->streamline */
	void getSegmentToStreamlineVec(std::vector<int>& segmentToStreamline,
			const std::vector<std::vector<int> >& segmentOfStreamlines);

	/* use while loop to find K nearest segments for chi-test calculation */
	void processWithBinsOnSegments(const std::vector<int>& pointToSegment, std::vector<double>& segmentScalars,
			const std::vector<Eigen::VectorXd>& lineCurvatures);

	/* -----------------------------------------------------------------------------------------------------------
	 * ---------------------------------Spatial bining operation to find knn--------------------------------------
	 * -----------------------------------------------------------------------------------------------------------
	 */
	// rectilinear grid information
	int X_RESOLUTION = -1, Y_RESOLUTION = -1, Z_RESOLUTION = -1;
	double X_STEP = -1.0, Y_STEP = -1.0, Z_STEP = 1.0;

	// spatial binning for checking whether current streamline points are close to existing points or not
	std::vector<std::vector<streamPoint> > spatialBins;

	coordinateLimit range[3];

	/* spatial bining for all the points in the grid */
	void assignPointsToBins();

	/* -----------------------------------------------------------------------------------------------------------
	 * -----------------------------Based on 3-1-3 alignment, with point distance --------------------------------
	 * -----------------------------------------------------------------------------------------------------------
	 */

	/* follow a 3-1-3 alignment, judge point distance with lnk/(lnk+1) mapping such that
	 * 0, infinite large--->1 (stronger separation), 1--->0 (weaker separation)
	 */
	void processAlignmentByPointWise();

	// compute the scalar value on point-wise element of the streamline
	void getScalarsOnPointWise(const DirectionSearch& directionOption, const bool& sameLineEnabled, const int& j,
			const std::vector<int>& vertexArray, std::vector<int>& pointCandidate);

	// find KNN closest point in the spatial bining
	void searchKNNPointByBining(const bool& sameLineEnabled, const int& j,
			const std::vector<int>& vertexArray, std::vector<int>& pointCandidate);

	// search through two directions
	void searchClosestThroughDirections(const bool& sameLineEnabled, const int& j,
			const std::vector<int>& vertexArray, std::vector<int>& pointCandidate);

	// search through eight perpendicular directions and each find the streamline passing through the point
	void searchNeighborThroughSeparateDirections(const bool& sameLineEnabled, const int& j,
			const std::vector<int>& vertexArray, std::vector<int>& pointCandidate);

	// search through all nine voxels. If one voxel has em	// double angles[] = {M_PI/6.0, M_PI/3.0, M_PI/2.0, M_PI/3.0*2.0, M_PI/6.0*5.0};

	// double angles[] = {M_PI/6.0, M_PI/3.0, M_PI/2.0, M_PI/3.0*2.0, M_PI/6.0*5.0};pty, will search into other voxel along that direction
	void searchNeighborThroughVoxels(const bool& sameLineEnabled, const int& j,
			const std::vector<int>& vertexArray, std::vector<int>& pointCandidate);

	// get scalar value based on neighborhood
	void getScalarsFromNeighbors(const int& j, const std::vector<int>& vertexArray,
			const int& TOTALSIZE, const std::vector<int>& closestPoint, double& scalar);

	// get scalar value based on difference deviation of point-wise distance
	void getScalarsFromDeviation(const int& j, const std::vector<int>& vertexArray,
			const int& TOTALSIZE, const std::vector<int>& closestPoint, double& scalar);

	/*
	 * Some functions to be placed here to get necessary information
	 */

	// given a planar, get eight normalized directions separated by 45 degrees
	void findDirectionVectors(const Eigen::Vector3d& tangential, const Eigen::Vector3d& center,
			std::vector<Eigen::Vector3d>& directionVectors);

	// use quadratic system to get the two directions w.r.t. reference
	void findDirectionsToAngles(const Eigen::Vector3d& tangential, const Eigen::Vector3d& reference,
			const double& angle, std::vector<Eigen::Vector3d>& directionVectors);

};

// perform Newton iteration to solve non-linear equation
void findSolutionByNewtonIteration(const Eigen::Vector3d& normal, const Eigen::Vector3d& reference,
		const double& angle, Eigen::Vector3d& solution);

#endif /* SRC_LOCALSCALAR_LOCALSCALAR_H_ */
