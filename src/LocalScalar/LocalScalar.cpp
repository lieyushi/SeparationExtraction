/*
 * LocalScalar.cpp
 *
 *  Created on: Mar 11, 2019
 *      Author: lieyu
 */

#include "LocalScalar.h"

LocalScalar::LocalScalar(std::vector<Eigen::VectorXd>& coordinates, const string& name):
	coordinates(coordinates), streamlineCount(coordinates.size()), datasetName(name)
{
	// TODO Auto-generated constructor stub

	for(int i=0; i<streamlineCount; ++i)
		streamlineVertexCount+=coordinates.at(i).size()/3;
	/* vertex coordinates system */
	vertexVec.resize(streamlineVertexCount);

	std::cout << "There are " << streamlineVertexCount << " points and " <<
			streamlineCount << " lines inside!" << std::endl;
	/* build a vertex to streamline mapping */
	vertexToLine.resize(streamlineVertexCount);

	/* streamline to vertex mapping */
	streamlineToVertex.resize(streamlineCount);

	int pointID = 0;
	Eigen::VectorXd line;
	int lineSize;
	for(int i=0; i<streamlineCount; ++i)
	{
		line = coordinates.at(i);
		lineSize = line.size()/3;
		std::vector<int>& vertexArray = streamlineToVertex[i];
		for(int j=0; j<lineSize; ++j)
		{
			vertexVec[pointID+j] = Eigen::Vector3d(line(3*j), line(3*j+1), line(3*j+2));
			vertexToLine[pointID+j] = i;
			vertexArray.push_back(pointID+j);
		}
		pointID+=lineSize;
	}
}


LocalScalar::~LocalScalar() {
	// TODO Auto-generated destructor stub
	vertexVec.clear();
	vertexToLine.clear();
	streamlineToVertex.clear();
}


/* compute local scalar values on streamline segments */
void LocalScalar::getLocalScalar()
{
	/*
	 * each streamline might be decomposed into log2(vertexCount) segments and we would like to segment it base
	 * on vertices w.r.t. maximal discrete curvatures
	 */
	getUserInputParameter();

	if(segmentOption==1) // log2 with curvature-based sampling
	{
		processSegmentByCurvature();
	}

	else if(segmentOption==2)
	{
		processAlignmentByPointWise();
	}
}


/* get user-input option */
void LocalScalar::getUserInputParameter()
{
	/* separation scalar measure, 1.point-based, 2.discrete curvature, 3.line vector direction entropy */
	int scalarMeasurement;

	/* segment option, 1.log2(vertexSize) sampling, 2.3-1-3 sampling, 3.curvature threshold sampling */
	/*
	std::cout << "Choose a segmentation option for streamlines, 1.log(vertexSize) sampling, 2.3-1-3 sampling," << std::endl;
	std::cin >> segmentOption;
	assert(segmentOption==1 || segmentOption==2 || segmentOption==3);
	*/
	segmentOption = 2;

	/* k value in KNN */
	std::cout << "Choose the k for KNN kernel: " << std::endl;
	std::cin >> kNeighborNumber;
	assert(kNeighborNumber>0 && kNeighborNumber<=int(streamlineCount*0.1));

	/* option of point-distance expansion */
	std::cout << "Choose point expansion option: 1.point distance ratio, 2.standard deviation: " << std::endl;
	std::cin >> separationMeasureOption;
	assert(separationMeasureOption==1 || separationMeasureOption==2);

	if(separationMeasureOption==1)
	{
		/* whether use two of max value or not */
		int encodingOption;
		std::cout << "Choose whether to select max ratio of separation or not? 1.Yes, 0.No," << std::endl;
		assert(encodingOption==1 || encodingOption==0);
		std::cin >> encodingOption;
		useMaxRatio = (encodingOption==1);

		/* log encoding is enabled or not */
		std::cout << "Choose whether log encoding is enabled or not? 1.Yes, 0.No," << std::endl;
		std::cin >> encodingOption;
		assert(encodingOption==1 || encodingOption==0);
		encodingEnabled = (encodingOption==1);
	}

	else if(separationMeasureOption==2)
	{
		std::cout << "Whether use normalized standard deviation or not? 1.Yes, 0.No." << std::endl;
		int useDevNormalization;
		std::cin >> useDevNormalization;
		assert(useDevNormalization==1 || useDevNormalization==0);
		useNormalizedDevi = (useNormalizedDevi==1);
	}
}


/* segment option as 1 with signature-based segmentation */
void LocalScalar::processSegmentByCurvature()
{
	std::vector<Eigen::VectorXd> lineSegments, lineCurvatures;
	std::vector<std::vector<int> > segmentOfStreamlines(streamlineCount);
	std::vector<int> pointToSegment(streamlineVertexCount);
	/*
	 * Each streamline has log2(vertexCount) curvature-based samples, hence has log2(...)+1 segments
	 */
	int numberOfSegments = 0;
	for(int i = 0; i<streamlineCount; ++i)
	{
		numberOfSegments += int(log2(coordinates.at(i).size()/3))+1;
	}

	lineCurvatures = std::vector<Eigen::VectorXd>(numberOfSegments);
	lineSegments.resize(numberOfSegments);
	/*
	 * make segmentation based on signature for all streamlines. Similar to signature-based sampling
	 */
	int accumulated = 0;
	SimilarityDistance::computeSegments(lineSegments, lineCurvatures, segmentOfStreamlines,
					accumulated, coordinates, pointToSegment);

	std::cout << "Distance matrix computation enabled? 1.Yes, 2.No. (distance matrix on segments is time consuming,"
			<< " and recommended not to be enabled." << std::endl;
	int distanceMatrixOption;
	std::cin >> distanceMatrixOption;
	assert(distanceMatrixOption==1 || distanceMatrixOption==2);

	bool distanceMatrixEnabled = (distanceMatrixOption==1);

	std::vector<double> segmentScalars(numberOfSegments);

	/* use distanceMatrix and max_heap to find KNN. Slow but accurate and requires less coding */
	if(distanceMatrixEnabled)
	{
		/* compute distance matrix based on MCP distance among segments */
		Eigen::MatrixXd distanceMatrix(numberOfSegments, numberOfSegments);

	#pragma omp parallel for schedule(static) num_threads(8)
		for(int i=0; i<numberOfSegments; ++i)
		{
			for(int j=0; j<numberOfSegments; ++j)
			{
				if(i==j)
					continue;
				distanceMatrix(i,j) = SimilarityDistance::getMcpDistance(lineSegments[i], lineSegments[j]);
			}
		}

		std::vector<int> segmentToStreamline(numberOfSegments);

		std::cout << distanceMatrix.rows() << " " << distanceMatrix.cols() << std::endl;

		getSegmentToStreamlineVec(segmentToStreamline, segmentOfStreamlines);

		getScalarsOnSegments(distanceMatrix, segmentScalars, lineCurvatures, segmentToStreamline);
	}

	/* Use spatial bining to find closest point, and keep checking segments related to it.
	 * Efficient and could save time but coding part is laborious
	 */
	else
	{
		assignPointsToBins();

		/* use while loop to find K nearest segments for chi-test calculation */
		processWithBinsOnSegments(pointToSegment, segmentScalars, lineCurvatures);
	}

	VTKWritter::printStreamlineScalarsOnSegments(coordinates,datasetName, streamlineVertexCount, lineSegments,
			segmentScalars);
}


/* spatial bining for all the points in the grid */
void LocalScalar::assignPointsToBins()
{
	/* get x y z coordinate limits */
	Eigen::VectorXd line;
	int lineSize;
	for(int i=0; i<streamlineCount; ++i)
	{
		line = coordinates[i];
		lineSize = line.size()/3;
		for(int j=0; j<lineSize; ++j)
		{
			for(int k=0; k<3; ++k)
			{
				range[k].inf = std::min(range[k].inf, line(3*j+k));
				range[k].sup = std::max(range[k].sup, line(3*j+k));
			}
		}
	}

	if(range[2].sup-range[2].inf<=1.0E-3)
	{
		float y_to_x = (range[1].sup-range[1].inf)/(range[0].sup-range[0].inf);
		X_RESOLUTION = 101, Y_RESOLUTION = y_to_x*X_RESOLUTION, Z_RESOLUTION = 1;
		X_STEP = (range[0].sup-range[0].inf)/float(X_RESOLUTION-1);
		Y_STEP = (range[1].sup-range[1].inf)/float(Y_RESOLUTION-1);
		Z_STEP = 1.0E5;
	}
	else
	{
		is3D = true;
		float y_to_x = (range[1].sup-range[1].inf)/(range[0].sup-range[0].inf);
		float z_to_x = (range[2].sup-range[2].inf)/(range[0].sup-range[0].inf);
		X_RESOLUTION = 101, Y_RESOLUTION = y_to_x*X_RESOLUTION, Z_RESOLUTION = z_to_x*X_RESOLUTION;
		X_STEP = (range[0].sup-range[0].inf)/float(X_RESOLUTION-1);
		Y_STEP = (range[1].sup-range[1].inf)/float(Y_RESOLUTION-1);
		Z_STEP = (range[2].sup-range[2].inf)/float(Z_RESOLUTION-1);
	}

	const int& totalBinSize = X_RESOLUTION*Y_RESOLUTION*Z_RESOLUTION;
	const int& xy_multiplication = X_RESOLUTION*Y_RESOLUTION;
	spatialBins.clear();
	spatialBins.resize(totalBinSize);

	int x_grid, y_grid, z_grid;
	double x, y, z;

	int pointID = 0;
	for(int i=0; i<streamlineCount; ++i)
	{
		line = coordinates[i];
		lineSize = line.size()/3;
		for(int j=0; j<lineSize; ++j)
		{
			// assign vertex into spatial bins w.r.t. spatial coordinates
			x = line(3*j), y = line(3*j+1), z = line(3*j+2);
			x_grid = (x-range[0].inf)/X_STEP;
			y_grid = (y-range[1].inf)/Y_STEP;
			z_grid = (z-range[2].inf)/Z_STEP;

			spatialBins[z_grid*xy_multiplication+X_RESOLUTION*y_grid+x_grid].push_back(streamPoint(i,pointID+j));
		}
		pointID+=lineSize;
	}
	std::cout << "Spatial bining process is done!" << std::endl;
}


/* get scalar value based on KNN */
void LocalScalar::getScalarsOnSegments(const Eigen::MatrixXd& distanceMatrix, std::vector<double>& segmentScalars,
		const std::vector<Eigen::VectorXd>& lineCurvatures, std::vector<int>& segmentToStreamline)
{
	std::cout << "Scalar calculation on segments start..." << std::endl;
	const int& numberOfSegments = distanceMatrix.rows();
#pragma omp parallel for schedule(static) num_threads(8)
	for(int i=0; i<numberOfSegments; ++i)
	{
		std::priority_queue<MinimalDist, std::vector<MinimalDist>, CompareDistRecord> max_heap;
		const Eigen::VectorXd& firstCurvatures = lineCurvatures[i];

		const int& targetStream = segmentToStreamline[i];
		for(int j=0; j<numberOfSegments; ++j)
		{
			if(i==j)
				continue;
			if(segmentToStreamline[j]==targetStream)
				continue;
			max_heap.push(MinimalDist(distanceMatrix(i,j),j));
			if(max_heap.size()>kNeighborNumber)
				max_heap.pop();
		}

		MinimalDist top;

		double scalar = 0;
		int index;
		while(!max_heap.empty())
		{
			top = max_heap.top();
			max_heap.pop();
			index = top.index;
			scalar += SimilarityDistance::getChiTestPair(firstCurvatures, lineCurvatures[index]);
		}
		segmentScalars[i] = scalar/kNeighborNumber;
	}

	std::cout << "Scalar calculation on segments finish!" << std::endl;

}


/* segment->streamline */
void LocalScalar::getSegmentToStreamlineVec(std::vector<int>& segmentToStreamline,
		const std::vector<std::vector<int> >& segmentOfStreamlines)
{
	const int& numberOfSegments = segmentToStreamline.size();
#pragma omp parallel for schedule(static) num_threads(8)
	for(int i=0; i<streamlineCount; ++i)
	{
		for(int j=0; j<segmentOfStreamlines[i].size(); ++j)
		{
			segmentToStreamline[segmentOfStreamlines[i][j]] = i;
		}
	}
}


/* use while loop to find K nearest segments for chi-test calculation */
void LocalScalar::processWithBinsOnSegments(const std::vector<int>& pointToSegment, std::vector<double>& segmentScalars,
		const std::vector<Eigen::VectorXd>& lineCurvatures)
{
	std::cout << "Process on segments w.r.t. spatial bining starts..." << std::endl;
	const int& numberOfSegments = segmentScalars.size();

	/* find segment->point mapping */
	std::vector<std::vector<int> > segmentToPoint(numberOfSegments);
	for(int i=0; i<streamlineVertexCount; ++i)
		segmentToPoint[pointToSegment[i]].push_back(i);

#pragma omp parallel for schedule(static) num_threads(8)
	for(int i=0; i<numberOfSegments; ++i)
	{
		int current = kNeighborNumber;
		const std::vector<int>& pointArray = segmentToPoint[i];
		const int& arraySize = pointArray.size();
		const Eigen::VectorXd& targetCurvature = lineCurvatures[i];

		/*
		 * only concerned about center ID now. Cannot traverse from one by one
		 */
		int centerID = arraySize/2;

		int encompass = 1;

		int centerX = (vertexVec[centerID](0)-range[0].inf)/X_STEP;
		int centerY = (vertexVec[centerID](1)-range[1].inf)/Y_STEP;
		int centerZ = (vertexVec[centerID](2)-range[2].inf)/Z_STEP;

		Eigen::Vector3d& targetVertex = vertexVec[centerID];

		std::unordered_map<int, bool> binChosen;
		std::unordered_map<int, bool> segmentChosen;

		const int& xy_multiplication = X_RESOLUTION*Y_RESOLUTION;
		int binNumber, streamID;

		double summation = 0;
		while(current>0)
		{
			for(int z=std::max(centerZ-encompass, 0); z<=std::min(Z_RESOLUTION-1, centerZ+encompass)&&current>0; ++z)
			{
				for(int y=std::max(centerY-encompass, 0); y<=std::min(Y_RESOLUTION-1, centerY+encompass)&&current>0; ++y)
				{
					for(int x=std::max(centerX-encompass, 0); x<=std::min(X_RESOLUTION-1, centerX+encompass)&&current>0; ++x)
					{
						binNumber = xy_multiplication*z+X_RESOLUTION*y+x;
						if(binChosen.find(binNumber)==binChosen.end())
						{
							std::vector<streamPoint>& bins = spatialBins[binNumber];
							std::vector<MinimalDist> pointIDArray;
							for(int k=0; k<bins.size(); ++k)
							{
								streamID = bins[k].vertexID;
								if(streamID==centerID)	// same vertex
									continue;
								if(vertexToLine[streamID]==vertexToLine[centerID])	// on same streamlines
									continue;
								if(pointToSegment[streamID]==pointToSegment[centerID]) // on same segments
									continue;
								pointIDArray.push_back(MinimalDist((targetVertex-vertexVec[streamID]).norm(), streamID));
							}
							// sort distance from minimal to maximal
							std::sort(pointIDArray.begin(), pointIDArray.end(), CompareDistRecord());

							for(int k=0; k<pointIDArray.size()&&current>0; ++k)
							{
								int segmentID = pointToSegment[pointIDArray[k].index];
								if(segmentChosen.find(segmentID)==segmentChosen.end())
								{
									summation+=SimilarityDistance::getChiTestPair(targetCurvature, lineCurvatures[segmentID]);
									segmentChosen[segmentID] = true;
									--current;
								}
							}
							binChosen[binNumber] = true;
						}
					}
				}
			}

			if(current>0)
				++encompass;
		}
		segmentScalars[i] = summation/kNeighborNumber;
	}
}


/*
 * follow a 3-1-3 alignment, judge point distance with lnk/(lnk+1) mapping such that
 * 0, infinite large--->1 (stronger separation), 1--->0 (weaker separation)
 */
void LocalScalar::processAlignmentByPointWise()
{
	assignPointsToBins();

	// point-wise scalar value calculation on streamlines
	segmentScalars.resize(streamlineVertexCount);

	std::cout << "Choose neighborhood search strategy: 1.left and right closest (2D only), 2. KNN, "
			<< "3.search along 8 directions on the plane (3D only), 4.search along directions of 9 voxels (3D only)."
			<< std::endl;
	int directionOption;
	std::cin >> directionOption;
	assert(directionOption>=1 && directionOption<=4);
	DirectionSearch direction = static_cast<DirectionSearch>(directionOption);

	/* how many directions are used */
	if(directionOption==3)
	{
		std::cout << "Choose how many directions want to select? 8 or 12." << std::endl;
		std::cin >> directionNumbers;
		assert(directionNumbers==8 || directionNumbers==12);
	}

	std::cout << "Whether candidates on same streamlines enabled? 1.Yes, 2.No." << std::endl;
	int sameLineOption;
	std::cin >> sameLineOption;
	assert(sameLineOption==1 || sameLineOption==2);
	bool sameLineEnabled = (sameLineOption==1);

	const int& TOTALSIZE = 7;

	struct timeval start, end;
	double timeTemp;
	gettimeofday(&start, NULL);

#pragma omp parallel for schedule(static) num_threads(8)
	for(int i=0; i<streamlineCount; ++i)
	{
		const std::vector<int>& vertexArray = streamlineToVertex[i];
		const int& arraySize = vertexArray.size();

		for(int j=0; j<arraySize; ++j)
		{
			std::vector<int> closestPoint;
			getScalarsOnPointWise(direction, sameLineEnabled, j, vertexArray, closestPoint);

			// check the separation w.r.t. ratio of distance
			if(separationMeasureOption==1)
				getScalarsFromNeighbors(j, vertexArray, TOTALSIZE, closestPoint, segmentScalars[vertexArray[j]]);

			// check the separation w.r.t. standard deviation of point array
			else if(separationMeasureOption==2)
				getScalarsFromDeviation(j, vertexArray, TOTALSIZE, closestPoint, segmentScalars[vertexArray[j]]);
		}
	}

	// record the events and time spent
	std::vector<string> events;
	std::vector<double> timeSpent;

	// record the time for local scalar value calculation
	gettimeofday(&end, NULL);
	timeTemp = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
	std::cout << "The time for scalar value calculation is " << timeTemp << " seconds!" << std::endl;

	events.push_back("local scalar calculation");
	timeSpent.push_back(timeTemp);

	VTKWritter::printStreamlineScalarsOnSegments(coordinates,datasetName, streamlineVertexCount, segmentScalars,
			"noSmooth");

	// select the option whether to enable automaticly adaptive bandwidth
	const double& r = get_kde_bandwidth();

	/* perform the smoothing the points along the lines to re-sample the voxel information */
	performSmoothingOnLine(r);

	/* print the visualization for the scalar values after smoothing operation */
	VTKWritter::printStreamlineScalarsOnSegments(coordinates,datasetName, streamlineVertexCount, segmentScalars,
			"Smooth");

	std::vector<double> voxelScalars;
	/* sample on the voxels from discrete 3D points */
	sampleOnVoxels(r, voxelScalars);

	// finish recording the time
	gettimeofday(&end, NULL);
	timeTemp = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
	std::cout << "The time for smoothing and voxelization is " << timeTemp << " seconds!" << std::endl;

	events.push_back("smoothing and voxelization calculation");
	timeSpent.push_back(timeTemp);

	VTKWritter::printVolumeScalars(datasetName, voxelScalars, range, X_GRID_RESOLUTION, Y_GRID_RESOLUTION,
			Z_GRID_RESOLUTION, X_GRID_STEP, Y_GRID_STEP, Z_GRID_STEP);

	recordTime(events, timeSpent);
}


/* record time spent in the readme in case they are required again */
void LocalScalar::recordTime(const std::vector<string>& events, std::vector<double>& timeSpent)
{
	// find the last '/' in the data set name
	int lastSlash = -1;
	for(int i=datasetName.size()-1; i>=0; --i)
	{
		if(datasetName[i]=='/')
		{
			lastSlash = i;
			break;
		}
	}
	string _data_set = datasetName.substr(lastSlash+1);

	// record them into the readme
	std::ofstream readme(datasetName.substr(0,lastSlash+1)+"README", ios::out|ios::app);
	if(readme.fail())
	{
		std::cout << "Error for creating the file for output!" << std::endl;
		exit(1);
	}
	readme << "--------------------------------------------------------------------------------------" << std::endl;
	readme << "For " << _data_set << " [" << X_GRID_RESOLUTION << "x" << Y_GRID_RESOLUTION << "x" 
		   << Z_GRID_RESOLUTION << "]:" << std::endl;

	for(int i=0; i<events.size(); ++i)
		readme << "The time for " << events[i] << " takes " << timeSpent[i] << " seconds." << std::endl;
	readme.close();
}


// compute the scalar value on point-wise element of the streamline
void LocalScalar::getScalarsOnPointWise(const DirectionSearch& directionOption, const bool& sameLineEnabled, const int& j,
		const std::vector<int>& vertexArray, std::vector<int>& pointCandidate)
{
	int current;
	int targetID = vertexArray[j];

	Eigen::Vector3d& targetVertex = vertexVec[targetID];

	switch(directionOption)
	{
	case 1: // left and right neighbors in 2D space
		searchClosestThroughDirections(sameLineEnabled, j, vertexArray, pointCandidate);
		break;

	case 2: // find KNN neighboring points, e.g., k==8
		searchKNNPointByBining(sameLineEnabled, j, vertexArray, pointCandidate);
		break;

	case 3: // find k neighboring points by direction-based search, e.g., k==8
		searchNeighborThroughSeparateDirections(sameLineEnabled, j, vertexArray, pointCandidate);
		break;

	default:
		exit(1);
	}
}


// find KNN closest point in the spatial bining
void LocalScalar::searchKNNPointByBining(const bool& sameLineEnabled, const int& j,
		const std::vector<int>& vertexArray, std::vector<int>& pointCandidate)
{
	int current = kNeighborNumber;
	int targetID = vertexArray[j];
	int centerX = (vertexVec[targetID](0)-range[0].inf)/X_STEP;
	int centerY = (vertexVec[targetID](1)-range[1].inf)/Y_STEP;
	int centerZ = (vertexVec[targetID](2)-range[2].inf)/Z_STEP;
	const int& xy_multiplication = X_RESOLUTION*Y_RESOLUTION;

	Eigen::Vector3d& targetVertex = vertexVec[targetID];

	int encompass = 1;

	std::unordered_map<int, bool> binChosen;
	std::unordered_map<int, bool> streamChosen;

	int binNumber, streamID;

	while(current>0)
	{
		for(int z=std::max(centerZ-encompass, 0); z<=std::min(Z_RESOLUTION-1, centerZ+encompass)&&current>0; ++z)
		{
			for(int y=std::max(centerY-encompass, 0); y<=std::min(Y_RESOLUTION-1, centerY+encompass)&&current>0; ++y)
			{
				for(int x=std::max(centerX-encompass, 0); x<=std::min(X_RESOLUTION-1, centerX+encompass)&&current>0; ++x)
				{
					binNumber = xy_multiplication*z+X_RESOLUTION*y+x;
					if(binChosen.find(binNumber)==binChosen.end())
					{
						std::vector<streamPoint>& bins = spatialBins[binNumber];
						std::vector<MinimalDist> pointIDArray;
						for(int k=0; k<bins.size(); ++k)
						{
							streamID = bins[k].vertexID;
							if(streamID==targetID)	// same vertex
								continue;
							if(vertexToLine[streamID]==vertexToLine[targetID])	// on same streamlines
								continue;
							pointIDArray.push_back(MinimalDist((targetVertex-vertexVec[streamID]).norm(), streamID));
						}
						// sort distance from minimal to maximal
						std::sort(pointIDArray.begin(), pointIDArray.end(), CompareDistRecord());

						for(int k=0; k<pointIDArray.size()&&current>0; ++k)
						{
							int streamID = pointIDArray[k].index;
							if(sameLineEnabled)
							{
								pointCandidate.push_back(streamID);
							}
							else
							{
								if(streamChosen.find(vertexToLine[streamID])==streamChosen.end())
								{
									pointCandidate.push_back(streamID);
									streamChosen[vertexToLine[streamID]] = true;
								}
							}
							--current;
						}
						binChosen[binNumber] = true;
					}
				}
			}
		}

		if(current>0)
			++encompass;
	}
}


// search through two directions (just left and right) for 2D data set
void LocalScalar::searchClosestThroughDirections(const bool& sameLineEnabled, const int& j,
		const std::vector<int>& vertexArray, std::vector<int>& pointCandidate)
{
	int targetID = vertexArray[j];
	int centerX = (vertexVec[targetID](0)-range[0].inf)/X_STEP;
	int centerY = (vertexVec[targetID](1)-range[1].inf)/Y_STEP;
	int centerZ = (vertexVec[targetID](2)-range[2].inf)/Z_STEP;
	const int& xy_multiplication = X_RESOLUTION*Y_RESOLUTION;

	Eigen::Vector3d& center = vertexVec[targetID];
	// find basis segment vector along the flow direction
	Eigen::Vector3d tangential;
	if(j==0)
		tangential = vertexVec[j+1]-vertexVec[j];
	else if(j==vertexArray.size()-1)
		tangential = vertexVec[j]-vertexVec[j-1];
	else
		tangential = vertexVec[j+1]-vertexVec[j-1];
	/* Already known z == 0, so we can definitely know the left and right direction vectors
	 * Assume tangential is (a,b,0), left is (-b,a,0), right is (b,-a,0).
	 */
	tangential/=tangential.norm();
	Eigen::Vector3d searchDirection(-tangential(1),tangential(0),tangential(2));

	double searchingStep = std::min(X_STEP, Y_STEP)*0.1;

	// closest threshold
	searchThreshold=3.0*searchingStep;

	Eigen::Vector3d currentPos = center;
	int currentBin, streamID;

	while(1)
	{
		currentPos += 2.0*searchingStep*searchDirection;

		int x = (currentPos(0)-range[0].inf)/X_STEP;
		int y = (currentPos(1)-range[1].inf)/Y_STEP;
		int z = (currentPos(2)-range[2].inf)/Z_STEP;

		// already exit from the domain
		if(x<0 || x>=X_RESOLUTION || y<0 || y>=Y_RESOLUTION || z<0 || z>=Z_RESOLUTION)
			break;

		currentBin = xy_multiplication*z+X_RESOLUTION*y+x;

		std::vector<streamPoint>& bins = spatialBins[currentBin];
		std::vector<MinimalDist> pointIDArray;

		// find the points that are not the same vertex and not on the same streamline
		for(int k=0; k<bins.size(); ++k)
		{
			streamID = bins[k].vertexID;
			if(streamID==targetID)	// same vertex
				continue;
			if(vertexToLine[streamID]==vertexToLine[targetID])	// on same streamlines
				continue;
			Eigen::Vector3d zDirection = tangential.cross(vertexVec[streamID]-center);
			if(zDirection(2)<0)	// search along left directions
				continue;
			pointIDArray.push_back(MinimalDist((currentPos-vertexVec[streamID]).norm(), streamID));
		}
		// sort distance from minimal to maximal
		std::sort(pointIDArray.begin(), pointIDArray.end(), CompareDistRecord());

		// don't handle if no points are nearly enough
		if(pointIDArray.empty() || pointIDArray[0].dist>searchThreshold)
			continue;

		// otherwise, the first vertex should be close enough for the direction
		pointCandidate.push_back(pointIDArray[0].index);
		break;
	}

	/*
	 * Along right directions search for the closest right
	 */
	searchDirection = -searchDirection;
	currentPos = center;
	while(1)
	{
		currentPos += searchingStep*searchDirection;

		int x = (currentPos(0)-range[0].inf)/X_STEP;
		int y = (currentPos(1)-range[1].inf)/Y_STEP;
		int z = (currentPos(2)-range[2].inf)/Z_STEP;

		if(x<0 || x>=X_RESOLUTION || y<0 || y>=Y_RESOLUTION || z<0 || z>=Z_RESOLUTION)
			break;

		currentBin = xy_multiplication*z+X_RESOLUTION*y+x;

		std::vector<streamPoint>& bins = spatialBins[currentBin];
		std::vector<MinimalDist> pointIDArray;
		for(int k=0; k<bins.size(); ++k)
		{
			streamID = bins[k].vertexID;
			if(streamID==targetID)	// same vertex
				continue;
			if(vertexToLine[streamID]==vertexToLine[targetID])	// on same streamlines
				continue;
			Eigen::Vector3d zDirection = tangential.cross(vertexVec[streamID]-center);
			if(zDirection(2)>0)
				continue;
			pointIDArray.push_back(MinimalDist((currentPos-vertexVec[streamID]).norm(), streamID));
		}
		// sort distance from minimal to maximal
		std::sort(pointIDArray.begin(), pointIDArray.end(), CompareDistRecord());

		// don't handle if no points are nearly enough
		if(pointIDArray.empty() || pointIDArray[0].dist>searchThreshold)
			continue;

		// since it's already sorted, then should push inside the first candidate
		pointCandidate.push_back(pointIDArray[0].index);
		break;
	}
}


// get scalar value based on neighborhood
void LocalScalar::getScalarsFromNeighbors(const int& j, const std::vector<int>& vertexArray,
		const int& TOTALSIZE, const std::vector<int>& closestPoint, double& scalar)
{
	std::vector<int> first(TOTALSIZE);
	const int& arraySize = vertexArray.size();
	if(arraySize<TOTALSIZE)
	{
		scalar = 0;
		return;
	}

	const int& HALF = TOTALSIZE/2;
	double summation = 0;
	if(j<=HALF)
	{
		for(int k=0; k<TOTALSIZE; ++k)
		{
			first[k] = vertexArray[0]+k;
		}
	}
	else if(j>=arraySize-HALF-1)
	{
		for(int k=0; k<TOTALSIZE; ++k)
		{
			first[k] = vertexArray[arraySize-TOTALSIZE+k];
		}
	}
	else
	{
		for(int k=-HALF; k<=HALF; ++k)
		{
			first[k+HALF] = vertexArray[j+k];
		}
	}
	int candidate, candSize;
	std::vector<int> candStreamline;

	const int& firstBegin = first[0];
	const int& firstEnd = first.back();

	double ratio, start_dist, end_dist, mid_dist;
	int effective = 0;
	for(int i=0; i<closestPoint.size(); ++i)
	{
		candidate = closestPoint[i];
		candStreamline = streamlineToVertex[vertexToLine[candidate]];
		candSize = candStreamline.size();
		if(candSize<TOTALSIZE)
			return;

		if(candidate-candStreamline[0]<=HALF)
		{
			end_dist = (vertexVec[firstEnd]-vertexVec[candStreamline[0]+TOTALSIZE-1]).norm();
			start_dist = (vertexVec[firstBegin]-vertexVec[candStreamline[0]]).norm();
			mid_dist = (vertexVec[firstBegin+HALF]-vertexVec[candStreamline[0]+HALF]).norm();
		}
		else if(candStreamline.back()-candidate<=HALF)
		{
			end_dist = (vertexVec[firstEnd]-vertexVec[candStreamline.back()]).norm();
			start_dist = (vertexVec[firstBegin]-vertexVec[candStreamline[candSize-TOTALSIZE]]).norm();
			mid_dist = (vertexVec[firstBegin+HALF]-vertexVec[candStreamline[candSize-HALF-1]]).norm();
		}
		else
		{
			end_dist = (vertexVec[firstEnd]-vertexVec[candidate+HALF]).norm();
			start_dist = (vertexVec[firstBegin]-vertexVec[candidate-HALF]).norm();
			mid_dist = (vertexVec[firstBegin+HALF]-vertexVec[candidate]).norm();
		}

		if(mid_dist<1.0E-6)
			continue;

		// ratio = std::max(end_dist/mid_dist, start_dist/mid_dist);
		// ratio = end_dist/start_dist;
		ratio = end_dist*start_dist/mid_dist/mid_dist;

		// use the max of divergence and convergence rate, so only consider 0 < ratio < 1
		if(useMaxRatio)
		{
			if(ratio>0 && ratio<1)
				ratio = 1.0/ratio;
		}

		summation+=ratio;
		++effective;
	}

	/* can not find appropriate neighborhood lines */
	if(effective==0)
	{
		scalar = 0;
		return;
	}

	summation/=effective;

	if(std::isnan(summation))
	{
		std::cout << "Error for nan!" << std::endl;
		exit(1);
	}

	ratio = abs(log2(summation));
	scalar = ratio/(ratio+1);
}


// get scalar value based on difference deviation of point-wise distance
void LocalScalar::getScalarsFromDeviation(const int& j, const std::vector<int>& vertexArray,
		const int& TOTALSIZE, const std::vector<int>& closestPoint, double& scalar)
{
	std::vector<int> first(TOTALSIZE);
	const int& arraySize = vertexArray.size();
	if(arraySize<TOTALSIZE)
	{
		scalar = 0;
		return;
	}

	const int& HALF = TOTALSIZE/2;
	double summation = 0;
	if(j<=HALF)
	{
		for(int k=0; k<TOTALSIZE; ++k)
		{
			first[k] = vertexArray[0]+k;
		}
	}
	else if(j>=arraySize-HALF-1)
	{
		for(int k=0; k<TOTALSIZE; ++k)
		{
			first[k] = vertexArray[arraySize-TOTALSIZE+k];
		}
	}
	else
	{
		for(int k=-HALF; k<=HALF; ++k)
		{
			first[k+HALF] = vertexArray[j+k];
		}
	}
	int candidate, candSize;
	std::vector<int> candStreamline;

	const int& firstBegin = first[0];
	const int& firstEnd = first.back();
	double dist_mean, dist_deviation, min_dist, max_dist, point_dist;
	int effective = 0;
	for(int i=0; i<closestPoint.size(); ++i)
	{
		candidate = closestPoint[i];
		candStreamline = streamlineToVertex[vertexToLine[candidate]];
		candSize = candStreamline.size();
		if(candSize<TOTALSIZE)
			return;

		dist_mean = dist_deviation = 0;
		min_dist = DBL_MAX;
		max_dist = -1.0;

		if(candidate-candStreamline[0]<=HALF)
		{
			for(int k=0; k<TOTALSIZE; ++k)
			{
				point_dist = (vertexVec[firstBegin+k]-vertexVec[candStreamline[0]+k]).norm();
				dist_mean += point_dist;
				dist_deviation += point_dist*point_dist;
				min_dist = std::min(min_dist, point_dist);
				max_dist = std::max(max_dist, point_dist);
			}
		}
		else if(candStreamline.back()-candidate<=HALF)
		{
			for(int k=0; k<TOTALSIZE; ++k)
			{
				point_dist = (vertexVec[firstBegin+k]-vertexVec[candStreamline.back()-TOTALSIZE+k+1]).norm();
				dist_mean += point_dist;
				dist_deviation += point_dist*point_dist;
				min_dist = std::min(min_dist, point_dist);
				max_dist = std::max(max_dist, point_dist);
			}
		}
		else
		{
			for(int k=0; k<TOTALSIZE; ++k)
			{
				point_dist = (vertexVec[firstBegin+k]-vertexVec[candidate-HALF+k]).norm();
				dist_mean += point_dist;
				dist_deviation += point_dist*point_dist;
				min_dist = std::min(min_dist, point_dist);
				max_dist = std::max(max_dist, point_dist);
			}
		}

		dist_deviation = (dist_deviation - dist_mean*dist_mean/double(TOTALSIZE))/double(TOTALSIZE-1);

		if(dist_deviation<0)
		{
			continue;
		}

		dist_deviation = sqrt(dist_deviation);

		/* whether use normalization */
		if(useNormalizedDevi)
			dist_deviation/=max_dist-min_dist;

		summation+=dist_deviation;
		++effective;
	}

	/* can not find appropriate neighborhood lines */
	if(effective==0)
	{
		scalar = 0;

		return;
	}

	summation/=effective;

	scalar = summation;
}


// search through eight perpendicular directions and each find the streamline passing through the point
void LocalScalar::searchNeighborThroughSeparateDirections(const bool& sameLineEnabled, const int& j,
		const std::vector<int>& vertexArray, std::vector<int>& pointCandidate)
{
	// compute the threshold to tell whether points are close to search directions or not
	if(Z_STEP>1.0E4)
	{
		searchThreshold = sqrt(X_STEP*X_STEP+Y_STEP*Y_STEP);
	}
	else
	{
		searchThreshold = sqrt(X_STEP*X_STEP+Y_STEP*Y_STEP+Z_STEP*Z_STEP);
	}
	// search step size
	double searchingStep = searchThreshold*0.1;

	// closest threshold
	searchThreshold*=0.3;

	// find the perpendicular plane w.r.t. the tangent direction
	int targetID = vertexArray[j];
	int centerX = (vertexVec[targetID](0)-range[0].inf)/X_STEP;
	int centerY = (vertexVec[targetID](1)-range[1].inf)/Y_STEP;
	int centerZ = (vertexVec[targetID](2)-range[2].inf)/Z_STEP;
	const int& xy_multiplication = X_RESOLUTION*Y_RESOLUTION;

	Eigen::Vector3d& center = vertexVec[targetID];
	// find basis segment vector along the flow direction
	Eigen::Vector3d tangential;
	if(j==0)
		tangential = vertexVec[j+1]-vertexVec[j];
	else if(j==vertexArray.size()-1)
		tangential = vertexVec[j]-vertexVec[j-1];
	else
		tangential = vertexVec[j+1]-vertexVec[j-1];
	/* Already known z == 0, so we can definitely know the left and right direction vectors
	 * Assume tangential is (a,b,0), left is (-b,a,0), right is (b,-a,0).
	 */
	tangential/=tangential.norm();

	// find eight direction vectors separated by 45 degrees each other
	std::vector<Eigen::Vector3d> directionVectors;
	findDirectionVectors(tangential, center, directionVectors);

	Eigen::Vector3d current;
	int currentBin, streamID;

	std::unordered_map<int,int> streamlineChosen;

	for(int i=0; i<directionVectors.size(); ++i)
	{
		current = center;
		std::vector<MinimalDist> pointIDArray;

		// how many steps already taken
		int count = 0;
		while(1)
		{
			// this is to avoid searching neigbhorhood that is too far away from you
			if(count>=maxStep)
				break;

			current += 2.0*searchingStep*directionVectors[i];

			int x = (current(0)-range[0].inf)/X_STEP;
			int y = (current(1)-range[1].inf)/Y_STEP;
			int z = (current(2)-range[2].inf)/Z_STEP;

			// exit the search boundary, terminate the loop
			if(x<0 || x>=X_RESOLUTION || y<0 || y>=Y_RESOLUTION || z<0 || z>=Z_RESOLUTION)
				break;

			currentBin = xy_multiplication*z+X_RESOLUTION*y+x;

			pointIDArray.clear();

			std::vector<streamPoint>& bins = spatialBins[currentBin];
			for(int k=0; k<bins.size(); ++k)
			{
				streamID = bins[k].vertexID;
				if(streamID==targetID)	// same vertex
					continue;
				if(vertexToLine[streamID]==vertexToLine[targetID])	// on same streamlines
					continue;
				pointIDArray.push_back(MinimalDist((current-vertexVec[streamID]).norm(), streamID));
			}
			// sort distance from minimal to maximal
			std::sort(pointIDArray.begin(), pointIDArray.end(), CompareDistRecord());
			if(!pointIDArray.empty())	// not empty
			{
				streamID = pointIDArray[0].index;
				if(pointIDArray[0].dist<=searchThreshold) // smallest is within range, accept it!
				{
					if(streamlineChosen.count(vertexToLine[streamID])==0) // its streamline is not included yet
					{
						pointCandidate.push_back(streamID); // add it
						++streamlineChosen[vertexToLine[streamID]]; // update the hash map
						break;	// this direction search is done
					}
				}
			}
			++count;
		}
	}
}


// search through all nine voxels. If one voxel has empty, will search into other voxel along that direction
void LocalScalar::searchNeighborThroughVoxels(const bool& sameLineEnabled, const int& j,
		const std::vector<int>& vertexArray, std::vector<int>& pointCandidate)
{

}


// given a planar, get eight normalized directions separated by 45 degrees
void LocalScalar::findDirectionVectors(const Eigen::Vector3d& tangential, const Eigen::Vector3d& center,
		std::vector<Eigen::Vector3d>& directionVectors)
{
	// given any initial directions
	Eigen::Vector3d temp(1.0, 1.0, 1.0);
	temp(2) = -(tangential(0)+tangential(1));

	// normalize the direction
	temp/=temp.norm();
	directionVectors.push_back(temp);
	// negative direction is also accepted
	directionVectors.push_back(-temp);

	// double angles[] = {M_PI/6.0, M_PI/3.0, M_PI/2.0, M_PI/3.0*2.0, M_PI/6.0*5.0};

	std::vector<double> angles;

	const int& intervalNumber = (directionNumbers-2)/2+1;
	const double& interval = M_PI/intervalNumber;

	for(int i=1; i<intervalNumber; ++i)
	{
		angles.push_back(interval*i);
	}

	for(int i=0; i<angles.size(); ++i)
	{
		findDirectionsToAngles(tangential, directionVectors[0], angles[i], directionVectors);
	}
}


// use quadratic system to get the two directions w.r.t. reference
void LocalScalar::findDirectionsToAngles(const Eigen::Vector3d& tangential, const Eigen::Vector3d& reference,
		const double& angle, std::vector<Eigen::Vector3d>& directionVectors)
{
	// I use matlab to directly get the numerical solutions
	/*
	 * tangential = (a, b, c), reference = (d, e, f), angle = pi/4.0
	 * should solve the nonlinear euqations
	 * ax + by + cz == 0
	 * dx + ey + fz - cos(angle) == 0
	 * xx + yy + zz -1 == 0
	 */

	double a = tangential(0), b = tangential(1), c = tangential(2);
	double d = reference(0), e = reference(1), f = reference(2);
	const double& cosVal = cos(angle);

	Eigen::Vector3d solution;

	// get first direction
	solution(2) = (a*e*sqrt(- a*a*cosVal*cosVal + a*a*e*e + a*a*f*f - 2*a*b*d*e - 2*a*c*d*f - b*b*cosVal*cosVal
			+ b*b*d*d + b*b*f*f - 2*b*c*e*f - c*c*cosVal*cosVal + c*c*d*d + c*c*e*e)
			- b*d*sqrt(- a*a*cosVal*cosVal + a*a*e*e + a*a*f*f - 2*a*b*d*e - 2*a*c*d*f - b*b*cosVal*cosVal
		    + b*b*d*d + b*b*f*f - 2*b*c*e*f - c*c*cosVal*cosVal + c*c*d*d + c*c*e*e) + a*a*cosVal*f
			+ b*b*cosVal*f - a*c*cosVal*d - b*c*cosVal*e)/(a*a*cosVal*cosVal - a*a*e*e + 2*a*b*d*e
			+ b*b*cosVal*cosVal - b*b*d*d);

	// get the direction search issues
	if(a*e-b*d==0)
	{
		return;
	}
	solution(0) = -(b*cosVal - b*f*solution(2) + c*e*solution(2))/(a*e - b*d);
	solution(1) = (a*cosVal - a*f*solution(2) + c*d*solution(2))/(a*e - b*d);
	directionVectors.push_back(solution);
	// make sure the solution we get is 100% correct
	assert(abs(solution.dot(reference)-cosVal)<1.0E-6);
	assert(abs(solution.dot(tangential))<1.0E-6);
	assert(abs(solution.norm()-1.0)<1.0E-6);

	// get second direction
	solution(2) =  -(a*e*sqrt(- a*a*cosVal*cosVal + a*a*e*e + a*a*f*f - 2*a*b*d*e - 2*a*c*d*f - b*b*cosVal*cosVal
			+ b*b*d*d + b*b*f*f - 2*b*c*e*f - c*c*cosVal*cosVal + c*c*d*d + c*c*e*e)
			- b*d*sqrt(- a*a*cosVal*cosVal + a*a*e*e + a*a*f*f - 2*a*b*d*e - 2*a*c*d*f - b*b*cosVal*cosVal
			+ b*b*d*d + b*b*f*f - 2*b*c*e*f - c*c*cosVal*cosVal + c*c*d*d + c*c*e*e)
			- a*a*cosVal*f - b*b*cosVal*f + a*c*cosVal*d + b*c*cosVal*e)/(a*a*cosVal*cosVal
			- a*a*e*e + 2*a*b*d*e + b*b*cosVal*cosVal - b*b*d*d);
	solution(0) = -(b*cosVal - b*f*solution(2) + c*e*solution(2))/(a*e - b*d);
	solution(1) = (a*cosVal - a*f*solution(2) + c*d*solution(2))/(a*e - b*d);
	directionVectors.push_back(solution);

	// make sure the solution we get is 100% correct
	assert(abs(solution.dot(reference)-cosVal)<1.0E-6);
	assert(abs(solution.dot(tangential))<1.0E-6);
	assert(abs(solution.norm()-1.0)<1.0E-6);
}


// perform Newton iteration to solve non-linear equation
void findSolutionByNewtonIteration(const Eigen::Vector3d& normal, const Eigen::Vector3d& reference,
		const double& angle, Eigen::Vector3d& solution)
{
	Eigen::Vector3d startingValue = reference;
	const double& cosVal = cos(angle);

	int iter = 0;
	double error = DBL_MAX;

	Eigen::MatrixXd Jacobian(3,3);
	Eigen::Vector3d f_value;

	for(int i=0; i<3; ++i)
	{
		Jacobian(0,i) = normal(i);
		Jacobian(1,i) = reference(i);
	}

	while(iter<NewtonIteration && error>ErrorThreshold)
	{
		for(int i=0; i<3; ++i)
			Jacobian(2,i) = 2.0*startingValue(i);

		f_value(0) = normal.dot(startingValue);
		f_value(1) = reference.dot(startingValue)-cosVal;
		f_value(2) = startingValue.dot(startingValue)-1.0;

		solution = startingValue-Relaxation*Jacobian.inverse()*f_value;

		++iter;
		error = (solution-startingValue).norm();
		startingValue = solution;
	}

	// get the solution point to satisfy the nonlinear equation arrays
}


/* find an adaptive kernel radius r for smoothing */
const double LocalScalar::getKernelRadius(const double& ratio)
{
	double radius;

	// calculate the diagonal distance of the voxel
	if(Z_RESOLUTION==1)	// for 2D case
		radius = std::sqrt(X_STEP*X_STEP+Y_STEP*Y_STEP);
	else
		radius = std::sqrt(X_STEP*X_STEP+Y_STEP*Y_STEP+Z_STEP*Z_STEP);
	radius*= ratio;
	return radius;
}


/* perform the smoothing the points along the lines to re-sample the voxel information */
void LocalScalar::performSmoothingOnLine(const double& r)
{
	// select the smoothing option
	int smoothingStrategy;
	std::cout << "Select the smoothing strategy: 1.Laplacian linear smoothing, 2.Kernel Density Estimation: " << std::endl;
	std::cin >> smoothingStrategy;
	assert(smoothingStrategy==1 || smoothingStrategy==2);

	const int& TotalSize = 7;

	// using the Laplacian linear smoothing
	if(smoothingStrategy==1)
		performLaplacianSmoothing(TotalSize);
	else if(smoothingStrategy==2)
		performKDE_smoothing(r, TotalSize);
}


// the smoothing is by using the Laplacian smoothing, it is a linearly combined smoothing w.r.t. distance
void LocalScalar::performLaplacianSmoothing(const int& TotalSize)
{
	std::vector<double> new_scalars = segmentScalars;

	// each scalar values will be calculated as the neighboring points on the line
#pragma omp parallel for schedule(static) num_threads(8)
	for(int i=0; i<streamlineCount; ++i)
	{
		const std::vector<int>& vertexArray = streamlineToVertex[i];
		const int& arraySize = vertexArray.size();

		// if there are fewer than 7 vertices on the streamlines, will not smooth it
		if(arraySize<TotalSize)
			continue;

		for(int j=0; j<arraySize; ++j)
		{
			double summation = 0.0;
			const int& current_index = vertexArray[j];

			// calculating the averaged values of the 7 neighboring points
			const int& HALF = TotalSize/2;
			if(j<=HALF)
			{
				for(int k=0; k<TotalSize; ++k)
				{
					if(k!=j)	// don't include itself
						summation += segmentScalars[vertexArray[k]];
				}
			}
			else if(j>=arraySize-HALF-1)
			{
				for(int k=0; k<TotalSize; ++k)
				{
					if(j!=arraySize-TotalSize+k)	// don't include itself
						summation += segmentScalars[vertexArray[arraySize-TotalSize+k]];
				}
			}
			else
			{
				for(int k=-HALF; k<=HALF; ++k)
				{
					if(k)	// don't include itself
						summation += segmentScalars[vertexArray[j+k]];
				}
			}
			new_scalars[current_index] = summation/TotalSize;
		}
	}

	// re-assign the segmentScalars
	segmentScalars = new_scalars;
}


// the smoothing is by using the Gaussian kernel (https://en.wikipedia.org/wiki/Kernel_smoother)
void LocalScalar::performKDE_smoothing(const double& r, const int& TotalSize)
{
	std::vector<double> new_scalars = segmentScalars;

	// each scalar values will be calculated as the neighboring points on the line
#pragma omp parallel for schedule(static) num_threads(8)
	for(int i=0; i<streamlineCount; ++i)
	{
		const std::vector<int>& vertexArray = streamlineToVertex[i];
		const int& arraySize = vertexArray.size();

		// if there are fewer than 7 vertices on the streamlines, will not smooth it
		if(arraySize<TotalSize)
			continue;

		for(int j=0; j<arraySize; ++j)
		{
			double summation = 0.0, sum_coefficient = 0.0, coefficient;
			const int& current_index = vertexArray[j];
			Eigen::Vector3d neighborCoordinate, difference;
			Eigen::Vector3d current_coordinate = Eigen::Vector3d(coordinates[i][3*j], coordinates[i][3*j+1],
					coordinates[i][3*j+2]);
			int neighborIndex;

			// calculating the averaged values of the 7 neighboring points
			const int& HALF = TotalSize/2;

			double one_divided_by_r_r, one_divided_by_r;
			// get the 2.0*r*r
			if(useSuperpoint)
			{
				one_divided_by_r = 1.0/bandwidth[j];
				one_divided_by_r_r = 1.0/2.0*bandwidth[j]*bandwidth[j];
			}
			else
			{
				one_divided_by_r = 1.0/r;
				one_divided_by_r_r = 1.0/(2.0*r*r);
			}

			if(j<=HALF)
			{
				for(int k=0; k<TotalSize; ++k)
				{
					if(k!=j)	// don't include itself
					{
						neighborCoordinate = Eigen::Vector3d(coordinates[i][3*k], coordinates[i][3*k+1],
											 coordinates[i][3*k+2]);
						difference = neighborCoordinate-current_coordinate;

						if(useManualNormalization)
						{
							coefficient = exp(-difference.transpose().dot(difference)*one_divided_by_r_r);
							sum_coefficient += coefficient;
						}
						else
						{
							if(is3D)	// 3D use cubic
								coefficient = one_divided_by_r*one_divided_by_r*one_divided_by_r
										  *exp(-difference.transpose().dot(difference)*one_divided_by_r_r);
							else	// 2D use quadratic
								coefficient = one_divided_by_r*one_divided_by_r*
								exp(-difference.transpose().dot(difference)*one_divided_by_r_r);
						}
						summation += coefficient*segmentScalars[vertexArray[k]];
					}
				}
			}
			else if(j>=arraySize-HALF-1)
			{
				for(int k=0; k<TotalSize; ++k)
				{
					if(j!=arraySize-TotalSize+k)	// don't include itself
					{
						neighborIndex = arraySize-TotalSize+k;
						neighborCoordinate = Eigen::Vector3d(coordinates[i][3*neighborIndex],
								coordinates[i][3*neighborIndex+1], coordinates[i][3*neighborIndex+2]);
						difference = neighborCoordinate-current_coordinate;

						if(useManualNormalization)
						{
							coefficient = exp(-difference.transpose().dot(difference)*one_divided_by_r_r);
							sum_coefficient += coefficient;
						}
						else
						{
							if(is3D)	// 3D case cubic
								coefficient = bandwidth[j]*bandwidth[j]*bandwidth[j]*
								exp(-difference.transpose().dot(difference)*one_divided_by_r_r);
							else	// 2D quadratic
								coefficient = bandwidth[j]*bandwidth[j]*
								exp(-difference.transpose().dot(difference)*one_divided_by_r_r);
						}
						summation += coefficient*segmentScalars[vertexArray[neighborIndex]];
					}
				}
			}
			else
			{
				for(int k=-HALF; k<=HALF; ++k)
				{
					if(k)	// don't include itself
					{
						neighborIndex = j+k;
						neighborCoordinate = Eigen::Vector3d(coordinates[i][3*neighborIndex],
								coordinates[i][3*neighborIndex+1], coordinates[i][3*neighborIndex+2]);
						difference = neighborCoordinate-current_coordinate;

						if(useManualNormalization)
						{
							coefficient = exp(-difference.transpose().dot(difference)*one_divided_by_r_r);
							sum_coefficient += coefficient;
						}
						else
						{
							if(is3D)
								coefficient = bandwidth[j]*bandwidth[j]*bandwidth[j]*
									exp(-difference.transpose().dot(difference)*one_divided_by_r_r);
							else
								coefficient = bandwidth[j]*bandwidth[j]*
								exp(-difference.transpose().dot(difference)*one_divided_by_r_r);
						}

						summation += coefficient*segmentScalars[vertexArray[neighborIndex]];
					}
				}
			}

			if(!useManualNormalization)	// use default KDE formula without additional normalization
			{
				if(is3D)	// is 3D data set
					sum_coefficient = TotalSize*(sqrt(2.0*M_PI), 3.0);
				else	// is 2D data set
					sum_coefficient = TotalSize*(sqrt(2.0*M_PI), 2.0);
			}

			if(sum_coefficient>=1.0E-6)
				new_scalars[current_index] = summation/sum_coefficient;
		}
	}

	// re-assign the segmentScalars
	segmentScalars = new_scalars;
}


/* sample on the voxels from discrete 3D points. To save computational efficiency, we will use splatting algorithm */
void LocalScalar::sampleOnVoxels(const double& r, std::vector<double>& voxelScalars)
{
	std::cout << "Sampling on [" << X_GRID_RESOLUTION << "," << Y_GRID_RESOLUTION << "," << Z_GRID_RESOLUTION
	          << "] point grid started..." << std::endl;

	const int& totalVoxelSize = X_GRID_RESOLUTION*Y_GRID_RESOLUTION*Z_GRID_RESOLUTION;
	const int& xy_multiplication = X_GRID_RESOLUTION*Y_GRID_RESOLUTION;

	// intialize the vector to be zero,
	voxelScalars = std::vector<double>(totalVoxelSize, 0.0);
	std::vector<double> coefficient_summation(totalVoxelSize, 0.0);
	std::vector<int> neighborCount(totalVoxelSize, 0);

	for(int i=0; i<streamlineVertexCount; ++i)
	{
		// calculate the splatting effect from the KDE function
		splat_kde(i, r, voxelScalars, coefficient_summation, neighborCount);
	}

	std::cout << "The splatting kde is done!" << std::endl;

#pragma omp parallel for schedule(static) num_threads(8)
	for(int i=0; i<totalVoxelSize; ++i)
	{
		if(neighborCount[i]==0)
			continue;
		// directly call the KDE normalization, then assign to Gaussian integrals
		if(!useManualNormalization)
		{
			double denominator;
			if(!is3D)	// 2D case
				denominator = std::pow(sqrt(2.0*M_PI),2.0)*neighborCount[i];
			else
				denominator = std::pow(sqrt(2.0*M_PI),3.0)*neighborCount[i];
			voxelScalars[i] /= denominator;
		}
		else	// call the manual normalization for the KDE interpolation
		{
			if(neighborCount[i]>0)
			{
				voxelScalars[i] /= coefficient_summation[i];
			}
		}
	}
}


// calculate the bandwidth with superpoint generation algorithm or not based on user selection of whether to
// invoke the manual input selection
const double LocalScalar::get_kde_bandwidth()
{
	double r = 0.0;

	// enable customized parameter for the resolution for x
	std::cout << "Choose the x resolution: " << std::endl;
	std::cin >> X_GRID_RESOLUTION;
	assert(X_GRID_RESOLUTION>0);

	if(!is3D)
	{
		float y_to_x = (range[1].sup-range[1].inf)/(range[0].sup-range[0].inf);
		Y_GRID_RESOLUTION = y_to_x*X_GRID_RESOLUTION+0.5, Z_GRID_RESOLUTION = 1;
		X_GRID_STEP = (range[0].sup-range[0].inf)/float(X_GRID_RESOLUTION-1);
		Y_GRID_STEP = (range[1].sup-range[1].inf)/float(Y_GRID_RESOLUTION-1);
		Z_GRID_STEP = 1.0E5;
	}
	else
	{
		float y_to_x = (range[1].sup-range[1].inf)/(range[0].sup-range[0].inf);
		float z_to_x = (range[2].sup-range[2].inf)/(range[0].sup-range[0].inf);
		Y_GRID_RESOLUTION = y_to_x*X_GRID_RESOLUTION+0.5, Z_GRID_RESOLUTION = z_to_x*X_GRID_RESOLUTION+0.5;
		X_GRID_STEP = (range[0].sup-range[0].inf)/float(X_GRID_RESOLUTION-1);
		Y_GRID_STEP = (range[1].sup-range[1].inf)/float(Y_GRID_RESOLUTION-1);
		Z_GRID_STEP = (range[2].sup-range[2].inf)/float(Z_GRID_RESOLUTION-1);
	}

	// choose how to get the bandwidth
	std::cout << "Whether use Superpoint Generation (SG) to get the adaptive bandwidth? 1. Yes, 0. No:" << std::endl;
	int useSG_option;
	std::cin >> useSG_option;
	assert(useSG_option==0 || useSG_option==1);
	useSuperpoint = (useSG_option==1);

	// use adaptive bandwidth
	if(useSuperpoint)
	{
		SuperpointGeneration sg;

		int numOfClusters;

		// decide whether should use the automatic calculation from resolution or not
		std::cout << "Input number of superpoint generation or decided by resolution or by point number? 1.resolution, 2.point: " << std::endl;
		int clusterOption;
		std::cin >> clusterOption;
		assert(clusterOption==1 || clusterOption==2);

		// automatically calculating from the image resolution
		if(clusterOption==1)
		{
			if(!is3D)
				numOfClusters = X_GRID_RESOLUTION*Y_GRID_RESOLUTION/16/16;
			else
				numOfClusters = X_GRID_RESOLUTION*Y_GRID_RESOLUTION*Z_GRID_RESOLUTION/16/16/16;
		}
		else	// enable user input just in case some number of clusters calculated are soo huge or sooo small
		{
			if(!is3D)
				numOfClusters = vertexVec.size()/16/16;
			else
				numOfClusters = vertexVec.size()/16/16/16;
			assert(numOfClusters>1 && numOfClusters<vertexVec.size());
		}
		
		std::cout << "The number of superpoint is " << numOfClusters << std::endl;
		sg.get_superpoint_bandwidth(vertexVec, numOfClusters, bandwidth);
	}
	else
	{
		// select the ratio for Gaussian kernel size
		std::cout << "Select the ratio for the Gaussian kernel size: " << std::endl;
		double ratio;
		std::cin >> ratio;
		assert(ratio>0);

		r = getKernelRadius(ratio);
	}

	return r;
}


// calculate the splatting effect from the KDE function
void LocalScalar::splat_kde(const int& index, const double& kernel_radius, std::vector<double>& voxelScalars,
		std::vector<double>& coefficient_summation, std::vector<int>& neighborCount)
{
	const Eigen::Vector3d& current_pos = vertexVec[index];

	// find the existing index of the point in the grid points
	const int& x = (current_pos(0)-range[0].inf)/X_GRID_STEP;
	const int& y = (current_pos(1)-range[1].inf)/Y_GRID_STEP;
	const int& z = (current_pos(2)-range[2].inf)/Z_GRID_STEP;
	const int& xy_multiplication = X_GRID_RESOLUTION*Y_GRID_RESOLUTION;

	// get the bandwidth for this point
	double current_bandwdith, diff, coefficient, reciprocal_bandwidth;
	if(!useSuperpoint)
		current_bandwdith = kernel_radius;
	else
		current_bandwdith = 1.0/bandwidth[index];

	reciprocal_bandwidth = 1.0/current_bandwdith;

	// find the neighboring voxel size
	const int& X_SIZE = 4*current_bandwdith/X_GRID_STEP;
	const int& Y_SIZE = 4*current_bandwdith/Y_GRID_STEP;
	const int& Z_SIZE = 4*current_bandwdith/Z_GRID_STEP;

	// the neighboring index
	int neighbor_index;

	for(int k=std::max(0,z-Z_SIZE); k<=std::min(Z_GRID_RESOLUTION-1,z+Z_SIZE); ++k)
	{
		for(int j=std::max(0,y-Y_SIZE); j<=std::min(Y_GRID_RESOLUTION-1,y+Y_SIZE); ++j)
		{
			for(int i=std::max(0,x-X_SIZE); i<=std::min(X_GRID_RESOLUTION-1,x+X_SIZE); ++i)
			{
				neighbor_index = xy_multiplication*k+X_GRID_RESOLUTION*j+i;
				diff = (current_pos-Eigen::Vector3d(range[0].inf+i*X_GRID_STEP, range[1].inf+j*Y_GRID_STEP,
						range[2].inf+k*Z_GRID_STEP)).norm();

				if(diff<=4.0*current_bandwdith)
				{
					if(useManualNormalization)	// use enforced normalization
					{
						coefficient = exp(-diff*diff/2.0/current_bandwdith/current_bandwdith);
						coefficient_summation[neighbor_index] += coefficient;
					}
					else	// no manual normalization required, just use regular KDE
					{
						if(is3D)	// 3D data set
							coefficient = reciprocal_bandwidth*reciprocal_bandwidth*reciprocal_bandwidth
							*exp(-diff*diff/2.0/current_bandwdith/current_bandwdith);
						else	// 2D data set
							coefficient = reciprocal_bandwidth*reciprocal_bandwidth*
							exp(-diff*diff/2.0/current_bandwdith/current_bandwdith);					
					}
					voxelScalars[neighbor_index] += coefficient*segmentScalars[index];
					++neighborCount[neighbor_index];
				}
			}
		}
	}
}
