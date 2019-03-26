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
	std::cout << "Choose a segmentation option for streamlines, 1.log(vertexSize) sampling, 2.3-1-3 sampling," << std::endl;
	std::cin >> segmentOption;
	assert(segmentOption==1 || segmentOption==2 || segmentOption==3);

	/* k value in KNN */
	std::cout << "Choose the k for KNN kernel: " << std::endl;
	std::cin >> kNeighborNumber;
	assert(kNeighborNumber>0 && kNeighborNumber<=int(streamlineCount*0.1));

	/* encoding is enabled or not */
	std::cout << "Choose whether scalar encoding is enabled or not? 1.Yes, 0.No," << std::endl;
	int encodingOption;
	std::cin >> encodingOption;
	assert(encodingOption==1 || encodingOption==0);
	encodingEnabled = (encodingOption==1);
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
		X_RESOLUTION = 51, Y_RESOLUTION = 51, Z_RESOLUTION = 1;
		X_STEP = (range[0].sup-range[0].inf)/float(X_RESOLUTION-1);
		Y_STEP = (range[1].sup-range[1].inf)/float(Y_RESOLUTION-1);
		Z_STEP = 1.0;
	}
	else
	{
		X_RESOLUTION = 31, Y_RESOLUTION = 31, Z_RESOLUTION = 31;
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


/* follow a 3-1-3 alignment, judge point distance with lnk/(lnk+1) mapping such that
 * 0, infinite large--->1 (stronger separation), 1--->0 (weaker separation)
 */
void LocalScalar::processAlignmentByPointWise()
{
	assignPointsToBins();

	// point-wise scalar value calculation on streamlines
	std::vector<double> segmentScalars(streamlineVertexCount);

	std::cout << "Whether search for left/right neighbors or not? 1.Yes, 2.No." << std::endl;
	int directionOption;
	std::cin >> directionOption;
	assert(directionOption==1 || directionOption==2);
	bool directionEanbled = (directionOption==1);

	std::cout << "Whether candidates on same streamlines enabled? 1.Yes, 2.No." << std::endl;
	int sameLineOption;
	std::cin >> sameLineOption;
	assert(sameLineOption==1 || sameLineOption==2);
	bool sameLineEnabled = (sameLineOption==1);

	const int& TOTALSIZE = 7;

//#pragma omp parallel for schedule(static) num_threads(8)
	for(int i=0; i<streamlineCount; ++i)
	{
		const std::vector<int>& vertexArray = streamlineToVertex[i];
		const int& arraySize = vertexArray.size();

		for(int j=0; j<arraySize; ++j)
		{
			std::vector<int> closestPoint;
			getScalarsOnPointWise(directionEanbled, sameLineEnabled, j, vertexArray,
					closestPoint);

			getScalarsFromNeighbors(j, vertexArray, TOTALSIZE, closestPoint, segmentScalars[vertexArray[j]]);
		}
	}

	VTKWritter::printStreamlineScalarsOnSegments(coordinates,datasetName, streamlineVertexCount,
				segmentScalars);
}


// compute the scalar value on point-wise element of the streamline
void LocalScalar::getScalarsOnPointWise(const bool& directionEanbled, const bool& sameLineEnabled, const int& j,
		const std::vector<int>& vertexArray, std::vector<int>& pointCandidate)
{
	int current;
	int targetID = vertexArray[j];

	Eigen::Vector3d& targetVertex = vertexVec[targetID];

	/* search left-closest and right-closest in 2D space */
	if(directionEanbled)
	{
		searchClosestThroughDirections(sameLineEnabled, j, vertexArray, pointCandidate);
	}
	else
	{
		searchKNNPointByBining(sameLineEnabled, j, vertexArray, pointCandidate);
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


// search through two directions
void LocalScalar::searchClosestThroughDirections(const bool& sameLineEnabled, const int& j,
		const std::vector<int>& vertexArray, std::vector<int>& pointCandidate)
{
	int current;
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

	Eigen::Vector3d currentPos = center;
	int currentBin, streamID;

	const double& THRESHOLD = 0.7074;
	double dotProduct;

	bool found = false;
	std::unordered_map<int, bool> binChosen;
	while(1)
	{
		currentPos += searchingStep*searchDirection;

		int x = (currentPos(0)-range[0].inf)/X_STEP;
		int y = (currentPos(1)-range[1].inf)/Y_STEP;
		int z = (currentPos(2)-range[2].inf)/Z_STEP;
		if(x<0 || x>=X_RESOLUTION || y<0 || y>=Y_RESOLUTION || z<0 || z>=Z_RESOLUTION)
			break;

		currentBin = xy_multiplication*z+X_RESOLUTION*y+x;
		if(binChosen.find(currentBin)==binChosen.end())
		{
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
				if(zDirection(2)<0)
					continue;
				pointIDArray.push_back(MinimalDist((currentPos-vertexVec[streamID]).norm(), streamID));
			}
			// sort distance from minimal to maximal
			std::sort(pointIDArray.begin(), pointIDArray.end(), CompareDistRecord());

			for(int k=0; k<pointIDArray.size(); ++k)
			{
				int streamID = pointIDArray[k].index;

				// should be parallel as much as possible
				Eigen::Vector3d segmentCandidate;
				const std::vector<int>& vertexListOfLine = streamlineToVertex[vertexToLine[streamID]];
				const int& streamlineSelected = vertexListOfLine.size();
				if(streamID==vertexListOfLine[0])
					segmentCandidate = vertexVec[vertexListOfLine[1]]-vertexVec[vertexListOfLine[0]];
				else if(streamID==vertexListOfLine[streamlineSelected-1])
					segmentCandidate = vertexVec[streamID]-vertexVec[streamID-1];
				else
					segmentCandidate = vertexVec[streamID+1]-vertexVec[streamID-1];
				// normalize the tangential direction
				segmentCandidate/=segmentCandidate.norm();
				dotProduct = segmentCandidate.dot(tangential);
				dotProduct = std::min(dotProduct, 1.0);
				dotProduct = std::max(dotProduct, -1.0);

				if(abs(dotProduct)>=THRESHOLD)
				{
					found = true;
					pointCandidate.push_back(streamID);
					break;
				}
			}
			binChosen[currentBin] = true;
		}

		if(found)
			break;
	}

	/*
	 * Along right directions search for the closest right
	 */
	searchDirection = -searchDirection;
	currentPos = center;
	found = false;
	binChosen.clear();
	while(1)
	{
		currentPos += searchingStep*searchDirection;

		int x = (currentPos(0)-range[0].inf)/X_STEP;
		int y = (currentPos(1)-range[1].inf)/Y_STEP;
		int z = (currentPos(2)-range[2].inf)/Z_STEP;

		if(x<0 || x>=X_RESOLUTION || y<0 || y>=Y_RESOLUTION || z<0 || z>=Z_RESOLUTION)
			break;

		currentBin = xy_multiplication*z+X_RESOLUTION*y+x;
		if(binChosen.find(currentBin)==binChosen.end())
		{
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

			for(int k=0; k<pointIDArray.size(); ++k)
			{
				int streamID = pointIDArray[k].index;

				// should be parallel as much as possible
				Eigen::Vector3d segmentCandidate;
				const std::vector<int>& vertexListOfLine = streamlineToVertex[vertexToLine[streamID]];
				const int& streamlineSelected = vertexListOfLine.size();
				if(streamID==vertexListOfLine[0])
					segmentCandidate = vertexVec[vertexListOfLine[0]+1]-vertexVec[vertexListOfLine[0]];
				else if(streamID==vertexListOfLine[streamlineSelected-1])
					segmentCandidate = vertexVec[streamID]-vertexVec[streamID-1];
				else
					segmentCandidate = vertexVec[streamID+1]-vertexVec[streamID-1];

				segmentCandidate/=segmentCandidate.norm();
				dotProduct = segmentCandidate.dot(tangential);
				dotProduct = std::min(dotProduct, 1.0);
				dotProduct = std::max(dotProduct, -1.0);

				if(abs(dotProduct)>=THRESHOLD)
				{
					found = true;
					pointCandidate.push_back(streamID);
					break;
				}
			}
			binChosen[currentBin] = true;
		}

		if(found)
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

	double ratio, start_dist, end_dist;
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
		}
		else if(candStreamline.back()-candidate<=HALF)
		{
			end_dist = (vertexVec[firstEnd]-vertexVec[candStreamline.back()]).norm();
			start_dist = (vertexVec[firstBegin]-vertexVec[candStreamline[candSize-TOTALSIZE]]).norm();
		}
		else
		{
			end_dist = (vertexVec[firstEnd]-vertexVec[candidate+HALF]).norm();
			start_dist = (vertexVec[firstBegin]-vertexVec[candidate-HALF]).norm();
		}
		ratio = end_dist/start_dist;

		if(isinf(ratio))
		{
			std::cout << "Found inf! Program exit! " << end_dist << " " << start_dist << std::endl;
			exit(1);
		}
		summation+=ratio;
		++effective;
	}

	if(effective==0)
	{
		scalar = 0;

		return;
	}

	summation/=effective;

	/* non-linear encoding is enabled to squeeze the scalar value inside [0,1.0] */
	if(encodingEnabled)
	{
		if(summation<1.0E-8)
		{
			scalar = 0.0;
		}
		else
		{
			summation = abs(log2(summation));
			scalar = summation/(summation+1);
		}
	}
	// otherwise, squeeze 1.0 to 0
	else
	{
		scalar = abs(summation-1.0);
	}
}
