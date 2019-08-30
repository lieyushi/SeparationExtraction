#include "ReadData.h"


// default constructor
VectorField::VectorField()
{
	vertexCount = 0;
	// nothing to do for the default constructor
}

// destructor
VectorField::~VectorField()
{

}

// read vector field data
void VectorField::readVectorField(const string& fileName)
{
	const auto& pos = fileName.find(".ply");
	if(pos!=std::string::npos)
	{
		readPlyFile(fileName);
		dataset_name = fileName.substr(0, pos);
	}
	else
	{
		const auto& txtPos = fileName.find(".txt");
		if(txtPos!=std::string::npos)	// read vector field from txt file
		{
			readPlainFile(fileName);
			dataset_name = fileName;
		}
		else
		{
			readVectorFieldFromRaw(fileName);
			dataset_name = fileName;
		}
	}

	if(Z_RESOLUTION>=2)
	{
		is3D = true;
		MINIMAL = minimal_ratio*std::sqrt(X_STEP*X_STEP+Y_STEP*Y_STEP+Z_STEP*Z_STEP);
	}
	else
		MINIMAL = minimal_ratio*std::sqrt(X_STEP*X_STEP+Y_STEP*Y_STEP);
}


// trace streamlines w.r.t. some parameters and seeding strategy
void VectorField::traceStreamlines(const double& integrationStepRatio, const int& maxLength, int& maxSeeding)
{
	// choosing seeding strategy for streamline tracing
	int seedingStrategy;
	std::cout << "Choose uniform seeding or entropy-based sampling? 0.Uniform seeding, 1.entropy seeding. " << std::endl;
	std::cin >> seedingStrategy;
	assert(seedingStrategy==0 || seedingStrategy==1);

	// whether need to judge whether streamlines are too close to existing streamlines
	int judgeClose;
	std::cout << "Whether judge streamlines are too close to other streamlines? 0.No, 1.Yes." << std::endl;
	std::cin >> judgeClose;
	assert(judgeClose==0 || judgeClose==1);
	useSpatial = (judgeClose==1);

	if(useSpatial)
	{
		// direct apply the same resolution as in grid
		spatialBins.clear();
		spatialBins = std::vector<std::vector<StreamlinePoint> >(X_RESOLUTION*Y_RESOLUTION*Z_RESOLUTION);
	}

	std::vector<Vertex> seeds;

	// uniform sampling strategy applied
	if(seedingStrategy==0)
		uniformSeeding(seeds, maxSeeding);
	// entropy-based seeding
	else if(seedingStrategy==1)
		entropySeeding(seeds, maxSeeding);

	double integrationStep;
	if(is3D)
		integrationStep = integrationStepRatio*std::sqrt(X_STEP*X_STEP+Y_STEP*Y_STEP+Z_STEP*Z_STEP);
	else
		integrationStep = integrationStepRatio*std::sqrt(X_STEP*X_STEP+Y_STEP*Y_STEP);

	// trace streamlines
	traceStreamlinesBySeeds(seeds, integrationStep, maxLength);
}


// read vector field from ply file
void VectorField::readPlyFile(const string& fileName)
{
	// try-catch block for reading vector filed from .ply file
	std::ifstream readIn;
	readIn.exceptions(ifstream::badbit);
	try
	{
		readIn.open(fileName.c_str(), ios::in);
		string line;
		stringstream ss;

		// omit first four lines in .ply file
		for(int i=0; i<4; ++i)
			getline(readIn, line);

		// fifth line indicates how many vertices exist
		getline(readIn, line);
		ss.str(line);
		// filter out vertex numbers
		ss >> line;
		ss >> line;
		ss >> line;
		vertexCount = std::atoi(line.c_str());

		// re-initialization of stringstream ss
		ss.str("");
		ss.clear();

		// omit another 9 lines
		for(int i=0; i<9; ++i)
			getline(readIn, line);

		vertexVec = std::vector<Vertex>(vertexCount);

		std::vector<double> value;
		// read vertex coordinates and velocity components

		for(int i=0; i<vertexCount; ++i)
		{
			// extract the six double values for vertex information
			getline(readIn, line);
			ss.str(line);
			while(ss >> line)
			{
				value.push_back(std::atof(line.c_str()));
			}
			// assign value to the vector
			vertexVec[i].assignValue(value);

			// update the limits[3] for x, y and z
			for(int j=0; j<3; ++j)
			{
				limits[j].inf = std::min(limits[j].inf, value[j]);
				limits[j].sup = std::max(limits[j].sup, value[j]);
			}
			// clear buffer
			value.clear();
			ss.str("");
			ss.clear();
		}
		readIn.close();
	}
	catch(const ifstream::failure& e)
	{
		std::cout << "Exception opening for file reading!" << std::endl;
		exit(1);
	}

	const int& size_sqrt = (int)std::sqrt(vertexCount);
	X_RESOLUTION = size_sqrt;
	Y_RESOLUTION = size_sqrt;
	Z_RESOLUTION = 1;
	X_STEP = (limits[0].sup-limits[0].inf)/(double)(X_RESOLUTION-1);
	Y_STEP = (limits[1].sup-limits[1].inf)/(double)(Y_RESOLUTION-1);

	std::cout << "File reading finishes!" << std::endl;
}


// read 2D vector field from plain file
void VectorField::readPlainFile(const string& fileName)
{
	// default file order is x, y, z, vx, vy, vz
	std::ifstream readIn;
	readIn.exceptions(ifstream::badbit);
	try
	{
		readIn.open(fileName.c_str(), ios::in);
		string line;
		stringstream ss;

		// dynamically assign the vertex
		vertexCount = 0;
		std::vector<double> value;
		// read vertex coordinates and velocity components

		while(getline(readIn, line))
		{
			ss.str(line);
			while(ss >> line)
			{
				value.push_back(std::atof(line.c_str()));
			}
			// assign value to the vector
			vertexVec.push_back(Vertex(value));
			// update the limits[3] for x, y and z
			for(int j=0; j<3; ++j)
			{
				limits[j].inf = std::min(limits[j].inf, value[j]);
				limits[j].sup = std::max(limits[j].sup, value[j]);
			}
			// clear buffer
			value.clear();
			ss.str("");
			ss.clear();
			++vertexCount;
		}
		readIn.close();
	}
	catch(const ifstream::failure& e)
	{
		std::cout << "Exception opening for file reading!" << std::endl;
		exit(1);
	}

	const int& size_sqrt = (int)std::sqrt(vertexCount);
	X_RESOLUTION = size_sqrt;
	Y_RESOLUTION = size_sqrt;
	Z_RESOLUTION = 1;
	X_STEP = (limits[0].sup-limits[0].inf)/(double)(X_RESOLUTION-1);
	Y_STEP = (limits[1].sup-limits[1].inf)/(double)(Y_RESOLUTION-1);

	std::cout << "File reading finishes!" << std::endl;
}

// print vector field for certification
void VectorField::printVectorFieldVTK()
{
	// get information for rectilinear grids
	for(int i=0; i<3; ++i)
		std::cout << "[" << limits[i].inf << ", " << limits[i].sup << "]" << std::endl;

	// create vtk file
	stringstream ss;
	ss << dataset_name << "_VectorField.vtk";
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
    fout << "DIMENSIONS " << X_RESOLUTION << " " << Y_RESOLUTION << " " << Z_RESOLUTION << endl;
    fout << "ASPECT_RATIO " << X_STEP << " " << Y_STEP << " " << Z_STEP << endl;
    fout << "ORIGIN " << limits[0].inf << " " << limits[1].inf << " " << limits[2].inf << endl;
    fout << "POINT_DATA " << vertexCount << endl;
    fout << "SCALARS velocity_magnitude double 1" << endl;
    fout << "LOOKUP_TABLE velo_table" << endl;

    const int& SLICE_NUMBER = Y_RESOLUTION*X_RESOLUTION;
    for (int i = 0; i < Z_RESOLUTION; ++i)
    {
		for (int j = 0; j < Y_RESOLUTION; ++j)
		{
			for (int k = 0; k < X_RESOLUTION; ++k)
			{
				fout << vertexVec[SLICE_NUMBER*i+X_RESOLUTION*j+k].v_magnitude << endl;
			}
		}
    }
    fout << "VECTORS velocityDirection double" << endl;

    Vertex vertex;
    for (int i = 0; i < Z_RESOLUTION; ++i)
    {
		for (int j = 0; j < Y_RESOLUTION; ++j)
		{
			for (int k = 0; k < X_RESOLUTION; ++k)
			{
				vertex = vertexVec[SLICE_NUMBER*i+X_RESOLUTION*j+k];
				fout << vertex.vx << " " << vertex.vy << " " << vertex.vz << endl;
			}
		}
    }
	fout.close();
}

// fourth-order Runge-Kutta integration method for streamline tracing
bool VectorField::getIntegrationValue(const double& step, const Eigen::Vector3d& position,
								      const Eigen::Vector3d& velocity, Eigen::Vector3d& nextPos)
{
	Eigen::Vector3d k_1 = velocity;
	Eigen::Vector3d k_2, k_3, k_4;

	if(is3D)
	{
		if(!getInterpolatedVelocity3D(position+k_1*step/2.0, k_2))
			return false;
		if(!getInterpolatedVelocity3D(position+k_2*step/2.0, k_3))
			return false;
		if(!getInterpolatedVelocity3D(position+k_3*step, k_4))
			return false;
		nextPos = position+(k_1+2.0*k_2+2.0*k_3+k_4)/6.0*step;
	}
	else	// for 2D case
	{
		if(!getInterpolatedVelocity2D(position+k_1*step/2.0, k_2))
			return false;
		if(!getInterpolatedVelocity2D(position+k_2*step/2.0, k_3))
			return false;
		if(!getInterpolatedVelocity2D(position+k_3*step, k_4))
			return false;
		nextPos = position+(k_1+2.0*k_2+2.0*k_3+k_4)/6.0*step;
	}
	return true;
}


// get the velocity of temporary position
bool VectorField::getInterpolatedVelocity(const Eigen::Vector3d& position, Eigen::Vector3d& velocity)
{
	if(is3D)
		return getInterpolatedVelocity3D(position, velocity);
	else
		return getInterpolatedVelocity2D(position, velocity);
}


// get the velocity of temporary position
bool VectorField::getInterpolatedVelocity2D(const Eigen::Vector3d& position, Eigen::Vector3d& velocity)
{
	// find four points that surrounding the target position
	int x_index = int((position(0)-limits[0].inf)/X_STEP);
	int y_index = int((position(1)-limits[1].inf)/Y_STEP);
	if(x_index==X_RESOLUTION-1)	// just in case it is the max, so guarantee x_index x1
		x_index = x_index-1;	// left
	if(y_index==Y_RESOLUTION-1)	// guarantee y_index is y1
		y_index = y_index-1;	// bottom

	if(x_index<0 || x_index>=X_RESOLUTION || y_index<0 || y_index>=Y_RESOLUTION)
		return false;

	/*
	 *    (x1,y1+1) _______________(x1+1,y1+1)
	 *        	    |              |
	 *        		|              |
	 *        		|______________|
	 *       (x1,y1)              (x1+1, y1)
	 */

	int bottom_left = X_RESOLUTION*y_index+x_index;
	int bottom_right = bottom_left+1;
	int top_left = bottom_left+X_RESOLUTION;
	int top_right = top_left+1;

	if(bottom_left>=vertexCount || bottom_left<0 || bottom_right>=vertexCount || bottom_right<0
	|| top_left>=vertexCount || top_left<0 || top_right>=vertexCount || top_right<0)
	{
		//std::cout << position(0) << " " << position(1) << " " << position(2) << std::endl;
		return false;
	}

	// interpolate for bottom and top horizontal lane
	double x_ratio = (position(0)-vertexVec[bottom_left].x)/X_STEP, minus_ratio = 1.0-x_ratio;

	Eigen::Vector3d bottom_velocity, top_velocity;
	bottom_velocity(0) = x_ratio*vertexVec[bottom_right].vx+minus_ratio*vertexVec[bottom_left].vx;
	bottom_velocity(1) = x_ratio*vertexVec[bottom_right].vy+minus_ratio*vertexVec[bottom_left].vy;
	bottom_velocity(2) = x_ratio*vertexVec[bottom_right].vz+minus_ratio*vertexVec[bottom_left].vz;

	top_velocity(0) = x_ratio*vertexVec[top_right].vx+minus_ratio*vertexVec[top_left].vx;
	top_velocity(1) = x_ratio*vertexVec[top_right].vy+minus_ratio*vertexVec[top_left].vy;
	top_velocity(2) = x_ratio*vertexVec[top_right].vz+minus_ratio*vertexVec[top_left].vz;

	// interpolate through y axis
	x_ratio = (position(1)-vertexVec[bottom_left].y)/Y_STEP;
	minus_ratio = 1.0-x_ratio;

	for(int i=0; i<3; ++i)
		velocity(i) = x_ratio*top_velocity(i)+minus_ratio*bottom_velocity(i);
	return true;
}


// get the velocity of temporary position for 3D
bool VectorField::getInterpolatedVelocity3D(const Eigen::Vector3d& position, Eigen::Vector3d& velocity)
{
	// find four points that surrounding the target position
	int x_index = int((position(0)-limits[0].inf)/X_STEP);
	int y_index = int((position(1)-limits[1].inf)/Y_STEP);
	int z_index = int((position(2)-limits[2].inf)/Z_STEP);

	if(x_index==X_RESOLUTION-1)	// make sure x_index==x1
		x_index = x_index-1;	// the left
	if(y_index==Y_RESOLUTION-1)	// make sure y_index==y1
		y_index = y_index-1;	// the bottom
	if(z_index==Z_RESOLUTION-1)	// make sure z_index==z1
		z_index = z_index-1;	// the back

	if(x_index<0 || x_index>=X_RESOLUTION || y_index<0 || y_index>=Y_RESOLUTION || z_index<0 || z_index>=Z_RESOLUTION)
		return false;

	const int& XY_MULTIPLY = X_RESOLUTION*Y_RESOLUTION;
	/*
	 * i,j+1,k+1->   ________   <- i+1,j+1,k+1
	 *    			/|      /|
	 * i,j+1,k->   / |____ /_|  <- i+1,j+1,k
	 * i,j,k+1->  /__/____/ /  <- i+1,j,k+1
	 *  		  | /    | /
	 *            |/_____|/
	 *     (i,j,k)   (i+1,j,k)
	 */
	int bottom_left = XY_MULTIPLY*z_index+X_RESOLUTION*y_index+x_index;
	int bottom_right = bottom_left+1;
	int top_left = bottom_left+X_RESOLUTION;
	int top_right = top_left+1;

	int bottom_left_front = bottom_left+XY_MULTIPLY;
	int bottom_right_front = bottom_left_front+1;
	int top_left_front = bottom_left_front+X_RESOLUTION;
	int top_right_front = top_left_front+1;

	// check whether the back face is within the range
	if(bottom_left>=vertexCount || bottom_left<0 || bottom_right>=vertexCount || bottom_right<0
	|| top_left>=vertexCount || top_left<0 || top_right>=vertexCount || top_right<0)
	{
		return false;
	}

	// check whether the front face is within the range
	if(bottom_left_front>=vertexCount||bottom_left_front<0||bottom_right_front>=vertexCount||bottom_right_front<0
	||top_left_front>=vertexCount||top_left_front<0||top_right_front>=vertexCount||top_right_front<0)
	{
		return false;
	}

	// the interpolated velocity on the back face and front face
	Eigen::Vector3d backVelocity, frontVelocity;

	// Interpolate the back face
	double x_ratio = (position(0)-vertexVec[bottom_left].x)/X_STEP, minus_ratio = 1.0-x_ratio;
	Eigen::Vector3d bottom_velocity, top_velocity;
	bottom_velocity(0) = x_ratio*vertexVec[bottom_right].vx+minus_ratio*vertexVec[bottom_left].vx;
	bottom_velocity(1) = x_ratio*vertexVec[bottom_right].vy+minus_ratio*vertexVec[bottom_left].vy;
	bottom_velocity(2) = x_ratio*vertexVec[bottom_right].vz+minus_ratio*vertexVec[bottom_left].vz;

	top_velocity(0) = x_ratio*vertexVec[top_right].vx+minus_ratio*vertexVec[top_left].vx;
	top_velocity(1) = x_ratio*vertexVec[top_right].vy+minus_ratio*vertexVec[top_left].vy;
	top_velocity(2) = x_ratio*vertexVec[top_right].vz+minus_ratio*vertexVec[top_left].vz;

	// interpolate through y axis
	x_ratio = (position(1)-vertexVec[bottom_left].y)/Y_STEP;
	minus_ratio = 1.0-x_ratio;

	// get the linearly interpolated velocity on the back face
	backVelocity = x_ratio*top_velocity+minus_ratio*bottom_velocity;

	// Interpolate the front face
	x_ratio = (position(0)-vertexVec[bottom_left_front].x)/X_STEP, minus_ratio = 1.0-x_ratio;

	bottom_velocity(0) = x_ratio*vertexVec[bottom_right_front].vx+minus_ratio*vertexVec[bottom_left_front].vx;
	bottom_velocity(1) = x_ratio*vertexVec[bottom_right_front].vy+minus_ratio*vertexVec[bottom_left_front].vy;
	bottom_velocity(2) = x_ratio*vertexVec[bottom_right_front].vz+minus_ratio*vertexVec[bottom_left_front].vz;

	top_velocity(0) = x_ratio*vertexVec[top_right_front].vx+minus_ratio*vertexVec[top_left_front].vx;
	top_velocity(1) = x_ratio*vertexVec[top_right_front].vy+minus_ratio*vertexVec[top_left_front].vy;
	top_velocity(2) = x_ratio*vertexVec[top_right_front].vz+minus_ratio*vertexVec[top_left_front].vz;

	// interpolate through y axis
	x_ratio = (position(1)-vertexVec[bottom_left_front].y)/Y_STEP;
	minus_ratio = 1.0-x_ratio;

	// get the linearly interpolated velocity on the front face
	frontVelocity = x_ratio*top_velocity+minus_ratio*bottom_velocity;

	// Interpolate along the z axis
	x_ratio = (position(2)-vertexVec[bottom_left].z)/Z_STEP;
	minus_ratio = 1.0-x_ratio;

	// get the linearly interpolated velocity on the point
	velocity = x_ratio*frontVelocity+minus_ratio*backVelocity;

	return true;
}


// trace streamlines given seeding vertex position
void VectorField::traceStreamlinesBySeeds(const std::vector<Vertex>& seeds, const double& step,
										  const int& maxLength)
{
	streamlineVector = std::vector<Eigen::VectorXd>(seeds.size());
	lineVelocity.resize(seeds.size());
	lineLength = std::vector<double>(seeds.size(), 0);

// #pragma omp parallel for schedule(static) num_threads(8)
	for(int i=0; i<seeds.size(); ++i)
	{
		std::vector<Vertex> forwardTracing, backwardTracing;
		std::vector<double> forwardVelocity, backwardVelocity;
		Eigen::Vector3d position, velocity, nextPos;
		int j = 0;

		double streamline_length = 0.0;

		// forward tracing the streamlines
		forwardTracing.push_back(seeds[i]);
		forwardVelocity.push_back(seeds[i].v_magnitude);
		do
		{
			position = Eigen::Vector3d(forwardTracing.back().x, forwardTracing.back().y, forwardTracing.back().z);
			velocity = Eigen::Vector3d(forwardTracing.back().vx, forwardTracing.back().vy, forwardTracing.back().vz);
			if(!getIntegrationValue(step, position, velocity, nextPos))
				break;

			// still stays in the domain?
			if(!stayInDomain(nextPos))
				break;

			if(!getInterpolatedVelocity(nextPos, velocity))
				break;

			// judge whether the point is far away from existing streamline points
			// well, stay too close to existing streamlines, exit the loop
			if(!isFarToStreamlines(nextPos, i))
				break;

			// adding the length for the cumulated size of the streamlines
			if(!forwardTracing.empty())
			{
				Vertex back = forwardTracing.back();
				streamline_length += (nextPos-Eigen::Vector3d(back.x, back.y, back.z)).norm();
			}

			forwardTracing.push_back(Vertex(nextPos, velocity));
			forwardVelocity.push_back(velocity.norm());

			++j;
		/*
		 * check whether streamlines are close to already existing streamlines with spatial spinning
		 */
		}while(j<maxLength/2 && forwardTracing.back().v_magnitude>=MINIMAL_VELOCITY);

		// backward tracing the streamlines
		j = 0;
		do
		{
			if(backwardTracing.empty())
			{
				position = Eigen::Vector3d(forwardTracing.at(0).x, forwardTracing.at(0).y, forwardTracing.at(0).z);
				velocity = Eigen::Vector3d(forwardTracing.at(0).vx, forwardTracing.at(0).vy, forwardTracing.at(0).vz);
			}
			else
			{
				position = Eigen::Vector3d(backwardTracing.back().x, backwardTracing.back().y, backwardTracing.back().z);
				velocity = Eigen::Vector3d(backwardTracing.back().vx, backwardTracing.back().vy, backwardTracing.back().vz);
			}
			if(!getIntegrationValue(-step, position, velocity, nextPos))
				break;

			// still stays in the domain?
			if(!stayInDomain(nextPos))
				break;

			if(!getInterpolatedVelocity(nextPos, velocity))
				break;

			// judge whether the point is far away from existing streamline points
			// well, stay too close to existing streamlines, exit the loop
			if(!isFarToStreamlines(nextPos, i))
				break;

			if(!backwardTracing.empty())
			{
				Vertex back = backwardTracing.back();
				streamline_length += (nextPos-Eigen::Vector3d(back.x,back.y,back.z)).norm();
			}
			else
			{
				Vertex front = forwardTracing.at(0);
				streamline_length += (nextPos-Eigen::Vector3d(front.x,front.y,front.z)).norm();
			}

			backwardTracing.push_back(Vertex(nextPos, velocity));
			backwardVelocity.push_back(velocity.norm());

			++j;
		}while(j<maxLength-forwardTracing.size() && backwardTracing.back().v_magnitude>=MINIMAL_VELOCITY);

		// reverse the backward coordinates
		std::reverse(backwardTracing.begin(), backwardTracing.end());
		std::reverse(backwardVelocity.begin(), backwardVelocity.end());

		// merge two vectors to make one
		backwardTracing.insert(backwardTracing.end(), forwardTracing.begin(), forwardTracing.end());
		backwardVelocity.insert(backwardVelocity.end(), forwardVelocity.begin(), forwardVelocity.end());

		// clean the cache to save memory
		forwardTracing.clear();

		// assign velocity norm to the points
		lineVelocity[i] = backwardVelocity;
		assert(lineVelocity[i].size()==backwardVelocity.size());

		// assign the streamline length
		lineLength[i] = streamline_length;

		// make storage of streamline coordinates
		Eigen::VectorXd& streamline = streamlineVector[i];
		streamline = Eigen::VectorXd(3*backwardTracing.size());
		for(j=0; j<backwardTracing.size(); ++j)
		{
			streamline(3*j) = backwardTracing.at(j).x;
			streamline(3*j+1) = backwardTracing.at(j).y;
			streamline(3*j+2) = backwardTracing.at(j).z;
		}
	}
}


// print vtk for streamlines
void VectorField::printStreamlinesVTK()
{
	if(streamlineVector.empty())
		return;

	// count how many vertices on all the streamlines
	int streamlineVertexCount = 0;
	for(int i=0; i<streamlineVector.size(); ++i)
		streamlineVertexCount+=streamlineVector[i].size()/3;

	stringstream ss;
	ss << dataset_name << "_streamline.vtk";
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
	for (int i = 0; i < streamlineVector.size(); ++i)
	{
		tempRow = streamlineVector[i];
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

	fout << "LINES " << streamlineVector.size() << " " << (streamlineVertexCount+streamlineVector.size()) << std::endl;

	subSize = 0;
	for (int i = 0; i < streamlineVector.size(); ++i)
	{
		arraySize = streamlineVector[i].size()/3;
		fout << arraySize << " ";
		for (int j = 0; j < arraySize; ++j)
		{
			fout << subSize+j << " ";
		}
		subSize+=arraySize;
		fout << std::endl;
	}
	fout << "POINT_DATA" << " " << streamlineVertexCount << std::endl;
	fout << "SCALARS group int 1" << std::endl;
	fout << "LOOKUP_TABLE group_table" << std::endl;

	for (int i = 0; i < streamlineVector.size(); ++i)
	{
		arraySize = streamlineVector[i].size()/3;
		for (int j = 0; j < arraySize; ++j)
		{
			fout << i << std::endl;
		}
	}

	fout << "SCALARS velocity double 1" << std::endl;
	fout << "LOOKUP_TABLE velocity_table" << std::endl;

	std::vector<double> velocity_norm_vec;
	for (int i = 0; i < streamlineVector.size(); ++i)
	{
		velocity_norm_vec = lineVelocity.at(i);
		arraySize = velocity_norm_vec.size();
		for (int j = 0; j < arraySize; ++j)
		{
			fout << velocity_norm_vec[j] << std::endl;
		}
	}

	fout.close();
}


// stay in the range of the domain
bool VectorField::stayInDomain(const Eigen::Vector3d& pos)
{
	if(limits[0].inf<=pos(0) && pos(0)<=limits[0].sup &&
	   limits[1].inf<=pos(1) && pos(1)<=limits[1].sup &&
	   limits[2].inf<=pos(2) && pos(2)<=limits[2].sup)
		return true;
	else
		return false;
}


// use uniform sampling for streamline tracing
void VectorField::uniformSeeding(std::vector<Vertex>& seeds, int& maxSeeding)
{
	int x_size, y_size, z_size;
	// should solve the function, x_*(Y/X*x_)*(Z/X*x_) = maxSeeding

	if(is3D)	// for 3D data set
	{
		x_size = int(std::pow(float(maxSeeding*X_RESOLUTION*X_RESOLUTION)/float(Y_RESOLUTION*Z_RESOLUTION),
				1.0/3.0))+1;

		y_size = int(float(Y_RESOLUTION*x_size)/float(X_RESOLUTION));
		z_size = int(float(Z_RESOLUTION*x_size)/float(X_RESOLUTION));
		maxSeeding = x_size*y_size*z_size;
	}
	else	// for 2D data set, it will set z_size == 1
	{
		x_size = int(std::sqrt(float(maxSeeding*X_RESOLUTION)/float(Y_RESOLUTION)))+1;
		y_size = x_size*Y_RESOLUTION/X_RESOLUTION;
		maxSeeding = x_size*y_size;
		z_size = Z_RESOLUTION;
	}
	const double& seed_step_x = (limits[0].sup-limits[0].inf)/(x_size-1);
	const double& seed_step_y = (limits[1].sup-limits[1].inf)/(y_size-1);
	double seed_step_z;

	if(is3D)
		seed_step_z = (limits[2].sup-limits[2].inf)/(z_size-1);
	else
		seed_step_z = Z_STEP;

	seeds = std::vector<Vertex>(maxSeeding);
	const int& squared_num = x_size*y_size;

	// assign the coordinates and get the velocity components
	for(int k=0; k<z_size; ++k)	//	z
	{
	#pragma omp parallel for schedule(static) num_threads(8)
		for(int i=0; i<y_size; ++i)	// y
		{
			Eigen::Vector3d position, velocity;
			for(int j=0; j<x_size; ++j)	// x
			{
				position = Eigen::Vector3d(j*seed_step_x+limits[0].inf, i*seed_step_y+limits[1].inf,
						k*seed_step_z+limits[2].inf);

				getInterpolatedVelocity(position, velocity);

				seeds[k*squared_num+i*x_size+j] = Vertex(position, velocity);
			}
		}
	}
	std::cout << "Uniform seed sampling is done!" << std::endl;
}

// use entropy-based sampling for streamline tracing
// entropy-based computing: http://web.cse.ohio-state.edu/~shen.94/papers/xu_vis10.pdf
// seeding strategy: http://vis.cs.ucdavis.edu/papers/pg2011paper.pdf
void VectorField::entropySeeding(std::vector<Vertex>& seeds, int& maxSeeding)
{
	/*
	 * neighborhood size is 13^2, bin number is 60. But input grid is only 50X50, so should
	 * firstly increase to bigger resolutions, e.g., 100x100 as sampling
	 */

	// Refining the grid
	std::cout << "Entropy-based seeding begins..." << std::endl;
	const int& X_SAMPLE = 100;
	const int& Y_SAMPLE = X_SAMPLE*(float(Y_RESOLUTION)/float(X_RESOLUTION));	// to scale

	const double& seed_step_x = (limits[0].sup-limits[0].inf)/(X_SAMPLE-1);
	const double& seed_step_y = (limits[1].sup-limits[1].inf)/(Y_SAMPLE-1);

	// considering z directions
	int Z_SAMPLE;
	double seed_step_z;

	if(is3D)
	{
		Z_SAMPLE = X_SAMPLE*(float(Z_RESOLUTION)/float(X_RESOLUTION));
		seed_step_z = (limits[2].sup-limits[2].inf)/(Z_SAMPLE-1);
	}
	else
	{
		Z_SAMPLE = Z_RESOLUTION;
		seed_step_z = Z_STEP;
	}

	// re-assign the coordinates
	std::vector<Vertex> new_samples(X_SAMPLE*Y_SAMPLE*Z_SAMPLE);
	const int& XY_MULTIPLICATION = X_SAMPLE*Y_SAMPLE;

	// get the coordinates and the velocities for all the vertices inside the grid
	for(int k=0; k<Z_SAMPLE; ++k)	//	z
	{
	#pragma omp parallel for schedule(static) num_threads(8)
		for(int i=0; i<Y_SAMPLE; ++i)	// y
		{
			for(int j=0; j<X_SAMPLE; ++j)	// x
			{
				Eigen::Vector3d position, velocity;
				position = Eigen::Vector3d(j*seed_step_x+limits[0].inf, i*seed_step_y+limits[1].inf,
						k*seed_step_z+limits[2].inf);
				getInterpolatedVelocity(position, velocity);
				new_samples[k*XY_MULTIPLICATION+i*X_SAMPLE+j] = Vertex(position, velocity);
			}
		}
	}

	// compute the repsective bin no. inside 60, i.e., 0->59
	std::vector<int> patches(X_SAMPLE*Y_SAMPLE*Z_SAMPLE, -1);
	const Eigen::Vector3d& REFERENCE = Eigen::Vector3d(1,0,0);
	const double& ANGLE_UNIT = 2.0*M_PI/60.0;

	// calculate which patch the angle of velocity lies in between [0,60)
	for(int k=0; k<Z_SAMPLE; ++k)	//	z
	{
	#pragma omp parallel for schedule(static) num_threads(8)
		for(int i=0; i<Y_SAMPLE; ++i)	// y
		{
			for(int j=0; j<X_SAMPLE; ++j) // x
			{
				const Vertex& v = new_samples[k*XY_MULTIPLICATION+i*X_SAMPLE+j];
				Eigen::Vector3d velocity = Eigen::Vector3d(v.vx, v.vy, v.vz);
				Eigen::Vector3d crossValue = REFERENCE.cross(velocity);
				double dotValue = REFERENCE.dot(velocity), angle;

				angle = dotValue/v.v_magnitude;
				angle = std::max(-1.0, angle);
				angle = std::min(1.0, angle);
				/* 0->pi */
				if(crossValue(2)>=0)
				{
					angle = acos(angle);
				}
				/* pi->2pi */
				else
				{
					angle = acos(angle);
					angle = 2.0*M_PI-angle;
				}
				patches[k*XY_MULTIPLICATION+i*X_SAMPLE+j] = int(angle/ANGLE_UNIT);
			}
		}
	}

	const int& BIN_SIZE = 13;
	const int& HALF_BIN = BIN_SIZE/2;

	// compute the entropy value with neigbhor size as 13X13 for each grid points
	std::vector<double> entropyVec(X_SAMPLE*Y_SAMPLE*Z_SAMPLE, -1.0);

	for(int z_=0; z_<Z_SAMPLE; ++z_)	//	z
	{
	#pragma omp parallel for schedule(static) num_threads(8)
		for(int i=0; i<Y_SAMPLE; ++i) // y
		{
			for(int j=0; j<X_SAMPLE; ++j) // x
			{
				// begin counting and storing the neighbor bin label
				std::vector<int> localBin(60,0);
				int index, x_index, y_index, z_index, total = 0;

				for(int m = -HALF_BIN; m<=HALF_BIN; ++m)	// z
				{
					for(int k=-HALF_BIN; k<=HALF_BIN; ++k) // y
					{
						for(int l=-HALF_BIN; l<=HALF_BIN; ++l) // x
						{
							x_index = j+l;
							y_index = i+k;
							z_index = z_+m;

							// not inside the area, then mirror the data, e.g., 100->99, -1->0, e.t.c..
							if(x_index<0)
							{
								x_index = -x_index-1;
							}
							else if(x_index>=X_SAMPLE)
							{
								x_index = 2*X_SAMPLE-x_index-1;
							}
							if(y_index<0)
							{
								y_index = -y_index-1;
							}
							else if(y_index>=Y_SAMPLE)
							{
								y_index = 2*Y_SAMPLE-y_index-1;
							}
							if(z_index<0)
							{
								z_index = -z_index-1;
							}
							else if(z_index>=Z_SAMPLE)
							{
								z_index = 2*Z_SAMPLE-z_index-1;
							}

							// The most special case is for the z coordinates
							if(z_index<0 || z_index>=Z_SAMPLE)	// z index is out of range
								continue;

							index = XY_MULTIPLICATION*z_index+X_SAMPLE*y_index+x_index;
							// increase the patch number by one
							++localBin[patches[index]];
							++total;
						}
					}
				}

				assert(total!=0);
				// computing the entropy values
				double entropy = 0.0, value;
				for(int k=0; k<60; ++k)
				{
					if(localBin[k]>0)
					{
						value = double(localBin[k])/double(total);
						entropy += value*log2(value);
					}
				}
				entropyVec[XY_MULTIPLICATION*z_+X_SAMPLE*i+j] = -entropy;
			}
		}
	}

	// sampling w.r.t. probability based on scalar value on grid pints
	double totalValue = 0.0;
	for(int i=0; i<entropyVec.size(); ++i)
	{
		totalValue+=entropyVec[i];
	}

	// generate accumulated density function given probability density values on each candidates
	for(int i=0; i<entropyVec.size(); ++i)
	{
		if(i==0)
			entropyVec[i]/=totalValue;
		else
			entropyVec[i]=entropyVec[i]/totalValue+entropyVec[i-1];
	}
	// simulating sampling with probability functions
	std::vector<bool> isChosen(X_SAMPLE*Y_SAMPLE*Z_SAMPLE, false);

	int starting = 0, sampleIndex;
	srand(time(NULL));
	while(starting<maxSeeding)
	{
		double r = ((double)rand()/(RAND_MAX));
		auto low = std::upper_bound(entropyVec.begin(), entropyVec.end(), r);
		sampleIndex = (low-entropyVec.begin());
		if(!isChosen[sampleIndex])
		{
			isChosen[sampleIndex] = true;
			++starting;
			seeds.push_back(new_samples[sampleIndex]);
		}
	}
	std::cout << "Entropy-based seeding is done!" << std::endl;
}


// write vector values to the streamline tracer
void VectorField::writeSeparationToStreamlines(const std::vector<int>& separationFlag, const string& flagName)
{
	if(separationFlag.empty())
		return;

	// count how many vertices on all the streamlines
	int streamlineVertexCount = 0;
	for(int i=0; i<streamlineVector.size(); ++i)
		streamlineVertexCount+=streamlineVector[i].size()/3;

	stringstream ss;
	ss << dataset_name << "_streamline.vtk";

	std::ofstream fout(ss.str().c_str(), ios::out | ios::app );
	if(!fout)
	{
		std::cout << "Error opening the file!" << std::endl;
		exit(1);
	}

	fout << "SCALARS " << flagName << " int 1" << std::endl;
	fout << "LOOKUP_TABLE " << flagName+string("_table") << std::endl;

	int arraySize;
	for (int i = 0; i < streamlineVector.size(); ++i)
	{
		arraySize = streamlineVector[i].size()/3;
		for (int j = 0; j < arraySize; ++j)
		{
			fout << separationFlag[i] << std::endl;
		}
	}
	fout.close();
}


// whether stay away from existing point in the streamline far enough
bool VectorField::isFarToStreamlines(const Eigen::Vector3d& nextPos, const int& id)
{
	// sometimes we will not eliminate streamlines that are close enough
	if(!useSpatial)
		return true;
	// find current spatial bin position
	int x_current = int((nextPos(0)-limits[0].inf)/X_STEP);
	int y_current = int((nextPos(1)-limits[1].inf)/Y_STEP);
	int z_current = int((nextPos(2)-limits[2].inf)/Z_STEP);

	// check whether the bin triple is not erroneous
	assert(x_current>=0 && x_current<=X_RESOLUTION-1);
	assert(y_current>=0 && y_current<=Y_RESOLUTION-1);
	assert(z_current>=0 && z_current<=Z_RESOLUTION-1);

	const int& SLICE_NUMBER = X_RESOLUTION*Y_RESOLUTION;
	// check neighboring 8 bins and see whether inside there is close enough streamline points
	int x, y, z;
	std::vector<StreamlinePoint> cell;
	for(int i=std::max(0,z_current-1); i<=std::min(Z_RESOLUTION-1,z_current+1); ++i)	// z
	{
		for(int j=std::max(0,y_current-1); j<=std::min(Y_RESOLUTION-1,y_current+1); ++j)	// y
		{
			for(int k=std::max(0,x_current-1); k<=std::min(X_RESOLUTION-1,x_current+1); ++k)
			{
				cell = spatialBins[SLICE_NUMBER*i+X_RESOLUTION*j+k];
				for(auto &p:cell)
				{
					// close to existing streamlines
					if(id!=p.id && (nextPos-p.coordinate).norm()<MINIMAL)
					{
						return false;
					}
				}
			}
		}
	}

	spatialBins[SLICE_NUMBER*z_current+X_RESOLUTION*y_current+x_current].push_back(StreamlinePoint(nextPos, id));
	return true;
}


// read 3D streamlines from streamline data set
void VectorField::readStreamlineFromFile(const string& fileName)
{
	dataset_name = fileName;
	vertexCount = 0;
	std::ifstream fin(fileName.c_str(), ios::in);
	if(!fin)
	{
		std::cout << "Error creating files!" << std::endl;
		exit(1);
	}
	stringstream ss;

	std::vector<double> tempVec;

	string line, part;

	std::vector<double> vec(3);
	double temp;
	while(getline(fin, line) /* && currentNumber < MAXNUMBER*/)
	{
		//currentDimensions = 0;
		int tag = 0, count = 0;
		ss.str(line);
		while(ss>>part /*&& currentDimensions<3*MAXDIMENSION*/)
		{
			/* operations below would remove duplicate vertices because that would damage our computation */
			temp = atof(part.c_str());
			if(tag>=3)
			{
				if(count<3)
				{
					vec[count] = temp;
					++tag;
					++count;
				}
				if(count==3)
				{
					int size = tempVec.size();
					if(!(abs(vec[0]-tempVec[size-3])<1.0e-5
							&&abs(vec[1]-tempVec[size-2])<1.0e-5
							&&abs(vec[2]-tempVec.back())<1.0e-5))
					{
						tempVec.push_back(vec[0]);
						tempVec.push_back(vec[1]);
						tempVec.push_back(vec[2]);
					}
					count = 0;
				}
				continue;
			}
			tempVec.push_back(temp);
			++tag;
			//currentDimensions++;
		}
		/* accept only streamlines with at least three vertices */
		if(tempVec.size()/3>2)
		{
			streamlineVector.push_back(Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>
			(tempVec.data(), tempVec.size()));
			vertexCount+=tempVec.size();
		}
		tempVec.clear();
		ss.clear();
		ss.str("");
		//currentNumber++;
	}
	fin.close();


	vertexCount/=3;
	std::cout << "File reader has been completed, and it toally has " << streamlineVector.size() << " trajectories and "
			  << vertexCount << " vertices!" << std::endl;
}


// print streamline data sets into clustering framework
void VectorField::printStreamlineTXT()
{
	if(streamlineVector.empty())
		return;

	stringstream ss;
	ss << dataset_name << "_streamline";
	std::ofstream fout(ss.str().c_str(), ios::out);
	if(!fout)
	{
		std::cout << "Error creating a new file!" << std::endl;
		exit(1);
	}

	Eigen::VectorXd line;
	for(int i=0; i<streamlineVector.size(); ++i)
	{
		line = streamlineVector[i];
		for(int j=0; j<line.size(); ++j)
			fout << line(j) << " ";
		fout << std::endl;
	}
	fout.close();
}

// read vector field from specific format
void VectorField::readVectorFieldFromRaw(const string& fileName)
{
	DataSet::read3DVectorField(fileName, vertexVec, limits, X_RESOLUTION, Y_RESOLUTION, Z_RESOLUTION);

	if(X_RESOLUTION>1)
		X_STEP = (limits[0].sup-limits[0].inf)/double(X_RESOLUTION);

	if(Y_RESOLUTION>1)
		Y_STEP = (limits[1].sup-limits[1].inf)/double(Y_RESOLUTION);

	if(Z_RESOLUTION>1)
		Z_STEP = (limits[2].sup-limits[2].inf)/double(Z_RESOLUTION);

	vertexCount = X_RESOLUTION*Y_RESOLUTION*Z_RESOLUTION;

	std::cout << "The data set space range is in [" << limits[0].inf << "," << limits[0].sup << "]x["
													<< limits[1].inf << "," << limits[1].sup << "]x["
													<< limits[2].inf << "," << limits[2].sup << "]" << std::endl;
	std::cout << "There are " << vertexCount << " vertices detected!" << std::endl;
}


// elimiate short length of streamlines below numSize points
void VectorField::filterShortStreamlines(const double& numRatio)
{
	const int& streamlineCount = streamlineVector.size();
	const int& remain_size = numRatio*streamlineCount;

	// use priority queue (max heap) to get the index of the most numRatio streamlines
	std::priority_queue<pair<double,int> > pq;

	for(int i=0; i<streamlineCount; ++i)
	{
		pq.push(make_pair(lineLength[i], i));
		if(pq.size()>remain_size)
			pq.pop();
	}

	// fetch from the
	std::unordered_set<int> deleted_index;
	while(!pq.empty())
	{
		deleted_index.insert(pq.top().second);
		pq.pop();
	}

	// start deleting those streamlines that has e.g., 20% smallest size of lengths
	std::vector<Eigen::VectorXd> temp;
	std::vector<std::vector<double> > norm_temp;
	for(int i=0; i<streamlineCount; ++i)
	{
		// not in the unordered_set<int>
		if(deleted_index.count(i)==0)
		{
			temp.push_back(streamlineVector[i]);
			norm_temp.push_back(lineVelocity[i]);
		}
	}
	streamlineVector = temp;
	lineVelocity = norm_temp;
}
