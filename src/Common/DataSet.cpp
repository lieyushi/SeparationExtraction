/*
 * DataSet.cpp
 *
 *  Created on: Feb 21, 2019
 *      Author: lieyu
 */

#include "DataSet.h"


/**
 * Find a substring in a string
 * @return a pointer to the first occurrence of searchString in the inputString
*/
const char* locateSubString(const char* inputString, const char* searchString)
{
    const char* foundLoc = strstr(inputString, searchString);
    if (foundLoc) return foundLoc + strlen(searchString);
    return inputString;
}


DataSet::DataSet() {
	// TODO Auto-generated constructor stub

}

DataSet::~DataSet() {
	// TODO Auto-generated destructor stub

}

/*
 * read 3D vector field from some given customized 3D data sets
 */
void DataSet::read3DVectorField(const string& fileName, std::vector<Vertex>& vertexVec,
		CoordinateLimits limits[3], int& x_resolution, int& y_resolution, int& z_resolution)
{
	// read bernard flow data
	if(fileName.find("bernard")!=std::string::npos)
	{
		readBernardRaw(fileName, vertexVec, limits, x_resolution, y_resolution, z_resolution);
	}

	// read crayfish, plume and two-swirl flow data
	else if(fileName.find("crayfish")!=std::string::npos || fileName.find("plume")!=std::string::npos
			|| fileName.find("two_swirl")!=std::string::npos)
	{
		readVecData(fileName, vertexVec, limits, x_resolution, y_resolution, z_resolution);
	}

	else if(fileName.find("tornado")!=std::string::npos)
	{
		generateTornado(fileName, vertexVec, limits, x_resolution, y_resolution, z_resolution);
	}

	else if(fileName.find("cylinder")!=std::string::npos)
	{
		readCylinderRaw(fileName, vertexVec, limits, x_resolution, y_resolution, z_resolution);
	}

	else if(fileName.find("Hurricane")!=std::string::npos)
	{
		readHurricaneFlow(fileName, vertexVec, limits, x_resolution, y_resolution, z_resolution);
	}

	std::cout << "Data loading is finished!" << std::endl;
}


/*
 * read Bernard.raw data into vertexVec
 */
void DataSet::readBernardRaw(const string& fileName, std::vector<Vertex>& vertexVec,
		CoordinateLimits limits[3], int& x_resolution, int& y_resolution, int& z_resolution)
{
	// make sure it is bernard data set
	auto pos = fileName.find("bernard");
	assert(pos!=std::string::npos);

	//
	x_resolution = 128;
	y_resolution = 32;
	z_resolution = 64;
	vertexVec.resize(x_resolution*y_resolution*z_resolution);

	const int& XY_Multipy = x_resolution*y_resolution;

	FILE* pFile2 = fopen (fileName.c_str(), "rb" );
	if (pFile2==NULL)
	{
		fputs ("File error",stderr);
		exit(1);
	}
	fseek( pFile2, 0L, SEEK_SET );

	//Read the data
	// - how much to read
	const size_t NumToRead = x_resolution * y_resolution * z_resolution * 3;
	// - prepare memory; use malloc() if you're using pure C
	unsigned char* pData = new unsigned char[NumToRead];
	if (pData)
	{
		// - do it
		const size_t ActRead = fread((void*)pData, sizeof(unsigned char), NumToRead, pFile2);
		// - ok?
		if (NumToRead != ActRead)
		{
			printf("Something went wrong while reading the binary data section.\nPremature end of file?\n");
			delete[] pData;
			fclose(pFile2);
			return ;
		}

		//Test: Print all data values
		//Note: Data runs x-fastest, i.e., the loop over the x-axis is the innermost
		//printf("\nPrinting all values in the same order in which they are in memory:\n");
		int Idx(0);
		double tmp[3];
		for(int k=0;k<z_resolution;k++)
		{
			for(int j=0;j<y_resolution;j++)
			{
				for(int i=0;i<x_resolution;i++)
				{
					//Note: Random access to the value (of the first component) of the grid point (i,j,k):
					// pData[((k * yDim + j) * xDim + i) * NumComponents]
					//assert(pData[((k * resolutionY + j) * resolutionX + i) * 3] == pData[Idx * 3]);

					for(int c=0;c<3;c++)
					{
						tmp[c] = (double)pData[Idx * 3 + c]/255. - 0.5;
					}
					double dist = sqrt(tmp[0]*tmp[0] + tmp[1]*tmp[1] + tmp[2]*tmp[2]);
					Vertex& temp = vertexVec[k*XY_Multipy+j*x_resolution+i];
					temp = Vertex(i, j, k, tmp[0]/dist, tmp[1]/dist, tmp[2]/dist);
					temp.v_magnitude = 1.0;
//					for(int c=0;c<3;c++)
//					{
//						loadedData[i][j][k][c] = tmp[c]/dist;
//					}

					Idx++;
				}
			}
		}
		delete[] pData;
	}
	fclose(pFile2);

	limits[0].inf = limits[1].inf = limits[2].inf = 0;
	limits[0].sup = x_resolution-1;
	limits[1].sup = y_resolution-1;
	limits[2].sup = z_resolution-1;
}


/*
 * generate tornado 3d vector field
 */
void DataSet::generateTornado(const string& fileName, std::vector<Vertex>& vertexVec,
		CoordinateLimits limits[3], int& x_resolution, int& y_resolution, int& z_resolution)
{
	auto pos = fileName.find("tornado");
	assert(pos!=std::string::npos);

	// get resolution and step size for the tornado
	x_resolution = y_resolution = z_resolution = 128;
	for(int i=0; i<3; ++i)
	{
		limits[i].inf = 0.0;
		limits[i].sup = 1.0;
	}

	// resize the array
	vertexVec.resize(x_resolution*y_resolution*z_resolution);

	const int& XY_MULTIPLY = x_resolution*y_resolution;

	const double& xdelta = 1.0 / (x_resolution-1.0);
	const double& ydelta = 1.0 / (y_resolution-1.0);
	const double& zdelta = 1.0 / (z_resolution-1.0);

	double x, y, z;
	int ix, iy, iz;
	double r, xc, yc, scale, temp, z0, vx, vy, vz;
	double r2 = 8;
	const double& SMALL = 0.00000000001;
	const int& time = 1000;

	int counter = 0;
	for( iz = 0; iz < z_resolution; iz++ )
	{
		z = iz * zdelta;                        // map z to 0->1
		xc = 0.5 + 0.1*sin(0.04*time+10.0*z);   // For each z-slice, determine the spiral circle.
		yc = 0.5 + 0.1*cos(0.03*time+3.0*z);    //    (xc,yc) determine the center of the circle.
		r = 0.1 + 0.4 * z*z + 0.1 * z * sin(8.0*z); //  The radius also changes at each z-slice.
		r2 = 0.2 + 0.1*z;                           //    r is the center radius, r2 is for damping
		for( iy = 0; iy < y_resolution; iy++ )
		{
			y = iy * ydelta;
			for( ix = 0; ix < x_resolution; ix++ )
			{
				x = ix * xdelta;
				temp = sqrt( (y-yc)*(y-yc) + (x-xc)*(x-xc) );
				scale = abs( r - temp );
				/*
				*  I do not like this next line. It produces a discontinuity
				*  in the magnitude. Fix it later.
				*
				*/
				if ( scale > r2 )
					scale = 0.8 - scale;
				else
					scale = 1.0;
				z0 = 0.1 * (0.1 - temp*z );
				if ( z0 < 0.0 )  z0 = 0.0;
				temp = sqrt( temp*temp + z0*z0 );
				scale = (r + r2 - temp) * scale / (temp + SMALL);
				scale = scale / (1+z);

				vx = scale * (y-yc) + 0.1*(x-xc);
				vy = scale * -(x-xc) + 0.1*(y-yc);;
				vz = scale * z0;

				// assign vertex information w.r.t. velocity field
				Vertex& currentVertex = vertexVec[iz*XY_MULTIPLY+iy*x_resolution+ix];
				currentVertex = Vertex(ix*xdelta, iy*ydelta, iz*zdelta, vx, vy, vz);
				currentVertex.v_magnitude = sqrt(vx*vx+vy*vy+vz*vz);
			}
		}
	}
}


/*
 * read plume.vec or two_swirl.vec for plume and two_swirl
 */
void DataSet::readVecData(const string& fileName, std::vector<Vertex>& vertexVec,
		CoordinateLimits limits[3], int& x_resolution, int& y_resolution, int& z_resolution)
{
	auto plume_pos = fileName.find("plume");
	auto two_swirl_pos = fileName.find("two_swirl");
	auto crayfish_pos = fileName.find("crayfish");

	// it's a plume data set
	if(plume_pos!=std::string::npos)
	{
		x_resolution = 126;
		y_resolution = 126;
		z_resolution = 512;
	}
	// it's a two_swirl data set
	else if(two_swirl_pos!=std::string::npos)
	{
		x_resolution = 64;
		y_resolution = 64;
		z_resolution = 64;
	}
	else if(crayfish_pos!=std::string::npos)
	{
		x_resolution = 322;
		y_resolution = 162;
		z_resolution = 119;
	}
	const int& XY_MULTIPLY = x_resolution*y_resolution;

	vertexVec.resize(x_resolution*y_resolution*z_resolution);

	struct vec3{
	    float x,y,z;
	};

	vec3* vec_field = new vec3[x_resolution*y_resolution*z_resolution];
	std::ifstream file(fileName.c_str()); // The raw data file
	file.read((char*)vec_field, x_resolution*y_resolution*z_resolution*sizeof(vec3));

	for (int z=0;z<z_resolution;z++)
	{
		for (int y=0;y<y_resolution;y++)
		{
			for (int x=0;x<x_resolution;x++)
			{
				vec3 v = vec_field[z*XY_MULTIPLY+y*x_resolution+x];
				Vertex& currentVertex = vertexVec[z*XY_MULTIPLY+y*x_resolution+x];
				currentVertex = Vertex(x, y, z, double(v.x), double(v.y), double(v.z));
				currentVertex.v_magnitude = sqrt(v.x*v.x+v.y*v.y+v.z*v.z);
			}
		}
	}

	if(vec_field!=NULL)
	{
		delete[] vec_field;
		vec_field = NULL;
	}

	limits[0].inf = limits[1].inf = limits[2].inf = 0;
	limits[0].sup = x_resolution-1;
	limits[1].sup = y_resolution-1;
	limits[2].sup = z_resolution-1;
}


/*
 * read cylinder flow for plume and two_swirl
 */
void DataSet::readCylinderRaw(const string& fileName, std::vector<Vertex>& vertexVec,
		CoordinateLimits limits[3], int& x_resolution, int& y_resolution, int& z_resolution)
{
	auto cylinder_pos = fileName.find("cylinder");
	assert(cylinder_pos!=std::string::npos);

	FILE* fp = fopen(fileName.c_str(), "rb");
	if (!fp)
	{
		printf("Could not find %s\n", fileName.c_str());
		return ;
	}

	//We read the first 2k bytes into memory to parse the header.
	//The fixed buffer size looks a bit like a hack, and it is one, but it gets the job done.
	char buffer[2048];
	fread(buffer, sizeof(char), 2047, fp);
	buffer[2047] = '\0'; //The following string routines prefer null-terminated strings

	if (!strstr(buffer, "# AmiraMesh BINARY-LITTLE-ENDIAN 2.1"))
	{
		printf("Not a proper AmiraMesh file.\n");
		fclose(fp);
		return ;
	}

	//Find the Lattice definition, i.e., the dimensions of the uniform grid
	int xDim(0), yDim(0), zDim(0);
	sscanf(locateSubString(buffer, "define Lattice"), "%d %d %d", &xDim, &yDim, &zDim);

	//Find the BoundingBox
	float xmin(1.0f), ymin(1.0f), zmin(1.0f);
	float xmax(-1.0f), ymax(-1.0f), zmax(-1.0f);
	sscanf(locateSubString(buffer, "BoundingBox"), "%g %g %g %g %g %g", &xmin, &xmax, &ymin, &ymax, &zmin, &zmax);

	//Is it a uniform grid? We need this only for the sanity check below.
	const bool bIsUniform = (strstr(buffer, "CoordType \"uniform\"") != NULL);

	//Type of the field: scalar, vector
	int NumComponents(0);
	if (strstr(buffer, "Lattice { float Data }"))
	{
		//Scalar field
		NumComponents = 1;
	}
	else
	{
		//A field with more than one component, i.e., a vector field
		sscanf(locateSubString(buffer, "Lattice { float["), "%d", &NumComponents);
	}

	//Sanity check
	if (xDim <= 0 || yDim <= 0 || zDim <= 0
			|| xmin > xmax || ymin > ymax || zmin > zmax
			|| !bIsUniform || NumComponents <= 0)
	{
		printf("Something went wrong\n");
		fclose(fp);
		return ;
	}

	x_resolution = xDim;
	y_resolution = yDim;
	z_resolution = zDim;

	const int& XY_MULTIPLY = x_resolution*y_resolution;
	vertexVec.resize(x_resolution*y_resolution*z_resolution);

	// update the inf and sup range of the vector field
	limits[0].inf = xmin;
	limits[1].inf = ymin;
	limits[2].inf = zmin;
	limits[0].sup = xmax;
	limits[1].sup = ymax;
	limits[2].sup = zmax;

	const double& delta_x = (xmax-xmin)/double(xDim-1);
	const double& delta_y = (ymax-ymin)/double(yDim-1);
	const double& delta_z = (zmax-zmin)/double(zDim-1);

	//Find the beginning of the data section
	const long idxStartData = strstr(buffer, "# Data section follows") - buffer;
	if (idxStartData > 0)
	{
		//Set the file pointer to the beginning of "# Data section follows"
		fseek(fp, idxStartData, SEEK_SET);
		//Consume this line, which is "# Data section follows"
		fgets(buffer, 2047, fp);
		//Consume the next line, which is "@1"
		fgets(buffer, 2047, fp);

		//Read the data
		// - how much to read
		const size_t NumToRead = xDim * yDim * zDim * NumComponents;
		// - prepare memory; use malloc() if you're using pure C
		float* pData = new float[NumToRead];


		if (pData)
		{
			// - do it
			const size_t ActRead = fread((void*)pData, sizeof(float), NumToRead, fp);
			// - ok?
			if (NumToRead != ActRead)
			{
				printf("Something went wrong while reading the binary data section.\nPremature end of file?\n");
				delete[] pData;
				fclose(fp);
				return ;
			}

			//Test: Print all data values
			//Note: Data runs x-fastest, i.e., the loop over the x-axis is the innermost
			//printf("\nPrinting all values in the same order in which they are in memory:\n");
			int Idx(0);
			float sum[3]={0.};

			for(int k=0;k<zDim;k++)
			{
				for(int j=0;j<yDim;j++)
				{
					for(int i=0;i<xDim;i++)
					{
						//Note: Random access to the value (of the first component) of the grid point (i,j,k):
						// pData[((k * yDim + j) * xDim + i) * NumComponents]
						assert(pData[((k * yDim + j) * xDim + i) * NumComponents] == pData[Idx * NumComponents]);

						vertexVec[k*XY_MULTIPLY+j*x_resolution+i] = Vertex(delta_x*i, delta_y*j, delta_z*k,
						double(pData[Idx * NumComponents + 0]), double(pData[Idx * NumComponents + 1]),
						double(pData[Idx * NumComponents + 2]));

						for(int c=0;c<NumComponents;c++)
						{
							sum[c]+=pData[Idx * NumComponents + c];
						}

						Idx++;
					}

				}
			}

			float avg[3]={0.};
			for(int c=0;c<3;c++)
				avg[c] = sum[c] / (zDim*yDim*xDim);

			for(int k=0;k<zDim;k++)
			{
				for(int j=0;j<yDim;j++)
				{
					for(int i=0;i<xDim;i++)
					{
						Vertex& currentVertex = vertexVec[k*XY_MULTIPLY+j*x_resolution+i];

						currentVertex.vx-=double(avg[0]);
						currentVertex.vy-=double(avg[1]);
						currentVertex.vz-=double(avg[2]);

						currentVertex.v_magnitude = sqrt(currentVertex.vx*currentVertex.vx
								+currentVertex.vy*currentVertex.vy+currentVertex.vz*currentVertex.vz);
					}
				}
			}
			delete[] pData;
		}
	}
	fclose(fp);
}


/*
 * read Hurricane flow from a given file
 */
void DataSet::readHurricaneFlow(const string& fileName, std::vector<Vertex>& vertexVec,
		CoordinateLimits limits[3], int& x_resolution, int& y_resolution, int& z_resolution)
{
	auto hurricane_pos = fileName.find("Hurricane");
	assert(hurricane_pos!=std::string::npos);

	// set resolution, grid limit and delta values
	x_resolution = 500;
	y_resolution = 500;
	z_resolution = 100;

	limits[0].inf = limits[1].inf = limits[2].inf = 0;
	limits[0].sup = 5.0;
	limits[1].sup = 5.0;
	limits[2].sup = 1.0;

	const int& XY_MULTIPLY = x_resolution*y_resolution;
	vertexVec.resize(x_resolution*y_resolution*z_resolution);

	const double& delta_x = (limits[0].sup-limits[0].inf)/double(x_resolution-1);
	const double& delta_y = (limits[1].sup-limits[1].inf)/double(y_resolution-1);
	const double& delta_z = (limits[2].sup-limits[2].inf)/double(z_resolution-1);

	string singleLineStr;
	std::ifstream inputFile(fileName.c_str());
	if (inputFile.is_open())
	{
		for(int k=0;k<z_resolution;k++)
		{
			for(int j=0;j<y_resolution;j++)
			{
				for(int i=0;i<x_resolution;i++)
				{
					if (inputFile.good()){

						getline (inputFile,singleLineStr);
						char * pch;
						vector<char> v(singleLineStr.length() + 1);
						strcpy(&v[0], singleLineStr.c_str());
						char* str = &v[0];

						pch = strtok (str," \n");
						int idx = 0;
						Vertex& currentVertex = vertexVec[k*XY_MULTIPLY+j*x_resolution+i];

						currentVertex.x = delta_x*i;
						currentVertex.y = delta_y*j;
						currentVertex.z = delta_z*k;

						currentVertex.vx = atof(pch);

						pch = strtok (NULL, " \n");
						currentVertex.vy = atof(pch);

						pch = strtok (NULL, " \n");
						currentVertex.vz = atof(pch);

						currentVertex.v_magnitude = sqrt(currentVertex.vx*currentVertex.vx+
								currentVertex.vy*currentVertex.vy+currentVertex.vz*currentVertex.vz);
					}
				}
			}
		}
	}
	inputFile.close();
}
