/*
 * main.cpp
 *
 *  Created on: Aug 27, 2019
 *      Author: lieyu
 */


#include "ReadData.h"
#include "VTKWritter.h"

int main(int argc, char* argv[])
{
	if(argc!=2)
	{
		std::cout << "Error for argument input! Sould be ./execute fileName..." << std::endl;
		exit(1);
	}

	VectorField vf;

	vf.readVectorField(string(argv[1]));

	VTKWritter::printVectorField(vf.dataset_name, vf.vertexVec, vf.limits, vf.X_RESOLUTION, vf.Y_RESOLUTION,
			vf.Z_RESOLUTION, vf.X_STEP, vf.Y_STEP, vf.Z_STEP);

	int automaticParameter;
	std::cout << "Enable user input for streamline tracing? 0. No, 1. Yes. " << std::endl;
	std::cin >> automaticParameter;
	assert(automaticParameter==0 || automaticParameter==1);

	double integrationStepRatio;
	int maxLength, maxSeeding;
	if(automaticParameter)
	{
		// parameter setting for streamline tracing, this is the integration step ratio to the diagonal distance
		// of the face (2D) or voxel (3D). If the diagonal dist is 0.1, and the ratio is 0.01, then the step size
		// will be h = 0.01*0.1 = 0.001
		std::cout << "Please input the streamline integration step ratio between (0,1.0): " << std::endl;
		std::cin >> integrationStepRatio;
		assert(integrationStepRatio>0 && integrationStepRatio<0.1);

		std::cout << "Please input the maximal count on a single streamline: " << std::endl;
		std::cin >> maxLength;
		assert(maxLength>0);

		std::cout << "Please input number of seeding points between [1," << vf.vertexCount << "]." << std::endl;
		std::cin >> maxSeeding;
		assert(maxSeeding>0);
	}
	else
	{
		integrationStepRatio = 0.15;
		maxLength = 2000;
		maxSeeding = 500;
	}

	// trace streamlines w.r.t. given parameters
	vf.traceStreamlines(integrationStepRatio, maxLength, maxSeeding);

	// 0 means not continue, 1 means to continue
	int toContinue;

	do
	{
		double filterSize;
		std::cout << "Select a filter size so that streamlines with number of points smaller than this will be dropped"
				<< " between [0,1.0): " << std::endl;
		std::cin >> filterSize;
		assert(filterSize>0);

		vf.filterShortStreamlines(filterSize);

		// print the number of filtered streamlines
		std::cout << "The number of filtered streamlines is " << vf.streamlineVector.size() << std::endl;

		// print streamline data sets into clustering framework
		vf.printStreamlineTXT();

		// print streamline vtk file for visualization
		vf.printStreamlinesVTK();

		std::cout << "Whether to continue filtering the short streamlines? 0.No, 1.Yes. " << std::endl;
		std::cin >> toContinue;

	}while(toContinue);

	return 0;
}


