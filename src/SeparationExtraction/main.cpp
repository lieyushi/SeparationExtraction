/*
 * main.cpp
 *
 *  Created on: Jan 7, 2019
 *      Author: lieyu
 */
#include <sys/time.h>
#include "ReadData.h"
#include "SeparationExtraction.h"

void extractForPlyFile(const int& argc, char* argv[], VectorField& vf);

void performExtraction(VectorField& vf, SeparationExtraction& se);

void readStreamlines(const int& argc, char* argv[], VectorField& vf);

int main(int argc, char* argv[])
{
	if(argc!=2)
	{
		std::cout << "Error for argument input! Sould be ./execute fileName..." << std::endl;
		exit(1);
	}

	/* multiple input */
	std::cout << "What type the file is? 1.ply vector field, 2.3D vector field, 3.streamline data sets." << std::endl;
	int datasetOption;
	std::cin >> datasetOption;
	assert(1<=datasetOption && datasetOption<=3);

	VectorField vf;
	SeparationExtraction se;

	switch(datasetOption)
	{
	/* input is .ply file, and need to trace streamlines first */
	case 1:
		extractForPlyFile(argc, argv, vf);
		break;

	case 2:
		break;

	case 3:
		readStreamlines(argc, argv, vf);
		break;

	default:
		break;
	}

	performExtraction(vf, se);

	return 0;
}

void extractForPlyFile(const int& argc, char* argv[], VectorField& vf)
{
	vf.readVectorField(string(argv[1]));
	vf.printVectorFieldVTK();

	double integrationStepRatio;
	int maxLength, maxSeeding;

	int automaticParameter;
	std::cout << "Enable user input for streamline tracing? 0. No, 1. Yes. " << std::endl;
	std::cin >> automaticParameter;
	assert(automaticParameter==0 || automaticParameter==1);

	if(automaticParameter)
	{
		// parameter setting for streamline tracing
		std::cout << "Please input the streamline integration ratio between (0,0.1): " << std::endl;
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
		integrationStepRatio = 0.2;
		maxLength = 800;
		maxSeeding = 625;
	}

	// trace streamlines w.r.t. given parameters
	vf.traceStreamlines(integrationStepRatio, maxLength, maxSeeding);

	// print streamline vtk file for visualization
	vf.printStreamlinesVTK();

	vf.printStreamlineTXT();
}

void readStreamlines(const int& argc, char* argv[], VectorField& vf)
{
	vf.readStreamlineFromFile(string(argv[1]));
	// print streamline vtk file for visualization
	vf.printStreamlinesVTK();
}


void performExtraction(VectorField& vf, SeparationExtraction& se)
{
	std::cout << "Judge separation by, 1.expansion, 2.chi-test of discrete curvatures, "
				"3.chi-test of discrete curvatures of line segments..." << std::endl;
	int measurementOption;
	std::cin >> measurementOption;

	struct timeval start, end;
	double timeTemp;
	gettimeofday(&start, NULL);

	if(measurementOption==1)
		se.extractSeparationLinesByExpansion(vf.streamlineVector);
	else if(measurementOption==2)
		se.extractSeparationLinesByChiTest(vf.streamlineVector);
	else if(measurementOption==3)
		se.extractSeparationLinesBySegments(vf.streamlineVector);

	// record the time for local scalar value calculation
	gettimeofday(&end, NULL);
	timeTemp = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
	std::cout << "The time for separation estimate is " << timeTemp << " seconds!" << std::endl;

	vf.writeSeparationToStreamlines(se.separationVector, string("Separation"));
}
