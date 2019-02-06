/*
 * main.cpp
 *
 *  Created on: Jan 7, 2019
 *      Author: lieyu
 */
#include "ReadData.h"
#include "SeparationExtraction.h"
#include "Visualization.h"

int main(int argc, char* argv[])
{
	if(argc!=2)
	{
		std::cout << "Error for argument input! Sould be ./execute fileName..." << std::endl;
		exit(1);
	}

	VectorField vf;
	vf.readVectorField(string(argv[1]));
	// vf.printVectorFieldVTK();

	double integrationStep;
	int maxLength, maxSeeding;

	int automaticParameter;
	std::cout << "Enable user input for streamline tracing? 0. No, 1. Yes. " << std::endl;
	std::cin >> automaticParameter;
	assert(automaticParameter==0 || automaticParameter==1);

	if(automaticParameter)
	{
		// parameter setting for streamline tracing
		std::cout << "Please input the streamline integration step: " << std::endl;
		std::cin >> integrationStep;
		assert(integrationStep>0);

		std::cout << "Please input the maximal count on a single streamline: " << std::endl;
		std::cin >> maxLength;
		assert(maxLength>0);

		std::cout << "Please input number of seeding points between [1," << vf.vertexCount << "]." << std::endl;
		std::cin >> maxSeeding;
		assert(maxSeeding>0);
	}
	else
	{
		integrationStep = 0.004;
		maxLength = 800;
		maxSeeding = 400;
	}

	// trace streamlines w.r.t. given parameters
	vf.traceStreamlines(integrationStep, maxLength, maxSeeding);

	// print streamline vtk file for visualization
	vf.printStreamlinesVTK();

	SeparationExtraction se;

	std::cout << "Judge separation by, 1.expansion, 2.chi-test of discrete curvatures..." << std::endl;
	int measurementOption;
	std::cin >> measurementOption;

	if(measurementOption==1)
		se.extractSeparationLinesByExpansion(vf.streamlineVector);
	else if(measurementOption==2)
		se.extractSeparationLinesByChiTest(vf.streamlineVector);

	vf.writeSeparationToStreamlines(se.separationVector, string("Separation"));

	Visualization vtkRender;

	vtkRender.renderStreamlines(vf.streamlineVector, se.separationVector);

	return 0;
}



