/*
 * main.cpp
 *
 *  Created on: Apr 27, 2019
 *      Author: lieyu
 */

#include "ParallelVector.h"
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

	ParallelVector pv(vf.dataset_name, vf.vertexVec, vf.limits, vf.X_RESOLUTION, vf.Y_RESOLUTION,
						vf.Z_RESOLUTION, vf.X_STEP, vf.Y_STEP, vf.Z_STEP);

	pv.performPVOperation();

	VTKWritter::printPoints(vf.dataset_name, pv.pvPointVec);

	return 0;
}



