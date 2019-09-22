/*
 * main.cpp
 *
 *  Created on: Mar 12, 2019
 *      Author: lieyu
 */

#include "LocalScalar.h"
#include "ReadData.h"


int main(int argc, char* argv[])
{
	if(argc!=2)
	{
		std::cout << "Error for argument input! Sould be ./execute argument_file..." << std::endl;
		exit(1);
	}

	/* get streamline coordinates */
	VectorField vf;
	vf.readStreamlineFromFile(string(argv[1]));

	std::cout << vf.streamlineVector.size() << std::endl;

	LocalScalar ls(vf.streamlineVector, string(argv[1]));

	ls.getLocalScalar();

	return 0;
}
