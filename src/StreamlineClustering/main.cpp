/*
 * main.cpp
 *
 *  Created on: Mar 6, 2019
 *      Author: lieyu
 */

#include "AHC.h"
#include "ReadData.h"
#include "Visualization.h"
#include "SimilarityDistance.h"
#include "SpectralClustering.h"


// compute distance matrix given options
void getDistanceMatrix(const std::vector<Eigen::VectorXd>& streamlines, const int& measureOption,
		Eigen::MatrixXd& distanceMatrix);

// get group id from clustering
void getGroupFromClustering(VectorField& vf, Eigen::MatrixXd& distanceMatrix,
		const int& clusteringOption, const string& name, const int& measureOption);


int main(int argc, char* argv[])
{
	if(argc!=2)
	{
		std::cout << "Error for argument input! Sould be ./execute fileName..." << std::endl;
		exit(1);
	}

	/* get streamline coordinates */
	VectorField vf;
	vf.readStreamlineFromFile(string(argv[1]));

	std::cout << vf.streamlineVector.size() << std::endl;

	/* option of distance measure */
	int measureOption;
	std::cout << "Select a distance measure, 1.MCP distance, 2.Hausdorff distance: " << std::endl;
	std::cin >> measureOption;
	assert(measureOption==1 || measureOption==2);

	/* get distance matrix */
	Eigen::MatrixXd distanceMatrix;
	getDistanceMatrix(vf.streamlineVector, measureOption, distanceMatrix);

	/* choose what clustering techniques */
	std::cout << "Select a clustering framework. 1.AHC, 2.Spectral clustering: " << std::endl;
	int clusteringOption;
	std::cin >> clusteringOption;
	assert(clusteringOption==1 || clusteringOption==2);

	/* get cluster id number */
	getGroupFromClustering(vf, distanceMatrix, clusteringOption, string(argv[1]), measureOption);

	// Visualization vtkRender;
	// vtkRender.renderStreamlines(vf.streamlineVector, se.separationVector);

	return 0;
}


// compute distance matrix given options
void getDistanceMatrix(const std::vector<Eigen::VectorXd>& streamlines, const int& measureOption,
		Eigen::MatrixXd& distanceMatrix)
{
	const int& streamlineSize = streamlines.size();
	distanceMatrix = Eigen::MatrixXd::Zero(streamlineSize, streamlineSize);

	switch(measureOption)
	{
	case 1:
		{
		#pragma omp parallel for schedule(static) num_threads(8)
			for(int i=0; i<streamlineSize; ++i)
			{
				for(int j=0; j<streamlineSize; ++j)
				{
					if(i==j)
						continue;
					distanceMatrix(i,j)=SimilarityDistance::getMcpDistance(streamlines[i],streamlines[j]);
					assert(isinf(distanceMatrix(i,j)));
				}
			}
		}
		break;

	case 2:
		{
		#pragma omp parallel for schedule(static) num_threads(8)
			for(int i=0; i<streamlineSize; ++i)
			{
				for(int j=0; j<streamlineSize; ++j)
				{
					if(i==j)
						continue;
					distanceMatrix(i,j)=SimilarityDistance::getHausdorffDistance(streamlines[i],streamlines[j]);
					assert(isinf(distanceMatrix(i,j)));
				}
			}
		}
		break;

	default:
		std::cout << "error!" << std::endl;
		exit(1);
	}

	std::cout << distanceMatrix(199,139) << " " << distanceMatrix(139,199) << std::endl;
	std::cout << distanceMatrix(199,107) << " " << distanceMatrix(107,199) << std::endl;
}


// get group id from clustering
void getGroupFromClustering(VectorField& vf, Eigen::MatrixXd& distanceMatrix,
		const int& clusteringOption, const string& name, const int& measureOption)
{
	/* what is the measure name? */
	string measureName;
	switch(measureOption)
	{
	case 1:
		measureName = "MCP";
		break;
	case 2:
		measureName = "Hausdorff";
		break;
	}

	switch(clusteringOption)
	{
	case 1:
		{
			AHC ahc(vf.streamlineVector, distanceMatrix, name, vf.vertexCount);
			ahc.performClustering();
			if(ahc.groupID.empty())
				return;
			ahc.writeLabelsIntoVTK("AHC_"+measureName);
		}
		break;

	case 2:
		{
			SpectralClustering sc(vf.streamlineVector, distanceMatrix, name, vf.vertexCount);
			sc.performClustering();
			if(sc.groupID.empty())
				return;
			sc.writeLabelsIntoVTK("SC_"+measureName);
		}
		break;
	}
}
