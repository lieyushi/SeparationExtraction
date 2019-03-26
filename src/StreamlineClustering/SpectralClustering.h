/*
 * SpectralClustering.h
 *
 *  Created on: Mar 7, 2019
 *      Author: lieyu
 */

#ifndef SRC_STREAMLINECLUSTERING_SPECTRALCLUSTERING_H_
#define SRC_STREAMLINECLUSTERING_SPECTRALCLUSTERING_H_

#include "LineClustering.h"
#include "Evrot.h"
#include "Initialization.h"
#include <queue>


/* update date size for gradient descent */
#ifndef GradientStep
	#define GradientStep 0.3
#endif


// remove two elements in template vector
template <class T>
void deleteVecElements(std::vector<T>& original, const T& first, const T& second);


void getMatrixPow(Eigen::DiagonalMatrix<double,Dynamic>& matrix, const double& powNumber);


class SpectralClustering: public LineClustering
{
public:
	// SpectralClustering();

	SpectralClustering(Eigen::MatrixXd& distanceMatrix, const string& name, const int& vertexCount);

	void performClustering();

	virtual ~SpectralClustering();

private:

	/* number of clusters */
	int numberOfClusters;

	/* post processing techniques, 1.kmeans, 2.eigen-rotation minimization */
	int postProcessing;

	/* scaling factor for spectral clustering to decide Gaussian kernel size */
	int SCALING;

	/* preset number */
	int presetNumber;

	/* Laplacian option: 1.Normalized Laplacian, 2.Unsymmetric Laplacian */
	int laplacianOption = 1;

	/* mMethod: 1.numerical derivative, 2.true derivative */
	int mMethod;

	/* Gaussian kernel radius for generating adjacency matrix */
	std::vector<double> sigmaVec;

	/* set input parameters for spectral clustering */
	void setInputParameters();

	/* get local scaling from NIPS 2002 paper */
	void getSigmaList();

	/**********************************************************************************************************
	 **********************************   Spectral Clustering Algorithm   **************************************
	 **********************************************************************************************************/

	/* get weighted adjacency matrix by Gaussian kernel */
	void getAdjacencyMatrix(Eigen::MatrixXd& adjacencyMatrix);

	/* get degree matrix */
	void getDegreeMatrix(const Eigen::MatrixXd& adjacencyMatrix, Eigen::DiagonalMatrix<double,Dynamic>& degreeMatrix);

	/* get Laplacian matrix */
	void getLaplacianMatrix(const Eigen::MatrixXd& adjacencyMatrix, Eigen::DiagonalMatrix<double,Dynamic>& degreeMatrix,
								Eigen::MatrixXd& laplacianMatrix);

	/* decide optimal cluster number by eigenvectors of Laplacian matrix */
	void getEigenClustering(const Eigen::MatrixXd& laplacianMatrix);

	/* normalize each row first */
	void normalizeEigenvec(Eigen::MatrixXd& eigenVec);

	/* perform k-means clustering */
	void performKMeans(const Eigen::MatrixXd& eigenVec);

	/* get cluster information based on eigenvector rotation */
	void getEigvecRotation(const Eigen::MatrixXd& X);


};

#endif /* SRC_STREAMLINECLUSTERING_SPECTRALCLUSTERING_H_ */
