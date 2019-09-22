/*
 * SpectralClustering.cpp
 *
 *  Created on: Mar 7, 2019
 *      Author: lieyu
 */

#include "SpectralClustering.h"


// remove two elements in template vector
template <class T>
void deleteVecElements(std::vector<T>& original, const T& first, const T& second)
{
	std::size_t size = original.size();
	assert(size>2);
	vector<T> result(size-2);
	int tag = 0;
	for(int i=0;i<size;++i)
	{
		//meet with target elements, not copied
		if(original[i]==first || original[i]==second)
			continue;
		result[tag++]=original[i];
	}
	assert(tag==size-2);
	original = result;
}


void getMatrixPow(Eigen::DiagonalMatrix<double,Dynamic>& matrix, const double& powNumber)
{
	Eigen::VectorXd& m_v = matrix.diagonal();
#pragma omp parallel for schedule(static) num_threads(8)
	for(int i=0;i<m_v.size();++i)
		m_v(i) = pow(m_v(i), powNumber);
}


SpectralClustering::SpectralClustering(Eigen::MatrixXd& distanceMatrix, const string& name, const int& vertexCount):
		LineClustering(distanceMatrix, name, vertexCount)
{
	setInputParameters();
}


SpectralClustering::~SpectralClustering() {
	// TODO Auto-generated destructor stub
}


void SpectralClustering::performClustering()
{
	groupID.resize(streamlineCount);

	getSigmaList();

	Eigen::MatrixXd adjacencyMatrix, laplacianMatrix;
	Eigen::DiagonalMatrix<double,Dynamic> degreeMatrix;

	/* get weighted adjacency matrix by Gaussian kernel */
	getAdjacencyMatrix(adjacencyMatrix);

	/* get degree matrix */
	getDegreeMatrix(adjacencyMatrix, degreeMatrix);

	/* get Laplacian matrix */
	getLaplacianMatrix(adjacencyMatrix, degreeMatrix, laplacianMatrix);

	getEigenClustering(laplacianMatrix);
}


/* set input parameters for spectral clustering */
void SpectralClustering::setInputParameters()
{
	std::cout << "Choose Laplacian option: 1.Normalized Laplacian, 2.Unsymmetric Laplacian. " << std::endl;
	std::cin >> laplacianOption;
	assert(laplacianOption==1 || laplacianOption==2);

	std::cout << "Choose post-processing techniques: 1.k-means, 2.eigen-rotation minimization." << std::endl;
	std::cin >> postProcessing;
	assert(postProcessing==1 || postProcessing==2);

	if(postProcessing==2)
	{
		std::cout << "Please input the preset number of clusters in [2, " << streamlineCount << "]: " << std::endl;
		std::cin >> presetNumber;
		assert(presetNumber>=2 && presetNumber<=streamlineCount);

		std::cout << "------------------------------------------------" << std::endl;
		std::cout << "Please input derivative method: 1.numerical derivative, 2.true derivative." << std::endl;
		std::cin >> mMethod;
		assert(mMethod==1 || mMethod==2);

	}
	else if(postProcessing==1)
	{
		std::cout << "Please input the preset number of clusters among [2, " << streamlineCount << "]: " << std::endl;
		std::cin >> presetNumber;
		assert(presetNumber>=2 && presetNumber<=streamlineCount);
	}

	SCALING = 0.05*streamlineCount;
}


/* get local scaling from NIPS 2002 paper */
void SpectralClustering::getSigmaList()
{
	sigmaVec = std::vector<double>(streamlineCount);

	/* get SCALING-th smallest dist */
#pragma omp parallel for schedule(static) num_threads(8)
	for(int i=0;i<streamlineCount;++i)
	{
		/* instead we implement a n*logk priority_queue method for finding k-th smallest element */
		std::priority_queue<double> limitQueue;
		double tempDist;
		for(int j=0;j<streamlineCount;++j)
		{
			if(i==j)
				continue;
			tempDist = distanceMatrix(i,j);
			// element is even larger than the biggest
			limitQueue.push(tempDist);
			if(limitQueue.size()>SCALING)
				limitQueue.pop();
		}

		sigmaVec[i] = limitQueue.top();
	}
	std::cout << "Finish local scaling..." << std::endl;
}

/* get weighted adjacency matrix by Gaussian kernel */
void SpectralClustering::getAdjacencyMatrix(Eigen::MatrixXd& adjacencyMatrix)
{
	//in case of diagonal matrix element is not assigned
	adjacencyMatrix = Eigen::MatrixXd::Zero(streamlineCount, streamlineCount);
#pragma omp parallel for schedule(static) num_threads(8)
	for(int i=0;i<streamlineCount;++i)
	{
		for(int j=0;j<streamlineCount;++j)
		{
			double dist_ij;
			if(i==j)
				continue;

			dist_ij = distanceMatrix(i,j);
			adjacencyMatrix(i,j)=exp(-dist_ij*dist_ij/sigmaVec[i]/sigmaVec[j]);
		}
	}

	std::cout << "Finish computing adjacency matrix!" << std::endl;
}

/* get degree matrix */
void SpectralClustering::getDegreeMatrix(const Eigen::MatrixXd& adjacencyMatrix,
		Eigen::DiagonalMatrix<double,Dynamic>& degreeMatrix)
{
	degreeMatrix = Eigen::DiagonalMatrix<double,Dynamic>(streamlineCount);
	Eigen::VectorXd v = VectorXd::Zero(streamlineCount);
#pragma omp parallel for schedule(static) num_threads(8)
	for(int i=0;i<v.size();++i)
	{
		double summation = 0;
		for(int j=0;j<adjacencyMatrix.cols();++j)
		{
			summation+=adjacencyMatrix(i,j);
		}
		v(i) = summation;
	}

	degreeMatrix.diagonal() = v;

	std::cout << "Fnish computing degree matrix!" << std::endl;
}

/* get Laplacian matrix */
void SpectralClustering::getLaplacianMatrix(const Eigen::MatrixXd& adjacencyMatrix,
		Eigen::DiagonalMatrix<double,Dynamic>& degreeMatrix, Eigen::MatrixXd& laplacianMatrix)
{
	switch(laplacianOption)
	{
	default:
	case 1:
	/* L = D^(-1)A */
		getMatrixPow(degreeMatrix, -1.0);
		laplacianMatrix=degreeMatrix*adjacencyMatrix;
		break;

	case 2:
		Eigen::MatrixXd dMatrix = Eigen::MatrixXd(adjacencyMatrix.rows(),adjacencyMatrix.cols());
		const Eigen::VectorXd& m_v = degreeMatrix.diagonal();
		for(int i=0;i<dMatrix.rows();++i)
			dMatrix(i,i) = m_v(i);
		laplacianMatrix = dMatrix-adjacencyMatrix;
		break;
	}
}

/* decide optimal cluster number by eigenvectors of Laplacian matrix */
void SpectralClustering::getEigenClustering(const Eigen::MatrixXd& laplacianMatrix)
{
	SelfAdjointEigenSolver<MatrixXd> eigensolver(laplacianMatrix);
	std::cout << "Eigen decomposition ends!..." << std::endl;

	// conventional spectral clustering applies this
	const int& eigenRows = presetNumber;

	// streamline embedding algorithm
	//const int& eigenRows = 3;

	Eigen::MatrixXd eigenVec(eigenRows, streamlineCount);

	const int& Row = laplacianMatrix.rows();

	/* from paper we know it should get largest eigenvalues, and from eigen library we know it's latter */
	for(int i=Row-1;i>Row-eigenRows-1;--i)
		eigenVec.row(Row-1-i) = eigensolver.eigenvectors().col(i).transpose();
	eigenVec.transposeInPlace();

	/* k-means as a post-processing */
	if(postProcessing==1)
	{
		normalizeEigenvec(eigenVec);

		performKMeans(eigenVec);

		if(storage.empty())
			return;

	}
	/* eigenvector rotation */
	else if(postProcessing==2)
	{
		getEigvecRotation(eigenVec);

		if(storage.empty())
			return;
	}

	reassignClusterAscending();
}


/* normalize the matrix */
void SpectralClustering::normalizeEigenvec(Eigen::MatrixXd& eigenVec)
{
	const int& rows = eigenVec.rows();
#pragma omp parallel for schedule(static) num_threads(8)
	for(int i=0;i<rows;++i)
	{
		eigenVec.row(i)/=eigenVec.row(i).norm();
	}
}


/* perform k-means clustering */
void SpectralClustering::performKMeans(const Eigen::MatrixXd& eigenVec)
{

	const int& Row = eigenVec.rows();
	const int& Column = eigenVec.cols();

	double moving=1000, tempMoving, before;

	numberOfClusters = presetNumber;

	/* centerTemp is temporary term for storing centroid position, clusterCenter is permanent */
	MatrixXd centerTemp, clusterCenter;

	/* chosen from sample for initialization of k-means */
	Initialization::generateFromSamples(clusterCenter,Column,eigenVec,numberOfClusters);

	int tag = 0;

	storage=std::vector< std::vector<int> >(numberOfClusters);

	double PCA_KMeans_delta, KMeans_delta;

	std::cout << "...k-means started!" << std::endl;

	do
	{
		before = moving;

		centerTemp = MatrixXd::Zero(numberOfClusters, Column);

	#pragma omp parallel for schedule(static) num_threads(8)
		for (int i = 0; i < numberOfClusters; ++i)
		{
			storage[i].clear();
		}

	#pragma omp parallel num_threads(8)
		{
		#pragma omp for nowait
			for (int i = 0; i < Row; ++i)
			{
				double dist = FLT_MAX;
				double temp;
				int clusTemp;
				for (int j = 0; j < numberOfClusters; ++j)
				{
					temp = (eigenVec.row(i)-clusterCenter.row(j)).norm();
					if(temp<dist)
					{
						dist = temp;
						clusTemp = j;
					}
				}

			#pragma omp critical
				{
					storage[clusTemp].push_back(i);
					centerTemp.row(clusTemp)+=eigenVec.row(i);
				}
			}
		}

		moving = FLT_MIN;

	#pragma omp parallel for reduction(max:moving) num_threads(8)
		for (int i = 0; i < numberOfClusters; ++i)
		{
			if(!storage[i].empty())
			{
				centerTemp.row(i)/=storage[i].size();
				tempMoving = (centerTemp.row(i)-clusterCenter.row(i)).norm();
				clusterCenter.row(i) = centerTemp.row(i);
				if(moving<tempMoving)
					moving = tempMoving;
			}
		}
		std::cout << "K-means iteration " << ++tag << " completed, and moving is "
		<< moving << "!" << std::endl;
	}while(abs(moving-before)/before >= 1.0e-3 && tag < 50 && moving>0.01);

	for(auto iter=storage.begin(); iter!=storage.end();)
	{
		if((*iter).empty())
			iter = storage.erase(iter);
		else
			 ++iter;
	}
	numberOfClusters = storage.size();
}


/* get cluster information based on eigenvector rotation */
void SpectralClustering::getEigvecRotation(const Eigen::MatrixXd& X)
{
	double mMaxQuality = 0;
	Eigen::MatrixXd vecRot;
	Eigen::MatrixXd vecIn = X.block(0,0,X.rows(),2);
	Evrot *e = NULL;

	const int& xCols = X.cols();

	std::cout << "Eigenvector rotation starts within " << xCols << " columns..." << std::endl;
	for (int g=2; g <= xCols; g++)
	{
		// make it incremental (used already aligned vectors)
		std::cout << "column " << g << ":";
		if( g > 2 )
		{
			vecIn.resize(X.rows(),g);
			vecIn.block(0,0,vecIn.rows(),g-1) = e->getRotatedEigenVectors();
			vecIn.block(0,g-1,X.rows(),1) = X.block(0,g-1,X.rows(),1);
			delete e;
		}
		//perform the rotation for the current number of dimensions
		e = new Evrot(vecIn, mMethod);

		//save max quality
		if (e->getQuality() > mMaxQuality)
		{
			mMaxQuality = e->getQuality();
		}

		if(std::isnan(e->getQuality())||std::isinf(e->getQuality()))
		{
			std::cout << "Meet with nan or inf! Stop! " << std::endl;
			return;
		}

		std::cout << " max quality is " << mMaxQuality << ", Evrot has quality " << e->getQuality() << std::endl;
		//save cluster data for max cluster or if we're near the max cluster (so prefer more clusters)
		if ((e->getQuality() > mMaxQuality) || (mMaxQuality - e->getQuality() <= 0.001))
		{
			storage = e->getClusters();
			vecRot = e->getRotatedEigenVectors();
		}
	}

	std::cout << "Eigenvector rotation finished..." << std::endl;
	if(storage.empty())
		return;

	numberOfClusters = storage.size();
}

