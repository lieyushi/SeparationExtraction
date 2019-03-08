#include "Initialization.h"

void Initialization::generateRandomPos(MatrixXd& clusterCenter,
								  	   const int& column,
								       const MatrixXd& cArray,
								       const int& Cluster)
{
	clusterCenter = MatrixXd::Random(Cluster, column);
	MatrixXd range(2, column);
	range.row(0) = cArray.colwise().maxCoeff();  //first row contains max
	range.row(1) = cArray.colwise().minCoeff();  //second row contains min
	VectorXd diffRange = range.row(0)-range.row(1);

	MatrixXd diagonalRange = MatrixXd::Zero(column,column);

#pragma omp parallel for schedule(static) num_threads(8)
	for (int i = 0; i < column; ++i)
	{
		diagonalRange(i,i) = diffRange(i);
	}
	clusterCenter = (clusterCenter+MatrixXd::Constant(Cluster,column,1.0))/2.0;

#pragma omp parallel for schedule(static) num_threads(8)
	for (int i = 0; i < Cluster; ++i)
	{
		clusterCenter.row(i) = clusterCenter.row(i)*diagonalRange+range.row(1);
	}
}


void Initialization::generateFromSamples(MatrixXd& clusterCenter,
								    	 const int& column,
								    	 const MatrixXd& cArray,
								    	 const int& Cluster)
{
	clusterCenter = MatrixXd(Cluster,column);
	std::vector<int> number(Cluster);
	srand(time(0));

	const int& MaxNum = cArray.rows();

	std::cout << MaxNum << std::endl;

	number[0] = rand()%MaxNum;
	int randNum, chosen = 1;
	bool found;
	for (int i = 1; i < Cluster; ++i)
	{
		do
		{
			randNum = rand()%MaxNum;
			found = false;
			for(int j=0;j<chosen;j++)
			{
				if(randNum==number[j])
				{
					found = true;
					break;
				}
			}
		}while(found!=false);
		number[i] = randNum;
		++chosen;
	}
	assert(chosen==Cluster);
	assert(column==cArray.cols());

#pragma omp parallel for schedule(static) num_threads(8)
	for (int i = 0; i < Cluster; ++i)
	{
		clusterCenter.row(i) = cArray.row(number[i]);
	}

}
