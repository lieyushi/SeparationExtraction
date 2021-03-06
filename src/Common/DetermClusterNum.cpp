/*
 * DetermClusterNum.cpp
 *
 *  Created on: Aug 22, 2018
 *      Author: lieyu
 */

#include "DetermClusterNum.h"

DetermClusterNum::DetermClusterNum() {
	// TODO Auto-generated constructor stub

}

DetermClusterNum::~DetermClusterNum() {
	// TODO Auto-generated destructor stub
}


/* use iterative refinement of knee to get optimal number for hierarchical clustering */
void DetermClusterNum::iterativeRefinement(std::map<int, double>& eval_graph)
{
	removeExtreme(eval_graph);

	int cutoff, lastKnee;
	int currentKnee = eval_graph.rbegin()->first;
	cutoff = currentKnee;
	do
	{
		lastKnee = currentKnee;
		currentKnee = LMethod(eval_graph, cutoff);
		std::cout << "returned value is " << currentKnee <<", cutoff is " << cutoff << std::endl;
		cutoff = currentKnee*2;
	}while(currentKnee < lastKnee);

	finalNumOfClusters = currentKnee;

	std::cout << finalNumOfClusters << std::endl;
}


/* return a knee by L method given a cutoff */
const int DetermClusterNum::LMethod(const std::map<int, double>& eval_graph, const int& cutoff)
{
	struct CompObj { double val; int index; };
// #pragma omp declare reduction(minimum : struct CompObj : omp_out = omp_in.val < omp_out.val ? omp_in : omp_out)
	struct CompObj RMSE;
	RMSE.val = DBL_MAX;
	RMSE.index = -1;

	const int& firstIndex = eval_graph.begin()->first;
	/* find the minimal c that minimizes RMSE for the selected cutoff */
#pragma omp parallel num_threads(8)
	{
	#pragma omp nowait
		for(int i=firstIndex;i<=cutoff;++i)
		{
			/* left segment linear least square fitting */
			std::vector<double> index_vec;
			std::vector<double> dist_vec;

			std::map<int, double>::const_iterator iter;
			for(int j=firstIndex;j<=i;++j)
			{
				iter = eval_graph.find(j);
				if(iter!=eval_graph.end())
				{
					index_vec.push_back(iter->first);
					dist_vec.push_back(iter->second);
				}
			}
			Eigen::MatrixXd A_sub(2, index_vec.size());
			A_sub.row(0) = Eigen::VectorXd::Map(&(index_vec[0]), index_vec.size()).transpose();
			A_sub.row(1) = Eigen::VectorXd::Constant(index_vec.size(), 1.0).transpose();
			Eigen::VectorXd b_sub = Eigen::VectorXd::Map(&(dist_vec[0]), index_vec.size());
			A_sub.transposeInPlace();
			int firstRows = A_sub.rows();

			Eigen::VectorXd c = A_sub.colPivHouseholderQr().solve(b_sub);
			Eigen::VectorXd error = b_sub-A_sub*c;
			double rmse_l = error.transpose()*error;

			/* right segment linear least square fitting */
			index_vec.clear();
			dist_vec.clear();

			for(int j=i+1;j<=cutoff;++j)
			{
				iter = eval_graph.find(j);
				if(iter!=eval_graph.end())
				{
					index_vec.push_back(iter->first);
					dist_vec.push_back(iter->second);
				}
			}
			A_sub = Eigen::MatrixXd(2, index_vec.size());
			A_sub.row(0) = Eigen::VectorXd::Map(&(index_vec[0]), index_vec.size()).transpose();
			A_sub.row(1) = Eigen::VectorXd::Constant(index_vec.size(), 1.0).transpose();
			b_sub = Eigen::VectorXd::Map(&(dist_vec[0]), index_vec.size());
			A_sub.transposeInPlace();
			int secondRows = A_sub.rows();

			c = A_sub.colPivHouseholderQr().solve(b_sub);
			error = b_sub-A_sub*c;
			double rmse_r = error.transpose()*error;

			/* compute the total weighted error */
			double rmse = double(firstRows)/double(firstRows+secondRows)*rmse_l+
					double(secondRows)/double(firstRows+secondRows)*rmse_r;

		#pragma omp critical
			if(RMSE.val>rmse)
			{
				RMSE.val=rmse;
				RMSE.index=i;
			}
		}
	}

	return RMSE.index;
}


/* write the number in a file */
void DetermClusterNum::recordLMethodResult(const int& normOption)
{
	std::ofstream readme("../dataset/LMethod",ios::out | ios::app);
	if(!readme)
	{
		std::cout << "Error creating readme!" << std::endl;
		exit(1);
	}
	readme << "Optimal cluster number of norm " << normOption << " is " << finalNumOfClusters << std::endl;
	readme << std::endl;
	readme.close();
}

/* remove extremely dissimilar cluster merges */
void DetermClusterNum::removeExtreme(std::map<int, double>& eval_graph)
{
	double maxDist = -1.0;
	int leftIndex = -1;
	for(auto iter:eval_graph)
	{
		if(maxDist<iter.second)
		{
			maxDist=iter.second;
			leftIndex=iter.first;
		}
	}
	auto iter_index = eval_graph.find(leftIndex);

	for(auto iter=eval_graph.begin();iter!=iter_index;)
	{
		if(iter->first<leftIndex&&iter->second<maxDist)
			eval_graph.erase(iter++);
		else
			++iter;
	}
}
