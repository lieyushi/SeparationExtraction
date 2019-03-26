/*
 * AHC.cpp
 *
 *  Created on: Mar 6, 2019
 *      Author: lieyu
 */

#include "AHC.h"

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

//AHC::AHC(): LineClustering() {
//	// TODO Auto-generated constructor stub
//
//}

AHC::~AHC() {
	// TODO Auto-generated destructor stub
}

AHC::AHC(Eigen::MatrixXd& distanceMatrix, const string& name, const int& vertexCount):
		LineClustering(distanceMatrix, name, vertexCount)
{
	setInputParameters();
}


void AHC::performClustering()
{
	std::unordered_map<int, Ensemble> node_map;
	std::vector<DistNode> dNodeVec;
	std::vector<Ensemble> nodeVec;

	/* set the ditNode vector */
	setValue_merge(dNodeVec, node_map);

	/* perform hiarchical clustering where within each step would merge two nodes */
	hierarchicalMerging(node_map, dNodeVec, nodeVec);

	if(!lMethod)
	{
		storage.resize(numberOfClusters);

		setLabel(nodeVec);

		nodeVec.clear();

		reassignClusterAscending();
	}
}


/* compute distance between two clusters based on likage type */
const double AHC::getDistAtNodes(const vector<int>& firstList, const vector<int>& secondList, const int& Linkage)
{
	const int& m = firstList.size();
	const int& n = secondList.size();
	assert(m!=0);
	assert(n!=0);
	/* 0: single linkage, min(x_i,y_j)
	 * 1: complete linkdage, max(x_i,y_j)
	 * 2: average linkage, sum/x_i/y_j
	 */
	double result, value;
	switch(Linkage)
	{
	case 0:	//single linkage
		{
			result = DBL_MAX;
		#pragma omp parallel for reduction(min:result) num_threads(8)
			for(int i=0;i<m;++i)
			{
				double value;
				for(int j=0;j<n;++j)
				{
					value = distanceMatrix(firstList[i],secondList[j]);
					result = std::min(result, value);
				}
			}
		}
		break;

	case 1:	//complete linkage
		{
			result = -DBL_MAX;
		#pragma omp parallel for reduction(max:result) num_threads(8)
			for(int i=0;i<m;++i)
			{
				for(int j=0;j<n;++j)
				{
					value = distanceMatrix(firstList[i],secondList[j]);
					result = std::max(result, value);
				}
			}
		}
		break;

	case 2:
		{
			result = 0;
		#pragma omp parallel for reduction(+:result) num_threads(8)
			for(int i=0;i<m;++i)
			{
				for(int j=0;j<n;++j)
				{
					value = distanceMatrix(firstList[i], secondList[j]);
					result+=value;
				}
			}
			result/=m*n;
		}
		break;

	default:
		std::cout << "error!" << std::endl;
		exit(1);
	}
	return result;
}


/* perform AHC merging by given a distance threshold */
void AHC::hierarchicalMerging(std::unordered_map<int, Ensemble>& node_map, std::vector<DistNode>& dNodeVec,
		std::vector<Ensemble>& nodeVec)
{

	std::map<int, double> dist_map;

	DistNode poped;

	/* find node-pair with minimal distance */
	double minDist = DBL_MAX;
	int target = -1;
	for (int i = 0; i < dNodeVec.size(); ++i)
	{
		if(dNodeVec[i].distance<minDist)
		{
			target = i;
			minDist = dNodeVec[i].distance;
		}
	}
	poped = dNodeVec[target];

	int index = streamlineCount, currentNumber;
	do
	{
		if(lMethod)
		{
			dist_map.insert(std::make_pair(node_map.size(), poped.distance));
		}
		//create new node merged and input it into hash unordered_map
		vector<int> first = (node_map[poped.first]).element;
		vector<int> second = (node_map[poped.second]).element;

		/* index would be starting from Row */
		Ensemble newNode(index);
		newNode.element = first;
		newNode.element.insert(newNode.element.end(), second.begin(), second.end());
		node_map.insert(make_pair(index, newNode));

		//delete two original nodes
		node_map.erase(poped.first);
		node_map.erase(poped.second);

		/* the difficulty lies how to update the min-heap with linkage
		 * This would take 2NlogN.
		 * Copy all node-pairs that are not relevant to merged nodes to new vec.
		 * For relevant, would update the mutual distance by linkage
		 */

		/* how many clusters exist */
		currentNumber = node_map.size();

		target = -1, minDist = DBL_MAX;

		std::vector<DistNode> tempVec(currentNumber*(currentNumber-1)/2);

		int current = 0, i_first, i_second;
		for(int i=0;i<dNodeVec.size();++i)
		{
			i_first=dNodeVec[i].first, i_second=dNodeVec[i].second;
			/* not relevant, directly copied to new vec */
			if(i_first!=poped.first&&i_first!=poped.second&&i_second!=poped.first&&i_second!=poped.second)
			{
				tempVec[current]=dNodeVec[i];
				if(tempVec[current].distance<minDist)
				{
					target = current;
					minDist = tempVec[current].distance;
				}
				++current;
			}
		}

		for (auto iter=node_map.begin();iter!=node_map.end();++iter)
		{
			if((*iter).first!=newNode.index)
			{
				tempVec[current].first = (*iter).first;
				tempVec[current].second = newNode.index;
				tempVec[current].distance=getDistAtNodes(newNode.element,(*iter).second.element, linkageOption);
				if(tempVec[current].distance<minDist)
				{
					target = current;
					minDist = tempVec[current].distance;
				}
				++current;
			}
		}

		/* judge whether current is assigned to right value */
		assert(current==tempVec.size());

		if(target>=0 && tempVec.size()>=1)
		{
			poped = tempVec[target];
			dNodeVec.clear();
			dNodeVec = tempVec;
			tempVec.clear();
			++index;
		}
		else if(target==-1)
			break;

	}while(node_map.size()!=numberOfClusters);	//merging happens whenever requested cluster is not met

	if(lMethod)
	{
		/* perform L-method computation to detect optimal number of AHC */
		DetermClusterNum dcn;
		dcn.iterativeRefinement(dist_map);
		std::cout << "Otimal number of clusters by L-Method is " << dcn.getFinalNumOfClusters() << std::endl;
	}

	else
	{
		nodeVec=std::vector<Ensemble>(node_map.size());
		int tag = 0;
		for(auto iter=node_map.begin();iter!=node_map.end();++iter)
			nodeVec[tag++]=(*iter).second;

		/* task completed, would delete memory contents */
		dNodeVec.clear();
		node_map.clear();
		/* use alpha function to sort the group by its size */
		std::sort(nodeVec.begin(), nodeVec.end(), [](const Ensemble& e1, const Ensemble& e2)
		{return e1.element.size()<e2.element.size()||(e1.element.size()==e2.element.size()&&e1.index<e2.index);});
	}
}


/* set a vector for min-heap */
void AHC::setValue_merge(std::vector<DistNode>& dNodeVec, std::unordered_map<int, Ensemble>& node_map)
{
	/* find the node of closest distance */
	std::vector<int> miniNode(streamlineCount);
#pragma omp parallel for schedule(static) num_threads(8)
	for(int i=0;i<streamlineCount;++i)
	{
		double miniDist = DBL_MAX, dist;
		int index = -1;
		for(int j=0;j<streamlineCount;++j)
		{
			if(i==j)
				continue;
			dist = distanceMatrix(i,j);

			if(miniDist>dist)
			{
				miniDist=dist;
				index=j;
			}
		}
		miniNode[i]=index;
	}

	std::vector<bool> isIn(streamlineCount, false);

	int tag = 0;
	for(int i=0;i<streamlineCount;++i)
	{
		if(!isIn[i])
		{
			Ensemble en;
			if(miniNode[miniNode[i]]==i)
			{
				en.element.push_back(i);
				en.element.push_back(miniNode[i]);
				isIn[i]=true;
				isIn[miniNode[i]]=true;
				node_map[tag] = en;
			}
			else
			{
				en.element.push_back(i);
				isIn[i]=true;
				node_map[tag] = en;
			}
			++tag;
		}
	}

	const int& mapSize = node_map.size();
	dNodeVec = std::vector<DistNode>(mapSize*(mapSize-1)/2);

	tag = 0;
	for(auto start = node_map.begin(); start!=node_map.end(); ++start)
	{
		for(auto end = node_map.begin(); end!=node_map.end() && end!=start; ++end)
		{
			dNodeVec[tag].first = start->first;
			dNodeVec[tag].second = end->first;
			dNodeVec[tag].distance = getDistAtNodes(start->second.element,end->second.element, linkageOption);
			++tag;
		}
	}
	assert(tag==dNodeVec.size());
}


/* set input parameters for AHC clustering */
void AHC::setInputParameters()
{
	std::cout << "Perform L-method to detect optimal num of clusters? 0: No, 1: Yes! " << std::endl;
	std::cin >> lMethod;
	assert(lMethod==0 || lMethod==1);
	lMethod = (lMethod==1);

	std::cout << "---------------------------" << std::endl;
	std::cout << "Input linkage option: 0.single linkage, 1.complete linkage, 2.average linkage" << std::endl;
	std::cin >> linkageOption;
	assert(linkageOption==0||linkageOption==1||linkageOption==2);

	if(!lMethod)
	{
		std::cout << "---------------------------" << std::endl;
		std::cout << "Choose cluster number input between [1," << streamlineCount << "]: " << std::endl;
		std::cin >> numberOfClusters;
		assert(numberOfClusters>=1&&numberOfClusters<=streamlineCount);
	}
	else
	{
		numberOfClusters = 1;
	}
}


/* perform group-labeling information */
void AHC::setLabel(const std::vector<Ensemble>& nodeVec)
{
	// group tag by increasing order
	int id = 0;

	// element list for each group
	vector<int> eachContainment;

	groupID.resize(streamlineCount);

	// find group id and neighboring vec
	for(auto iter = nodeVec.begin(); iter!=nodeVec.end();++iter)
	{
		eachContainment = (*iter).element;
		storage[id] = eachContainment;
	#pragma omp parallel for schedule(static) num_threads(8)
		for(int i=0;i<eachContainment.size();++i)
		{
			groupID[eachContainment[i]] = id;
		}
		++id;
		eachContainment.clear();
	}
}
