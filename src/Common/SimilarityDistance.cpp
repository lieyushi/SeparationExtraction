/*
 * SimilarityDistance.cpp
 *
 *  Created on: Feb 4, 2019
 *      Author: lieyu
 */

#include "SimilarityDistance.h"

SimilarityDistance::SimilarityDistance() {
	// TODO Auto-generated constructor stub

}

SimilarityDistance::~SimilarityDistance() {
	// TODO Auto-generated destructor stub
}

// get closest distance and record their index of pair
const double SimilarityDistance::getClosestPair(const Eigen::VectorXd& first, const Eigen::VectorXd& second,
											    std::pair<int,int>& closest, double& mcp_dist)
{
	const int& first_size = first.size()/3;
	const int& second_size = second.size()/3;

	double result, s_to_f=DBL_MAX, magnitude;
	mcp_dist = 0.0;
	for(int i=0;i<first_size;++i)
	{
		double minDist = DBL_MAX;
		int closest_second = -1;
		Vector3d m_i = Vector3d(first(3*i),first(3*i+1),first(3*i+2));
		for(int j=0;j<second_size;++j)
		{
			Vector3d n_j = Vector3d(second(3*j),second(3*j+1),second(3*j+2));
			magnitude = (m_i-n_j).norm();
			if(magnitude<minDist)
			{
				minDist = magnitude;
				closest_second = j;
			}
		}
		mcp_dist+=minDist;
		if(minDist<s_to_f)
		{
			s_to_f=minDist;
			closest.first = i;
			closest.second = closest_second;
		}
	}
	mcp_dist/=first_size;
	return s_to_f;
}


/* a chi-test for discrete curvatures with each attribute on each vertex */
const double SimilarityDistance::getChiTestOfCurvatures(const Eigen::VectorXd& first, const Eigen::VectorXd& second,
			const std::pair<int,int>& closest, const bool& useSegments)
{
	double result = 0;
	const int& firstSize = first.size()/3;
	const int& secondSize = second.size()/3;

	int firstHalf = std::min(closest.first, closest.second);
	int secondHalf = std::min(firstSize-closest.first, secondSize-closest.second);

	double left_half = 0.0, right_half = 0.0, firstDotValue, leftNorm, rightNorm;
	Eigen::Vector3d left, right, crossVec;
	if(!useSegments)
	{
		/* if 5, 0->5 has 4 discrete curvatures */
		firstHalf-=1;
		secondHalf-=1;

		if(firstHalf>1)
			left_half = getChiTestOfCurvaturesOnVertex(first, second, firstHalf, closest, "-");
		if(firstHalf>1)
			right_half = getChiTestOfCurvaturesOnVertex(first, second, secondHalf, closest, "+");

		result = left_half*(double(firstHalf)/double(firstHalf+secondHalf))
				+right_half*(double(secondHalf)/double(firstHalf+secondHalf));
	}
	else
	{
		firstHalf-=1;
		secondHalf-=1;

		if(firstHalf>1)
			left_half = getChiTestOfCurvaturesOnSegments(first, second, firstHalf, closest, "-");
		if(firstHalf>1)
			right_half = getChiTestOfCurvaturesOnSegments(first, second, secondHalf, closest, "+");

		result = left_half*(double(firstHalf)/double(firstHalf+secondHalf))
				+right_half*(double(secondHalf)/double(firstHalf+secondHalf));
	}

	return result;
}


/* a chi-test with two equal-size attribute arrays */
/*
 * Implementation can be seen https://github.com/lieyushi/FlowCurveClustering/blob/master/src/Common/Distance.cpp
 */
const double SimilarityDistance::getChiTest(const Eigen::VectorXd& first, const Eigen::VectorXd& second)
{
	double chi_test = 0.0, difference, summation;
	assert(first.size()==second.size());
	for(int i=0; i<first.size(); ++i)
	{
		difference = double(first(i)-second(i));
		difference = difference*difference;
		summation = double(first(i)+second(i));
		if(summation<1.0E-8)
			continue;
		chi_test += difference/summation;
	}
	return chi_test;
}


/* chi-test with two attribute arrays that have discrete curvatures over each vertex */
const double SimilarityDistance::getChiTestOfCurvaturesOnVertex(const Eigen::VectorXd& first,
		const Eigen::VectorXd& second, const int& firstHalf, const std::pair<int,int>& closest, const string& sign)
{
	double result = 0;

	bool isIncreased=(sign=="+");
	int current;
	Eigen::Vector3d left, right, crossVec;
	double dotValue, leftNorm, rightNorm, firstAngle, secondAngle, angleDiff, angleSum;
	for(int i=0; i<firstHalf; ++i)
	{
		/*
		 * get the signed discrete curvatures for the first streamline segments
		 */
		if(isIncreased)
			current=closest.first+i;
		else
			current=closest.first-2-i;

		left << first(3*i+3)-first(3*i), first(3*i+4)-first(3*i+1), first(3*i+5)-first(3*i+2);
		right << first(3*i+6)-first(3*i+3), first(3*i+7)-first(3*i+4), first(3*i+8)-first(3*i+5);

		// get the discrete curvatures
		dotValue = left.dot(right);
		leftNorm = left.norm();
		rightNorm = right.norm();
		if(leftNorm>=1.0E-8 && rightNorm>=1.0E-8)
		{
			dotValue = dotValue/leftNorm/rightNorm;
			dotValue = std::min(1.0, dotValue);
			dotValue = std::max(-1.0, dotValue);
		}
		firstAngle = acos(dotValue);
		// get its sign symbol, either + or -
		crossVec = left.cross(right);
		if(crossVec(2)<0)
			firstAngle = -firstAngle;

		/*
		 * get the signed discrete curvatures for the second streamline segments
		 */
		if(isIncreased)
			current=closest.second+i;
		else
			current=closest.second-2-i;
		left << second(3*i+3)-second(3*i), second(3*i+4)-second(3*i+1), second(3*i+5)-second(3*i+2);
		right << second(3*i+6)-second(3*i+3), second(3*i+7)-second(3*i+4), second(3*i+8)-second(3*i+5);

		// get the discrete curvatures
		dotValue = left.dot(right);
		leftNorm = left.norm();
		rightNorm = right.norm();
		if(leftNorm>=1.0E-8 && rightNorm>=1.0E-8)
		{
			dotValue = dotValue/leftNorm/rightNorm;
			dotValue = std::min(1.0, dotValue);
			dotValue = std::max(-1.0, dotValue);
		}
		secondAngle = acos(dotValue);
		// get its sign symbol, either + or -
		crossVec = left.cross(right);
		if(crossVec(2)<0)
			secondAngle = -secondAngle;

		/*
		 * compute the chi-test of two angles
		 */
		angleDiff = firstAngle-secondAngle;
		angleDiff = angleDiff*angleDiff;
		angleSum = firstAngle+secondAngle;
		if(angleSum<1.0E-8)
			continue;
		result += angleDiff/angleSum;
	}

	return result;
}


/* chi-test with two attribute arrays that have discrete curvatures over segments */
const double SimilarityDistance::getChiTestOfCurvaturesOnSegments(const Eigen::VectorXd& first,
		const Eigen::VectorXd& second, const int& firstHalf, const std::pair<int,int>& closest, const string& sign)
{
	double result = 0;

	bool isIncreased=(sign=="+");
	int current;
	Eigen::Vector3d left, right, crossVec;
	double dotValue, leftNorm, rightNorm, firstAngle, secondAngle, angleDiff, angleSum;

	int heapSize = int(log2(firstHalf));

	std::priority_queue<CurvatureObject, std::vector<CurvatureObject>, CompareFunc> firstQueue;
	std::priority_queue<CurvatureObject, std::vector<CurvatureObject>, CompareFunc> secondQueue;

	Eigen::VectorXd firstCurvatureVec(firstHalf), secondCurvatureVec(firstHalf);
	for(int i=0; i<firstHalf; ++i)
	{
		/*
		 * get the signed discrete curvatures for the first streamline segments
		 */
		if(isIncreased)
			current=closest.first+i;
		else
			current=closest.first-2-i;

		left << first(3*i+3)-first(3*i), first(3*i+4)-first(3*i+1), first(3*i+5)-first(3*i+2);
		right << first(3*i+6)-first(3*i+3), first(3*i+7)-first(3*i+4), first(3*i+8)-first(3*i+5);

		// get the discrete curvatures
		dotValue = left.dot(right);
		leftNorm = left.norm();
		rightNorm = right.norm();
		if(leftNorm>=1.0E-8 && rightNorm>=1.0E-8)
		{
			dotValue = dotValue/leftNorm/rightNorm;
			dotValue = std::min(1.0, dotValue);
			dotValue = std::max(-1.0, dotValue);
		}
		firstAngle = acos(dotValue);
		// get its sign symbol, either + or -
		crossVec = left.cross(right);
		if(crossVec(2)<0)
			firstAngle = -firstAngle;

		firstQueue.push(CurvatureObject(firstAngle, i));
		if(firstQueue.size()>heapSize)
			firstQueue.pop();

		firstCurvatureVec(i) = firstAngle;
		/*
		 * get the signed discrete curvatures for the second streamline segments
		 */
		if(isIncreased)
			current=closest.second+i;
		else
			current=closest.second-2-i;
		left << second(3*i+3)-second(3*i), second(3*i+4)-second(3*i+1), second(3*i+5)-second(3*i+2);
		right << second(3*i+6)-second(3*i+3), second(3*i+7)-second(3*i+4), second(3*i+8)-second(3*i+5);

		// get the discrete curvatures
		dotValue = left.dot(right);
		leftNorm = left.norm();
		rightNorm = right.norm();
		if(leftNorm>=1.0E-8 && rightNorm>=1.0E-8)
		{
			dotValue = dotValue/leftNorm/rightNorm;
			dotValue = std::min(1.0, dotValue);
			dotValue = std::max(-1.0, dotValue);
		}
		secondAngle = acos(dotValue);
		// get its sign symbol, either + or -
		crossVec = left.cross(right);
		if(crossVec(2)<0)
			secondAngle = -secondAngle;

		secondQueue.push(CurvatureObject(secondAngle, i));
		if(secondQueue.size()>heapSize)
			secondQueue.pop();

		secondCurvatureVec(i) = secondAngle;
	}

	/* get the index list of both segments */
	std::vector<int> firstIndex, secondIndex;
	while(!firstQueue.empty() && !secondQueue.empty())
	{
		firstIndex.push_back(firstQueue.top().index);
		firstQueue.pop();
		secondIndex.push_back(secondQueue.top().index);
		secondQueue.pop();
	}

	std::sort(firstIndex.begin(), firstIndex.end());
	std::sort(secondIndex.begin(), secondIndex.end());

	double firstSum, secondSum;
	int l_index_0 = 0, l_index_1 = 0, r_index;
	for(int i=0; i<heapSize+1; ++i)
	{
		if(i!=heapSize)
		{
			r_index = firstIndex[i];
			firstSum = 0.0;
			for(int j=l_index_0; j<=r_index; ++j)
				firstSum+=firstCurvatureVec(j);
			l_index_0 = r_index+1;

			r_index = secondIndex[i];
			secondSum = 0.0;
			for(int j=l_index_1; j<=r_index; ++j)
				secondSum+=secondCurvatureVec(j);

			angleDiff = firstSum-secondSum;
			angleSum = firstSum+secondSum;
			if(angleSum<1.0E-8)
				continue;
			result+=angleDiff*angleDiff/angleSum;

			l_index_1 = r_index+1;
		}
		else
		{
			r_index = firstCurvatureVec.size()-1;
			firstSum = 0.0;
			for(int j=l_index_0; j<=r_index; ++j)
				firstSum+=firstCurvatureVec(j);
			l_index_0 = r_index+1;

			r_index = secondCurvatureVec.size()-1;
			secondSum = 0.0;
			for(int j=l_index_1; j<=r_index; ++j)
				secondSum+=secondCurvatureVec(j);

			angleDiff = firstSum-secondSum;
			angleSum = firstSum+secondSum;
			if(angleSum<1.0E-8)
				continue;
			result+=angleDiff*angleDiff/angleSum;
		}
	}
	return result;
}


/* based on signature-based segmentation to get segments for input streamlines */
void SimilarityDistance::computeSegments(std::vector<Eigen::VectorXd>& lineSegments,
		std::vector<Eigen::VectorXd>& lineCurvatures, std::vector<std::vector<int> >& segmentsToLines,
		int& accumulated, const std::vector<Eigen::VectorXd>& streamlineVector)
{
	accumulated = 0;
	const int& streamlineSize = streamlineVector.size();
	{
		for(int i=0; i<streamlineSize; ++i)
		{
			// get streamline vector
			const Eigen::VectorXd& streamline = streamlineVector.at(i);
			const int& vertexCount = streamline.size()/3;
			const int& segmentNumber = int(log2(vertexCount));

			// should partition the streamline into segmentNumber segments
			std::priority_queue<CurvatureObject, std::vector<CurvatureObject>, CompareFunc> firstQueue;
			Eigen::VectorXd curvatureVec(vertexCount-2);
			Eigen::Vector3d left, right, crossVec;

			std::vector<int>& segmentList = segmentsToLines[i];
			double dotValue, leftNorm, rightNorm, angle;

			for(int j=0; j<vertexCount-2; ++j)
			{
				/*
				 * get the signed discrete curvatures for the first streamline segments
				 */
				left << streamline(3*j+3)-streamline(3*j),
						streamline(3*j+4)-streamline(3*j+1),
						streamline(3*j+5)-streamline(3*j+2);
				right << streamline(3*j+6)-streamline(3*j+3),
						streamline(3*j+7)-streamline(3*j+4),
						streamline(3*j+8)-streamline(3*j+5);

				// get the discrete curvatures
				dotValue = left.dot(right);
				leftNorm = left.norm();
				rightNorm = right.norm();
				if(leftNorm>=1.0E-8 && rightNorm>=1.0E-8)
				{
					dotValue = dotValue/leftNorm/rightNorm;
					dotValue = std::min(1.0, dotValue);
					dotValue = std::max(-1.0, dotValue);
				}
				angle = acos(dotValue);
				// get its sign symbol, either + or -
				crossVec = left.cross(right);
				if(crossVec(2)<0)
					angle = -angle;

				firstQueue.push(CurvatureObject(angle, j));
				if(firstQueue.size()>segmentNumber)
					firstQueue.pop();

				curvatureVec(j) = angle;
			}

			/* get the index list of both segments */
			std::vector<int> firstIndex;
			while(!firstQueue.empty())
			{
				firstIndex.push_back(firstQueue.top().index);
				firstQueue.pop();
			}

			std::sort(firstIndex.begin(), firstIndex.end());
			segmentList = std::vector<int>(segmentNumber+1);

			int l_index_0 = 0, r_index;
			for(int j=0; j<segmentNumber+1; ++j)
			{
				Eigen::VectorXd& curvatureOfEachLine = lineCurvatures[accumulated+j];
				Eigen::VectorXd& segmentCoordinate = lineSegments[accumulated+j];
				if(j!=segmentNumber)
				{
					r_index = firstIndex[j];

					curvatureOfEachLine = Eigen::VectorXd(r_index-l_index_0+1);
					/*
					 * curvature index is 0,1,2. But vertex index is 0,1,2,3
					 */
					if(j==0)
					{
						segmentCoordinate = Eigen::VectorXd((r_index-l_index_0+2)*3);
						// add first vertex to first segment
						for(int k=0; k<3; ++k)
							segmentCoordinate(k) = streamline(k);
						// add vertices with index of segmentIndex+1 to segment
						for(int k=l_index_0; k<=r_index; ++k)
						{
							// firstSum+=curvatureVec[k];
							curvatureOfEachLine(k-l_index_0) = curvatureVec(k);
							for(int l=0; l<3; ++l)
							{
								segmentCoordinate(3*(k-l_index_0+1)+l) = streamline(3*(k+1)+l);
							}
						}
					}
					/*
					 * curvature index is 3,4,5, and vertex index is 4,5,6
					 */
					else
					{
						segmentCoordinate = Eigen::VectorXd((r_index-l_index_0+1)*3);
						// add vertices with index of segmentIndex+1 to segment
						for(int k=l_index_0; k<=r_index; ++k)
						{
							// firstSum+=curvatureVec[k];
							curvatureOfEachLine(k-l_index_0) = curvatureVec(k);
							for(int l=0; l<3; ++l)
							{
								segmentCoordinate(3*(k-l_index_0)+l) = streamline(3*(k+1)+l);
							}
						}
					}
					l_index_0 = r_index+1;

				}
				/*
				 * curvature index is 9. But vertex index is 10, 11
				 */
				else
				{
					r_index = curvatureVec.size()-1;
					curvatureOfEachLine = Eigen::VectorXd(r_index-l_index_0+1);
					segmentCoordinate = Eigen::VectorXd((r_index-l_index_0+2)*3);
					for(int k=l_index_0; k<=r_index; ++k)
					{
						// firstSum+=curvatureVec[k];
						curvatureOfEachLine(k-l_index_0) = curvatureVec(k);
						for(int l=0; l<3; ++l)
						{
							segmentCoordinate(3*(k-l_index_0)+l) = streamline(3*(k+1)+l);
						}
					}

					// add last index
					for(int k=0; k<3; ++k)
					{
						segmentCoordinate(3*(r_index-l_index_0+1)+k) = streamline(3*(r_index+2)+k);
					}

					l_index_0 = r_index+1;
				}

				segmentList[j] = accumulated+j;
			}
			accumulated += segmentNumber+1;
		}
	}
}


/* get chi-test-distance given two curvature array */
const double SimilarityDistance::getChiTestPair(const Eigen::VectorXd& first, const Eigen::VectorXd& second)
{
	int firstSize = first.size();
	int secondSize = second.size();
	/*
	 * just like FlowString method, choose the smallest length
	 */
	const int& shorterLength = std::min(firstSize, secondSize);

	double chi_test = 0.0, difference, summation;
	for(int i=0; i<shorterLength; ++i)
	{
		difference = double(first(i)-second(i));
		difference = difference*difference;
		summation = double(first(i)+second(i));
		if(summation<1.0E-8)
			continue;
		chi_test += difference/summation;
	}
	return chi_test;
}

/* get the streamline id given segment number */
const int SimilarityDistance::getStreamlineID(const int& segmentID,
		const std::vector<std::vector<int> >& segmentOfStreamlines)
{
	assert(segmentID>=0 && segmentID<=segmentOfStreamlines.back().back());
	for(int i=0; i<segmentOfStreamlines.size(); ++i)
	{
		if(segmentID<=segmentOfStreamlines[i].back())
			return i;
	}
	return -1;
}

