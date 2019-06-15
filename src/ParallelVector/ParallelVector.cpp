/*
 * ParallelVector.cpp
 *
 *  Created on: Apr 27, 2019
 *      Author: lieyu
 */

#include "ParallelVector.h"

//ParallelVector::ParallelVector() {
//	// TODO Auto-generated constructor stub
//
//}

ParallelVector::~ParallelVector() {
	// TODO Auto-generated destructor stub
}


/*
 * calculate the PV operator for input vector field
 */
ParallelVector::ParallelVector(const string& fileName, std::vector<Vertex>& vertexVec, CoordinateLimits li[3],
		const int& x_resolution, const int& y_resolution, const int& z_resolution,
		const double& x_step, const double& y_step, const double& z_step): pvMethodOption(-1), dataset_name(fileName),
		X_RESOLUTION(x_resolution), Y_RESOLUTION(y_resolution), Z_RESOLUTION(z_resolution), X_STEP(x_step),
		Y_STEP(y_step), Z_STEP(z_step), vertexVec(vertexVec)
{
	for(int i=0; i<3; ++i)
	{
		limits[i] = li[i];
		std::cout << "[" << limits[i].inf << "," << limits[i].sup << "]" << std::endl;
	}
}

// perform parallel vector operation
void ParallelVector::performPVOperation()
{
	getJacobianOnGridPoints();

	// decide which strategy for parallel vector operator
	std::cout << "Which PV operator is opted? 1.vX(Jv)==0, 2.vXe_i: " << std::endl;
	std::cin >> pvMethodOption;
	assert(pvMethodOption==1 || pvMethodOption==2);

	if(Z_RESOLUTION==1)
	{
	#pragma omp parallel num_threads(8)
		{
		#pragma omp for nowait
			for(int j=1; j<Y_RESOLUTION-1; ++j)
			{
				for(int i=1; i<X_RESOLUTION-1; ++i)
				{
					std::vector<Eigen::Vector3d> solutionPoint;
					getSolutionFor2D(i,j,solutionPoint);
				#pragma omp critical
					{
						pvPointVec.insert(pvPointVec.end(), solutionPoint.begin(), solutionPoint.end());
					}
				}
			}
		}
	}
	else
	{
#pragma omp parallel num_threads(8)
		{
		#pragma omp for nowait
			for(int k=1; k<Z_RESOLUTION-1; ++k)
			{
				for(int j=1; j<Y_RESOLUTION-1; ++j)
				{
					for(int i=1; i<X_RESOLUTION-1; ++i)
					{
						std::vector<Eigen::Vector3d> solutionPoint;
						getSolutionFor3D(i,j,k,solutionPoint);
					#pragma omp critical
						{
							pvPointVec.insert(pvPointVec.end(), solutionPoint.begin(), solutionPoint.end());
						}
					}
				}
			}
		}
	}
}


// calculate the Jacobian values on the grid points based on finite difference method
void ParallelVector::getJacobianOnGridPoints()
{
	// assign length for vector, however, we will only consider
	JacobianGridVec.resize(vertexVec.size(), Eigen::Matrix3d::Zero());

	// notice that we will ignore the Jacobian computation on the boundary point, i.e.,
	// if x_resolution = 64, only consider 1-63 that have two neighbors

	// in case z_resolution == 1 for those 2D vector field
	if(Z_RESOLUTION==1)
	{
	#pragma omp parallel for schedule(static) num_threads(8)
		for(int j=1; j<Y_RESOLUTION-2; ++j)
		{
			for(int k=1; k<X_RESOLUTION-2; ++k)
			{
				const int& current = j*X_RESOLUTION+k;
				int left = current-1, right = current+1, front = (j-1)*X_RESOLUTION+k, back = (j+1)*X_RESOLUTION+k;
				Eigen::Matrix3d& jacobian = JacobianGridVec[current];
				jacobian(0,0) = (vertexVec[right].vx-vertexVec[left].vx)/2.0/X_STEP;
				jacobian(0,1) = (vertexVec[back].vx-vertexVec[front].vx)/2.0/Y_STEP;
				jacobian(1,0) = (vertexVec[right].vy-vertexVec[left].vy)/2.0/X_STEP;
				jacobian(1,1) = (vertexVec[back].vy-vertexVec[front].vy)/2.0/Y_STEP;
			}
		}
	}
	// normal 3D data sets
	else
	{
		const int& XY_MULTIPLY = X_RESOLUTION*Y_RESOLUTION;
	#pragma omp parallel for schedule(static) num_threads(8)
		for(int i=1; i<Z_RESOLUTION-2; ++i)
		{
			for(int j=1; j<Y_RESOLUTION-2; ++j)
			{
				for(int k=1; k<X_RESOLUTION-2; ++k)
				{
					const int& current = i*XY_MULTIPLY+j*X_RESOLUTION+k;
					int left = current-1, right = current+1, front = i*XY_MULTIPLY+(j-1)*X_RESOLUTION+k,
						back = i*XY_MULTIPLY+(j+1)*X_RESOLUTION+k, bottom = (i-1)*XY_MULTIPLY+j*X_RESOLUTION+k,
						top = (i+1)*XY_MULTIPLY+j*X_RESOLUTION+k;
					Eigen::Matrix3d& jacobian = JacobianGridVec[current];
					jacobian(0,0) = (vertexVec[right].vx-vertexVec[left].vx)/2.0/X_STEP;
					jacobian(0,1) = (vertexVec[back].vx-vertexVec[front].vx)/2.0/Y_STEP;
					jacobian(0,2) = (vertexVec[top].vx-vertexVec[bottom].vx)/2.0/Z_STEP;
					jacobian(1,0) = (vertexVec[right].vy-vertexVec[left].vy)/2.0/X_STEP;
					jacobian(1,1) = (vertexVec[back].vy-vertexVec[front].vy)/2.0/Y_STEP;
					jacobian(1,2) = (vertexVec[top].vy-vertexVec[bottom].vy)/2.0/Z_STEP;
					jacobian(2,0) = (vertexVec[right].vz-vertexVec[left].vz)/2.0/X_STEP;
					jacobian(2,1) = (vertexVec[back].vz-vertexVec[front].vz)/2.0/Y_STEP;
					jacobian(2,2) = (vertexVec[top].vz-vertexVec[bottom].vz)/2.0/Z_STEP;
				}
			}
		}
	}

	std::cout << "Jacobian computation finished on the grid points!" << std::endl;
}


// get PV solution point within the triangle w.r.t. eigen-decomposition method
void ParallelVector::getEigenDecompositionSolution(const std::tuple<int,int,int>& triangleIndex,
		std::vector<Eigen::Vector3d>& solutionPoint)
{
	// get the index of vertices
	const int& i = std::get<0>(triangleIndex);
	const int& j = std::get<1>(triangleIndex);
	const int& k = std::get<2>(triangleIndex);

	Eigen::Matrix3d invMatrix;
	invMatrix(0,0)=1, invMatrix(0,1)=0, invMatrix(0,2)=0;
	invMatrix(1,0)=0, invMatrix(1,1)=1, invMatrix(1,2)=0;
	invMatrix(2,0)=-1, invMatrix(2,1)=-1, invMatrix(2,2)=1;

	// calculate V = [v1, v2, v3]*invMatrix
	Eigen::Matrix3d velocityMatrix;
	velocityMatrix(0,0)=vertexVec[i].vx, velocityMatrix(1,0)=vertexVec[i].vy, velocityMatrix(2,0)=vertexVec[i].vz;
	velocityMatrix(0,1)=vertexVec[j].vx, velocityMatrix(1,1)=vertexVec[j].vy, velocityMatrix(2,1)=vertexVec[j].vz;
	velocityMatrix(0,2)=vertexVec[k].vx, velocityMatrix(1,2)=vertexVec[k].vy, velocityMatrix(2,2)=vertexVec[k].vz;

	Eigen::Matrix3d V = velocityMatrix*invMatrix;

	Eigen::Matrix3d accelerationMatrix;
	// calculate W = [w1, w2, w3]*invMatrix
	accelerationMatrix.col(0) = JacobianGridVec[i]*Eigen::Vector3d(vertexVec[i].vx, vertexVec[i].vy, vertexVec[i].vz);
	accelerationMatrix.col(1) = JacobianGridVec[j]*Eigen::Vector3d(vertexVec[j].vx, vertexVec[j].vy, vertexVec[j].vz);
	accelerationMatrix.col(2) = JacobianGridVec[k]*Eigen::Vector3d(vertexVec[k].vx, vertexVec[k].vy, vertexVec[k].vz);

	Eigen::Matrix3d W = accelerationMatrix*invMatrix;

	Eigen::FullPivLU<Matrix3d> vCheck(V), wCheck(W);

	Eigen::Matrix3d compositeMatrix;
	if(vCheck.isInvertible())
	{
		compositeMatrix = V.inverse()*W;
	}
	else if(wCheck.isInvertible())
	{
		compositeMatrix = W.inverse()*V;
	}
	else
		return;

	// get eigen-vector to compute [s,t,1]^T as barycentric coordinate solution
	const std::vector<std::pair<double,double> >& result = getParallelEigenVector(compositeMatrix);

	// compute the global coordinates from local coordinates
	Eigen::Matrix3d coordinates;
	coordinates(0,0)=vertexVec[i].x, coordinates(1,0)=vertexVec[i].y, coordinates(2,0)=vertexVec[i].z;
	coordinates(0,1)=vertexVec[j].x, coordinates(1,1)=vertexVec[j].y, coordinates(2,1)=vertexVec[j].z;
	coordinates(0,2)=vertexVec[k].x, coordinates(1,2)=vertexVec[k].y, coordinates(2,2)=vertexVec[k].z;

	// calculate global coordinates
	std::pair<double,double> candidate;
	Eigen::Vector3d barycentric;
	for(int count=0; count<result.size(); ++count)
	{
		candidate = result.at(count);
		barycentric << candidate.first, candidate.second, 1.0-candidate.first-candidate.second;
		barycentric = coordinates*barycentric;
		solutionPoint.push_back(barycentric);
	}
}

// get the velocity of the point w.r.t. pixel information
Eigen::Vector3d ParallelVector::getVelocityInVoxel(const Eigen::Vector3d& pos)
{
	return Eigen::Vector3d();
}

// get Jacobian information
Eigen::Matrix3d ParallelVector::getJacobian(const Eigen::Vector3d& pos)
{
	return Eigen::Matrix3d();
}

// get PV solution point within the triangle w.r.t. max eigen-vector method
// as indicated in https://www.cg.tuwien.ac.at/courses/Visualisierung/Various/FlowSTAR1.pdf
void ParallelVector::getMaxEigenvectorSolution(const std::tuple<int,int,int>& triangleIndex,
		std::vector<Eigen::Vector3d>& solutionPoint)
{
	// get the index of vertices
	const int& i = std::get<0>(triangleIndex);
	const int& j = std::get<1>(triangleIndex);
	const int& k = std::get<2>(triangleIndex);

	Eigen::Matrix3d invMatrix;
	invMatrix(0,0)=1, invMatrix(0,1)=0, invMatrix(0,2)=0;
	invMatrix(1,0)=0, invMatrix(1,1)=1, invMatrix(1,2)=0;
	invMatrix(2,0)=-1, invMatrix(2,1)=-1, invMatrix(2,2)=1;

	// calculate V = [v1, v2, v3]*invMatrix
	Eigen::Matrix3d velocityMatrix;
	velocityMatrix(0,0)=vertexVec[i].vx, velocityMatrix(1,0)=vertexVec[i].vy, velocityMatrix(2,0)=vertexVec[i].vz;
	velocityMatrix(0,1)=vertexVec[j].vx, velocityMatrix(1,1)=vertexVec[j].vy, velocityMatrix(2,1)=vertexVec[j].vz;
	velocityMatrix(0,2)=vertexVec[k].vx, velocityMatrix(1,2)=vertexVec[k].vy, velocityMatrix(2,2)=vertexVec[k].vz;

	Eigen::Matrix3d V = velocityMatrix*invMatrix;

	// regard the maximal eigenvectors at each vertex as linear vector field
	Eigen::Matrix3d eigenvectorMatrix;
	EigenSolver<Matrix3d> es;

	int index[3] = {i, j, k};
	for(int c=0; c<3; ++c)
	{
		// largest eigenvectors are usually vec[2]
		es.compute(JacobianGridVec[index[c]]);
		if(es.eigenvalues()[2].imag()>1.0E-7)
			return;
		eigenvectorMatrix.col(c) = es.eigenvalues().real().col(2);
	}

	Eigen::Matrix3d W = eigenvectorMatrix*invMatrix;

	// check whether is invertible or not
	Eigen::FullPivLU<Matrix3d> vCheck(V), wCheck(W);

	// solved eigenvalue decomposition
	Eigen::Matrix3d compositeMatrix;
	if(vCheck.isInvertible())
	{
		compositeMatrix = V.inverse()*W;
	}
	else if(wCheck.isInvertible())
	{
		compositeMatrix = W.inverse()*V;
	}
	else
		return;

	// get eigen-vector to compute [s,t,1]^T as barycentric coordinate solution
	const std::vector<std::pair<double,double> >& result = getParallelEigenVector(compositeMatrix);

	// compute the global coordinates from local coordinates
	Eigen::Matrix3d coordinates;
	coordinates(0,0)=vertexVec[i].x, coordinates(1,0)=vertexVec[i].y, coordinates(2,0)=vertexVec[i].z;
	coordinates(0,1)=vertexVec[j].x, coordinates(1,1)=vertexVec[j].y, coordinates(2,1)=vertexVec[j].z;
	coordinates(0,2)=vertexVec[k].x, coordinates(1,2)=vertexVec[k].y, coordinates(2,2)=vertexVec[k].z;

	// calculate global coordinates
	std::pair<double,double> candidate;
	Eigen::Vector3d barycentric;
	for(int count=0; count<result.size(); ++count)
	{
		candidate = result.at(count);
		barycentric << candidate.first, candidate.second, 1.0-candidate.first-candidate.second;
		barycentric = coordinates*barycentric;
		solutionPoint.push_back(barycentric);
	}
}


// use eigen-vector to get the [s,t,1] solution
std::vector<std::pair<double,double> > ParallelVector::getParallelEigenVector(const Eigen::Matrix3d& matrix)
{
	EigenSolver<Matrix3d> eigensolver(matrix);
	if (eigensolver.info() != Success)
		abort();

	std::vector<std::pair<double,double> > result;

	double lamda, s, t;
	Eigen::Vector3d temp;
	for(int i=0; i<3; ++i)
	{
		// imag() is not zero, then pass it
		if(abs(eigensolver.eigenvalues()[i].imag())>1.0E-7)
			continue;
		temp = eigensolver.eigenvectors().col(i).real();
		lamda = temp(2);
		s = temp(0)/lamda;
		t = temp(1)/lamda;

		if(0<=s && s<=1 && 0<=t && t<=1 && 0<=s+t && s+t<=1)
		{
			result.push_back(std::make_pair(s,t));
		}
	}
	return result;
}


// get solution point for 2D grid points
void ParallelVector::getSolutionFor2D(const int& x_index, const int& y_index,
		std::vector<Eigen::Vector3d>& solutionPoint)
{
	int x_y = y_index*X_RESOLUTION+x_index, x1_y = x_y+1, x_y1 = x_y+X_RESOLUTION, x1_y1 = x_y1+1;
	if(pvMethodOption==1)
	{
		getEigenDecompositionSolution(std::make_tuple(x_y, x1_y, x1_y1), solutionPoint);
		getEigenDecompositionSolution(std::make_tuple(x1_y1, x_y1, x_y), solutionPoint);
	}
	else if(pvMethodOption==2)
	{
		getMaxEigenvectorSolution(std::make_tuple(x_y, x1_y, x1_y1), solutionPoint);
		getMaxEigenvectorSolution(std::make_tuple(x1_y1, x_y1, x_y), solutionPoint);
	}
}


// get solution point for 3D grid points
void ParallelVector::getSolutionFor3D(const int& x_index, const int& y_index, const int& z_index,
		std::vector<Eigen::Vector3d>& solutionPoint)
{
    //  i,j+1,k+1->  ________   <- i+1,j+1,k+1
	//   			/|      /|
	// i,j+1,k->   / |____ /_|  <- i+1,j+1,k
	// i,j,k+1->  /__/____/ /  <- i+1,j,k+1
	//  		  | /    | /
	//            |/_____|/
	//     (i,j,k)   (i+1,j,k)

	// Quad 1: (i,j,k) (i+1,j,k) (i+1,j,k+1) (i,j,k+1)
	// Face 1: (i,j,k) (i+1,j,k) (i+1,j,k+1)
	// Face 2: (i+1,j,k+1) (i,j,k+1) (i,j,k)

	// Quad 2: (i,j,k) (i,j+1,k) (i,j+1,k+1) (i,j,k+1)
	// Face 3: (i,j,k) (i,j+1,k) (i,j+1,k+1)
	// Face 4: (i,j+1,k+1) (i,j,k+1) (i,j,k)

	// Quad 3: (i,j,k) (i+1,j,k) (i+1,j+1,k) (i,j+1,k)
	// Face 5:  (i,j,k) (i+1,j,k) (i+1,j+1,k)
	// Face 6:  (i+1,j+1,k) (i,j+1,k) (i,j,k)

	// Quad 4: (i+1,j,k) (i+1,j+1,k) (i+1,j+1,k+1) (i+1,j,k+1)
	// Face 7: (i+1,j,k) (i+1,j+1,k) (i+1,j+1,k+1)
	// Face 8: (i+1,j+1,k+1) (i+1,j,k+1) (i+1,j,k)

	// Quad 5: (i,j,k+1) (i+1,j,k+1) (i+1,j+1,k+1) (i,j+1,k+1)
	// Face 9: (i,j,k+1) (i+1,j,k+1) (i+1,j+1,k+1)
	// Face 10: (i+1,j+1,k+1) (i,j+1,k+1) (i,j,k+1)

	// Quad 6: (i,j+1,k) (i+1,j+1,k) (i+1,j+1,k+1) (i,j+1,k+1)
	// Face 11:(i,j+1,k) (i+1,j+1,k) (i+1,j+1,k+1)
	// Face 12:(i+1,j+1,k+1) (i,j+1,k+1) (i,j+1,k)

	const int& XY_MULTIPLY = X_RESOLUTION*Y_RESOLUTION;
	int x_y_z = z_index*XY_MULTIPLY+y_index*X_RESOLUTION+x_index, x1_y_z = x_y_z+1, x1_y1_z = x1_y_z+Y_RESOLUTION,
		x_y1_z = x_y_z+Y_RESOLUTION, x_y_z1 = x_y_z+XY_MULTIPLY, x1_y_z1=x_y_z1+1, x1_y1_z1 = x1_y_z1+Y_RESOLUTION,
		x_y1_z1 = x_y1_z+XY_MULTIPLY;

	// use eigen-decomposition to get PV solution points, not restricted to vortex cores only
	if(pvMethodOption==1)
	{
		getEigenDecompositionSolution(std::make_tuple(x_y_z,x1_y_z,x1_y_z1), solutionPoint);
		getEigenDecompositionSolution(std::make_tuple(x1_y_z1,x_y_z1,x_y_z), solutionPoint);

		getEigenDecompositionSolution(std::make_tuple(x_y_z,x_y1_z,x_y1_z1), solutionPoint);
		getEigenDecompositionSolution(std::make_tuple(x_y1_z1,x_y_z1,x_y_z), solutionPoint);

		getEigenDecompositionSolution(std::make_tuple(x_y_z,x1_y_z,x1_y1_z), solutionPoint);
		getEigenDecompositionSolution(std::make_tuple(x1_y1_z,x_y1_z,x_y_z), solutionPoint);

		getEigenDecompositionSolution(std::make_tuple(x1_y_z,x1_y1_z,x1_y1_z1), solutionPoint);
		getEigenDecompositionSolution(std::make_tuple(x1_y1_z1,x1_y_z1,x1_y_z), solutionPoint);

		getEigenDecompositionSolution(std::make_tuple(x_y_z1,x1_y_z1,x1_y1_z1), solutionPoint);
		getEigenDecompositionSolution(std::make_tuple(x1_y1_z1,x_y1_z1,x_y_z1), solutionPoint);

		getEigenDecompositionSolution(std::make_tuple(x_y1_z,x1_y1_z,x1_y1_z1), solutionPoint);
		getEigenDecompositionSolution(std::make_tuple(x1_y1_z1,x_y1_z1,x_y1_z), solutionPoint);
	}
	// use largest eigenvectors to find the PV solution points
	else if(pvMethodOption==2)
	{
		getMaxEigenvectorSolution(std::make_tuple(x_y_z,x1_y_z,x1_y_z1), solutionPoint);
		getMaxEigenvectorSolution(std::make_tuple(x1_y_z1,x_y_z1,x_y_z), solutionPoint);

		getMaxEigenvectorSolution(std::make_tuple(x_y_z,x_y1_z,x_y1_z1), solutionPoint);
		getMaxEigenvectorSolution(std::make_tuple(x_y1_z1,x_y_z1,x_y_z), solutionPoint);

		getMaxEigenvectorSolution(std::make_tuple(x_y_z,x1_y_z,x1_y1_z), solutionPoint);
		getMaxEigenvectorSolution(std::make_tuple(x1_y1_z,x_y1_z,x_y_z), solutionPoint);

		getMaxEigenvectorSolution(std::make_tuple(x1_y_z,x1_y1_z,x1_y1_z1), solutionPoint);
		getMaxEigenvectorSolution(std::make_tuple(x1_y1_z1,x1_y_z1,x1_y_z), solutionPoint);

		getMaxEigenvectorSolution(std::make_tuple(x_y_z1,x1_y_z1,x1_y1_z1), solutionPoint);
		getMaxEigenvectorSolution(std::make_tuple(x1_y1_z1,x_y1_z1,x_y_z1), solutionPoint);

		getMaxEigenvectorSolution(std::make_tuple(x_y1_z,x1_y1_z,x1_y1_z1), solutionPoint);
		getMaxEigenvectorSolution(std::make_tuple(x1_y1_z1,x_y1_z1,x_y1_z), solutionPoint);
	}
}


// check whether points are within range or not
bool ParallelVector::stayInDomain(const Eigen::Vector3d& point)
{
	for(int i=0; i<3; ++i)
	{
		if(point(i)<limits[i].inf || point(i)>limits[i].sup)
			return false;
	}
	return true;
}

