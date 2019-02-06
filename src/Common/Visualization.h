/*
 * Visualization.h
 *
 *  Created on: Feb 1, 2019
 *      Author: lieyu
 */

#ifndef SRC_COMMON_VISUALIZATION_H_
#define SRC_COMMON_VISUALIZATION_H_

#include <math.h>

#include <vtkVersion.h>
#include <vtkPolyData.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkDoubleArray.h>
#include <vtkPolyData.h>
#include <vtkPointData.h>
#include <vtkPolyLine.h>
#include <vtkLookupTable.h>
#include <vtkNamedColors.h>


#include <vtkLine.h>
#include <vtkCell.h>
#include <vtkCellData.h>
#include <vtkDataSet.h>
#include <vtkDataSetAttributes.h>
#include <vtkProperty.h>
#include <vtkSmartPointer.h>
#include <vtkTubeFilter.h>

#include <vtkDataSetMapper.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkCamera.h>
#include <vtkInteractorStyleTrackballCamera.h>

#include <vtkMath.h>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SVD>

using namespace std;
using namespace Eigen;

class Visualization {
public:
	Visualization();
	virtual ~Visualization();

	// visualize streamlines in vtk
	void renderStreamlines(const std::vector<Eigen::VectorXd>& streamlineVector,
						   const std::vector<int>& separationVector);
};

#endif /* SRC_COMMON_VISUALIZATION_H_ */
