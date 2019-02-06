/*
 * Visualization.cpp
 *
 *  Created on: Feb 1, 2019
 *      Author: lieyu
 */

#include "Visualization.h"

Visualization::Visualization() {
	// TODO Auto-generated constructor stub

}

Visualization::~Visualization() {
	// TODO Auto-generated destructor stub
}

// visualize streamlines in vtk
void Visualization::renderStreamlines(const std::vector<Eigen::VectorXd>& streamlineVector,
									  const std::vector<int>& separationVector)
{
	// Spiral tube
	double vX, vY, vZ;
	int nV = 0;
	for(int i=0; i<streamlineVector.size(); ++i)
	{
		nV += streamlineVector[i].size()/3;
	}

	int lower = INT_MAX, upper = INT_MIN;
	for(int i=0; i<streamlineVector.size(); ++i)
	{
		lower = std::min(lower, separationVector[i]);
		upper = std::max(upper, separationVector[i]);
	}

	double rT1 = 0.1, rT2 = 0.5;// Start/end tube radii

	unsigned int nTv = 8;       // No. of surface elements for each tube vertex

	// Create points and cells for the spiral
	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
	vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
	vtkSmartPointer<vtkCellArray> lineCells = vtkSmartPointer<vtkCellArray>::New();

	int cumulated = 0;
	Eigen::VectorXd line;
	for(int i=0; i<streamlineVector.size(); ++i)
	{
		line = streamlineVector[i];
		vtkSmartPointer<vtkPolyLine> cells = vtkSmartPointer<vtkPolyLine>::New();
		cells->GetPointIds()->SetNumberOfIds(line.size()/3);
		for(int j=0; j<line.size()/3; ++j)
		{
			vX = line(3*j);
			vY = line(3*j+1);
			vZ = line(3*j+2);
			points->InsertPoint(cumulated+j, vX, vY, vZ);
			cells->GetPointIds()->SetId(j, cumulated+j);
		}
		lineCells->InsertNextCell(cells);
		cumulated+=line.size()/3;
	}

	polyData->SetPoints(points);
	polyData->SetLines(lineCells);

	// Varying tube radius using sine-function
	vtkSmartPointer<vtkDoubleArray> tubeRadius = vtkSmartPointer<vtkDoubleArray>::New();
	tubeRadius->SetName("TubeRadius");
	tubeRadius->SetNumberOfTuples(nV);
	for (int i=0 ;i<nV; ++i)
	{
		tubeRadius->SetTuple1(i, 0.0015);
	}
	polyData->GetPointData()->AddArray(tubeRadius);
	polyData->GetPointData()->SetActiveScalars("TubeRadius");

	// RBG array (could add Alpha channel too I guess...)
	// Varying from blue to red

	int range = upper-lower;
	vtkSmartPointer<vtkUnsignedCharArray> colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
	colors->SetNumberOfTuples(nV);
	colors->SetName("Colors");
	colors->SetNumberOfComponents(3);
	cumulated = 0;

	/*
	int sValue, r, b;
	for(int i=0; i<streamlineVector.size(); ++i)
	{
		line = streamlineVector[i];
		sValue = separationVector[i];
		r = int(255*(double(sValue-lower)/double(range)));
		b = int(255*(double(upper-sValue)/double(range)));
		for(int j=0; j<line.size()/3; ++j)
		{
			colors->InsertTuple3(cumulated+j, r, 0, b);
		}
		cumulated+=line.size()/3;
	}
	polyData->GetPointData()->AddArray(colors);
	*/

	vtkSmartPointer<vtkLookupTable> colorLookupTable = vtkSmartPointer<vtkLookupTable>::New();
	colorLookupTable->SetTableRange(lower, upper);
	colorLookupTable->Build();
	int sValue;
	for(int i=0; i<streamlineVector.size(); ++i)
	{
		line = streamlineVector[i];
		sValue = separationVector[i];
		double dcolor[3];
		colorLookupTable->GetColor(sValue, dcolor);
		unsigned char color[3];
		color[0] = 255*(1.0-dcolor[0]/1.0);
		color[1] = 255*dcolor[1]/1.0;
		color[2] = 255*(1.0-dcolor[2]/1.0);
		for(int j=0; j<line.size()/3; ++j)
		{
			colors->InsertTuple3(cumulated+j, color[0], color[1], color[2]);
		}
		cumulated+=line.size()/3;
	}
	polyData->GetPointData()->AddArray(colors);

	vtkSmartPointer<vtkTubeFilter> tube = vtkSmartPointer<vtkTubeFilter>::New();
#if VTK_MAJOR_VERSION <= 5
	tube->SetInput(polyData);
#else
	tube->SetInputData(polyData);
#endif
	tube->SetNumberOfSides(nTv);
	tube->SetVaryRadiusToVaryRadiusByAbsoluteScalar();

	vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	mapper->SetInputConnection(tube->GetOutputPort());
	mapper->ScalarVisibilityOn();
	mapper->SetScalarModeToUsePointFieldData();
	mapper->SelectColorArray("Colors");

	vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
	actor->SetMapper(mapper);

	vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
	renderer->AddActor(actor);
	vtkSmartPointer<vtkNamedColors> namedColor = vtkSmartPointer<vtkNamedColors>::New();
	renderer->SetBackground(namedColor->GetColor3d("SlateGray").GetData());

	vtkSmartPointer<vtkRenderWindow> renWin = vtkSmartPointer<vtkRenderWindow>::New();
	renWin->SetWindowName("Streamline Separation VTK");
	vtkSmartPointer<vtkRenderWindowInteractor> iren = vtkSmartPointer<vtkRenderWindowInteractor>::New();

	iren->SetRenderWindow(renWin);
	renWin->AddRenderer(renderer);

	renWin->SetSize(500, 500);
	renWin->Render();

	vtkSmartPointer<vtkInteractorStyleTrackballCamera> style = vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();
	iren->SetInteractorStyle(style);

	iren->Start();
}
