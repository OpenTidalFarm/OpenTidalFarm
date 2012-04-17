import os
import re

try: paraview.simple
except: from paraview.simple import *
paraview.simple._DisableFirstRenderCameraReset()

RenderView1 = CreateRenderView()
RenderView1.LightSpecularColor = [1.0, 1.0, 1.0]
RenderView1.InteractionMode = '3D'
RenderView1.UseTexturedBackground = 0
RenderView1.UseLight = 1
RenderView1.CameraPosition = [100.0, 33.0, 200]
RenderView1.FillLightKFRatio = 3.0
RenderView1.Background2 = [0.0, 0.0, 0.16470588235294117]
RenderView1.FillLightAzimuth = -10.0
RenderView1.LODResolution = 50.0
RenderView1.BackgroundTexture = []
RenderView1.HeadPose = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
RenderView1.KeyLightAzimuth = 10.0
RenderView1.StencilCapable = 1
RenderView1.LightIntensity = 1.0
RenderView1.CameraFocalPoint = [100.0, 33.0, -6.812856554559699]
RenderView1.ImageReductionFactor = 2
RenderView1.CameraViewAngle = 30.0
RenderView1.CameraParallelScale = 10.30432089900205
RenderView1.EyeAngle = 2.0
RenderView1.HeadLightKHRatio = 3.0
RenderView1.StereoRender = 0
RenderView1.KeyLightIntensity = 0.75
RenderView1.BackLightAzimuth = 110.0
RenderView1.OrientationAxesInteractivity = 0
RenderView1.UseInteractiveRenderingForSceenshots = 0
RenderView1.UseOffscreenRendering = 0
RenderView1.Background = [1.0, 1.0, 1.0]
RenderView1.UseOffscreenRenderingForScreenshots = 1
RenderView1.NonInteractiveRenderDelay = 2
RenderView1.CenterOfRotation = [100.0, 33.0, 0.0]
RenderView1.CameraParallelProjection = 0
RenderView1.CompressorConfig = 'vtkSquirtCompressor 0 3'
RenderView1.HeadLightWarmth = 0.5
RenderView1.MaximumNumberOfPeels = 4
RenderView1.LightDiffuseColor = [1.0, 1.0, 1.0]
RenderView1.StereoType = 'Red-Blue'
RenderView1.DepthPeeling = 1
RenderView1.BackLightKBRatio = 3.5
RenderView1.StereoCapableWindow = 1
RenderView1.CameraViewUp = [0.0, 1.0, 0.0]
RenderView1.LightType = 'HeadLight'
RenderView1.LightAmbientColor = [1.0, 1.0, 1.0]
RenderView1.WandPose = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
RenderView1.RemoteRenderThreshold = 3.0
RenderView1.KeyLightElevation = 50.0
RenderView1.CenterAxesVisibility = 0
RenderView1.MaintainLuminance = 0
RenderView1.StillRenderImageReductionFactor = 1
RenderView1.BackLightWarmth = 0.5
RenderView1.FillLightElevation = -75.0
RenderView1.MultiSamples = 0
RenderView1.FillLightWarmth = 0.4
RenderView1.AlphaBitPlanes = 1
RenderView1.LightSwitch = 0
RenderView1.OrientationAxesVisibility = 0
RenderView1.CameraClippingRange = [181.16258161187127, 185.73739427883774]
RenderView1.BackLightElevation = 0.0
RenderView1.ViewTime = 0.0
RenderView1.OrientationAxesOutlineColor = [1.0, 1.0, 1.0]
RenderView1.LODThreshold = 5.0
RenderView1.CollectGeometryThreshold = 100.0
RenderView1.UseGradientBackground = 0
RenderView1.KeyLightWarmth = 0.6
RenderView1.OrientationAxesLabelColor = [1.0, 1.0, 1.0]
RenderView1.ViewSize = [2000, 1000] #[width, height]

def sort_nicely(l): 
   convert = lambda text: int(text) if text.isdigit() else text 
   alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
   l.sort( key=alphanum_key ) 

files = os.listdir('.')
pvtus = [pvtu for pvtu in files if re.match("turbines_t=.*\.p?vtu", pvtu)]
sort_nicely(pvtus)
pvtus = [pvtus[0], pvtus[-1]]
print pvtus

turbines_t_ = XMLPartitionedUnstructuredGridReader( guiName="turbines_t=.*", PointArrayStatus=['u'], CellArrayStatus=['connectivity', 'offsets', 'types'], FileName=pvtus)

a1_u_PiecewiseFunction = CreatePiecewiseFunction( Points=[0.0, 0.0, 1.0, 1.0] )

a1_u_PVLookupTable = GetLookupTableForArray( "u", 1, Discretize=1, RGBPoints=[0.0, 0.23, 0.299, 0.754, 0.9899498224258423, 0.706, 0.016, 0.15], UseLogScale=0, VectorComponent=0, NanColor=[0.25, 0.0, 0.0], NumberOfTableValues=256, ColorSpace='Diverging', VectorMode='Magnitude', HSVWrap=0, ScalarRangeInitialized=1.0, LockScalarRange=0 )

DataRepresentation1 = Show()
DataRepresentation1.CubeAxesZAxisVisibility = 1
DataRepresentation1.SelectionPointLabelColor = [0.5, 0.5, 0.5]
DataRepresentation1.SelectionPointFieldDataArrayName = 'vtkOriginalPointIds'
DataRepresentation1.SuppressLOD = 0
DataRepresentation1.CubeAxesXGridLines = 0
DataRepresentation1.CubeAxesYAxisTickVisibility = 1
DataRepresentation1.Position = [0.0, 0.0, 0.0]
DataRepresentation1.BackfaceRepresentation = 'Follow Frontface'
DataRepresentation1.SelectionOpacity = 1.0
DataRepresentation1.SelectionPointLabelShadow = 0
DataRepresentation1.CubeAxesYGridLines = 0
DataRepresentation1.OrientationMode = 'Direction'
DataRepresentation1.Source.TipResolution = 6
DataRepresentation1.ScaleMode = 'No Data Scaling Off'
DataRepresentation1.Diffuse = 1.0
DataRepresentation1.SelectionUseOutline = 0
DataRepresentation1.SelectionPointLabelFormat = ''
DataRepresentation1.CubeAxesZTitle = 'Z-Axis'
DataRepresentation1.Specular = 0.1
DataRepresentation1.SelectionVisibility = 1
DataRepresentation1.InterpolateScalarsBeforeMapping = 1
DataRepresentation1.CubeAxesZAxisTickVisibility = 1
DataRepresentation1.Origin = [0.0, 0.0, 0.0]
DataRepresentation1.CubeAxesVisibility = 0
DataRepresentation1.Scale = [1.0, 1.0, 1.0]
DataRepresentation1.SelectionCellLabelJustification = 'Left'
DataRepresentation1.DiffuseColor = [1.0, 1.0, 1.0]
DataRepresentation1.SelectionCellLabelOpacity = 1.0
DataRepresentation1.CubeAxesInertia = 1
DataRepresentation1.Source = "Arrow"
DataRepresentation1.Source.Invert = 0
DataRepresentation1.Masking = 0
DataRepresentation1.Opacity = 1.0
DataRepresentation1.LineWidth = 1.0
DataRepresentation1.MeshVisibility = 0
DataRepresentation1.Visibility = 1
DataRepresentation1.SelectionCellLabelFontSize = 18
DataRepresentation1.CubeAxesCornerOffset = 0.0
DataRepresentation1.SelectionPointLabelJustification = 'Left'
DataRepresentation1.SelectionPointLabelVisibility = 0
DataRepresentation1.SelectOrientationVectors = ''
DataRepresentation1.CubeAxesTickLocation = 'Inside'
DataRepresentation1.BackfaceDiffuseColor = [1.0, 1.0, 1.0]
DataRepresentation1.CubeAxesYAxisVisibility = 1
DataRepresentation1.SelectionPointLabelFontFamily = 'Arial'
DataRepresentation1.Source.ShaftResolution = 6
DataRepresentation1.CubeAxesFlyMode = 'Closest Triad'
DataRepresentation1.SelectScaleArray = ''
DataRepresentation1.CubeAxesYTitle = 'Y-Axis'
DataRepresentation1.ColorAttributeType = 'POINT_DATA'
DataRepresentation1.SpecularPower = 100.0
DataRepresentation1.Texture = []
DataRepresentation1.SelectionCellLabelShadow = 0
DataRepresentation1.AmbientColor = [1.0, 1.0, 1.0]
DataRepresentation1.MapScalars = 1
DataRepresentation1.PointSize = 2.0
DataRepresentation1.Source.TipLength = 0.35
DataRepresentation1.SelectionCellLabelFormat = ''
DataRepresentation1.Scaling = 0
DataRepresentation1.StaticMode = 0
DataRepresentation1.SelectionCellLabelColor = [0.0, 1.0, 0.0]
DataRepresentation1.Source.TipRadius = 0.1
DataRepresentation1.EdgeColor = [0.0, 0.0, 0.5000076295109483]
DataRepresentation1.CubeAxesXAxisTickVisibility = 1
DataRepresentation1.SelectionCellLabelVisibility = 0
DataRepresentation1.NonlinearSubdivisionLevel = 1
DataRepresentation1.CubeAxesColor = [1.0, 1.0, 1.0]
DataRepresentation1.Representation = 'Surface'
DataRepresentation1.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
DataRepresentation1.CubeAxesXAxisMinorTickVisibility = 1
DataRepresentation1.Orientation = [0.0, 0.0, 0.0]
DataRepresentation1.CubeAxesXTitle = 'X-Axis'
DataRepresentation1.ScalarOpacityUnitDistance = 11.227805366538481
DataRepresentation1.BackfaceOpacity = 1.0
DataRepresentation1.SelectionCellFieldDataArrayName = 'vtkOriginalCellIds'
DataRepresentation1.SelectionColor = [1.0, 0.0, 1.0]
DataRepresentation1.Ambient = 0.0
DataRepresentation1.SelectionPointLabelFontSize = 18
DataRepresentation1.ScaleFactor = 1.0
DataRepresentation1.BackfaceAmbientColor = [1.0, 1.0, 1.0]
DataRepresentation1.Source.ShaftRadius = 0.03
DataRepresentation1.ScalarOpacityFunction = a1_u_PiecewiseFunction
DataRepresentation1.SelectMaskArray = ''
DataRepresentation1.SelectionLineWidth = 2.0
DataRepresentation1.CubeAxesZAxisMinorTickVisibility = 1
DataRepresentation1.CubeAxesXAxisVisibility = 1
DataRepresentation1.Interpolation = 'Gouraud'
DataRepresentation1.SelectMapper = 'Projected tetra'
DataRepresentation1.SelectionCellLabelFontFamily = 'Arial'
DataRepresentation1.SelectionCellLabelItalic = 0
DataRepresentation1.CubeAxesYAxisMinorTickVisibility = 1
DataRepresentation1.CubeAxesZGridLines = 0
DataRepresentation1.ExtractedBlockIndex = 0
DataRepresentation1.SelectionPointLabelOpacity = 1.0
DataRepresentation1.Pickable = 1
DataRepresentation1.CustomBoundsActive = [0, 0, 0]
DataRepresentation1.SelectionRepresentation = 'Wireframe'
DataRepresentation1.SelectionPointLabelBold = 0
DataRepresentation1.ColorArrayName = 'u'
DataRepresentation1.SelectionPointLabelItalic = 0
DataRepresentation1.AllowSpecularHighlightingWithScalarColoring = 0
DataRepresentation1.SpecularColor = [1.0, 1.0, 1.0]
DataRepresentation1.LookupTable = a1_u_PVLookupTable
DataRepresentation1.SelectionPointSize = 5.0
DataRepresentation1.SelectionCellLabelBold = 0
DataRepresentation1.Orient = 0

RenderView1.ViewTime = turbines_t_.TimestepValues[0]
print "Setting timelevel to " + str(turbines_t_.TimestepValues[0])
Render()
WriteImage("turbine_visualisation_initial.jpg")

RenderView1.ViewTime = turbines_t_.TimestepValues[-1]
print "Setting timelevel to " + str(turbines_t_.TimestepValues[-1])
Render()
WriteImage("turbine_visualisation_final.jpg")
