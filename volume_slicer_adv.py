import numpy as np

from traits.api import HasTraits, Instance, Array, \
    Bool, Dict, on_trait_change
from traitsui.api import View, Item, HGroup, Group

from tvtk.api import tvtk
from tvtk.pyface.scene import Scene

from mayavi import mlab
from mayavi.core.api import PipelineBase, Source
from mayavi.core.ui.api import SceneEditor, MlabSceneModel

import dicom
import os


################################################################################
# The object implementing the dialog
class VolumeSlicer(HasTraits):
    # The data to plot
    data = Array

    # The position of the view
    position = Array(shape=(3,))

    # The 4 views displayed
    scene3d = Instance(MlabSceneModel, ())
    scene_x = Instance(MlabSceneModel, ())
    scene_y = Instance(MlabSceneModel, ())
    scene_z = Instance(MlabSceneModel, ())

    # The data source
    data_src = Instance(Source)

    # The image plane widgets of the 3D scene
    ipw_3d_x = Instance(PipelineBase)
    ipw_3d_y = Instance(PipelineBase)
    ipw_3d_z = Instance(PipelineBase)

    # The cursors on each view:
    cursors = Dict()

    disable_render = Bool

    _axis_names = dict(x=0, y=1, z=2)

    #---------------------------------------------------------------------------
    # Object interface
    #---------------------------------------------------------------------------
    def __init__(self, **traits):
        super(VolumeSlicer, self).__init__(**traits)
        # Force the creation of the image_plane_widgets:
        self.ipw_3d_x
        self.ipw_3d_y
        self.ipw_3d_z


    #---------------------------------------------------------------------------
    # Default values
    #---------------------------------------------------------------------------
    def _position_default(self):
        return 0.5*np.array(self.data.shape)

    def _data_src_default(self):
        return mlab.pipeline.scalar_field(self.data,
                            figure=self.scene3d.mayavi_scene,
                            name='Data',)

    def make_ipw_3d(self, axis_name):
        ipw = mlab.pipeline.image_plane_widget(self.data_src,
                        figure=self.scene3d.mayavi_scene,
                        plane_orientation='%s_axes' % axis_name,
                        name='Cut %s' % axis_name)
        return ipw

    def _ipw_3d_x_default(self):
        return self.make_ipw_3d('x')

    def _ipw_3d_y_default(self):
        return self.make_ipw_3d('y')

    def _ipw_3d_z_default(self):
        return self.make_ipw_3d('z')


    #---------------------------------------------------------------------------
    # Scene activation callbacks
    #---------------------------------------------------------------------------
    @on_trait_change('scene3d.activated')
    def display_scene3d(self):
        outline = mlab.pipeline.outline(self.data_src,
                        figure=self.scene3d.mayavi_scene,
                        )
        self.scene3d.mlab.view(40, 50)
        # Interaction properties can only be changed after the scene
        # has been created, and thus the interactor exists
        for ipw in (self.ipw_3d_x, self.ipw_3d_y, self.ipw_3d_z):
            ipw.ipw.interaction = 0
        self.scene3d.scene.background = (0, 0, 0)
        # Keep the view always pointing up
        self.scene3d.scene.interactor.interactor_style = \
                                 tvtk.InteractorStyleTerrain()
        self.update_position()


    def make_side_view(self, axis_name):
        scene = getattr(self, 'scene_%s' % axis_name)
        scene.scene.parallel_projection = True
        ipw_3d   = getattr(self, 'ipw_3d_%s' % axis_name)

        # We create the image_plane_widgets in the side view using a
        # VTK dataset pointing to the data on the corresponding
        # image_plane_widget in the 3D view (it is returned by
        # ipw_3d._get_reslice_output())
        side_src = ipw_3d.ipw._get_reslice_output()
        ipw = mlab.pipeline.image_plane_widget(
                            side_src,
                            plane_orientation='z_axes',
                            vmin=self.data.min(),
                            vmax=self.data.max(),
                            figure=scene.mayavi_scene,
                            name='Cut view %s' % axis_name,
                            )
        setattr(self, 'ipw_%s' % axis_name, ipw)

        # Extract the spacing of the side_src to convert coordinates
        # into indices
        spacing = side_src.spacing

        # Make left-clicking create a crosshair
        ipw.ipw.left_button_action = 0

        x, y, z = self.position
        cursor = mlab.points3d(x, y, z,
                            mode='axes',
                            color=(0, 0, 0),
                            scale_factor=2*max(self.data.shape),
                            figure=scene.mayavi_scene,
                            name='Cursor view %s' % axis_name,
                        )
        self.cursors[axis_name] = cursor

        # Add a callback on the image plane widget interaction to
        # move the others
        this_axis_number = self._axis_names[axis_name]
        def move_view(obj, evt):
            # Disable rendering on all scene
            position = list(obj.GetCurrentCursorPosition()*spacing)[:2]
            position.insert(this_axis_number, self.position[this_axis_number])
            # We need to special case y, as the view has been rotated.
            if axis_name is 'y':
                position = position[::-1]
            self.position = position

        ipw.ipw.add_observer('InteractionEvent', move_view)
        ipw.ipw.add_observer('StartInteractionEvent', move_view)

        # Center the image plane widget
        ipw.ipw.slice_position = 0.5*self.data.shape[
                                        self._axis_names[axis_name]]

        # 2D interaction: only pan and zoom
        scene.scene.interactor.interactor_style = \
                                 tvtk.InteractorStyleImage()
        scene.scene.background = (0, 0, 0)

        # Some text:
        mlab.text(0.01, 0.8, axis_name, width=0.08)

        # Choose a view that makes sens
        views = dict(x=(0, 0), y=(90, 180), z=(0, 0))
        mlab.view(views[axis_name][0],
                  views[axis_name][1],
                  focalpoint=0.5*np.array(self.data.shape),
                  figure=scene.mayavi_scene)
        scene.scene.camera.parallel_scale = 0.52*np.mean(self.data.shape)

    @on_trait_change('scene_x.activated')
    def display_scene_x(self):
        return self.make_side_view('x')

    @on_trait_change('scene_y.activated')
    def display_scene_y(self):
        return self.make_side_view('y')

    @on_trait_change('scene_z.activated')
    def display_scene_z(self):
        return self.make_side_view('z')


    #---------------------------------------------------------------------------
    # Traits callback
    #---------------------------------------------------------------------------
    @on_trait_change('position')
    def update_position(self):
        """ Update the position of the cursors on each side view, as well
            as the image_plane_widgets in the 3D view.
        """
        # First disable rendering in all scenes to avoid unecessary
        # renderings
        self.disable_render = True

        # For each axis, move image_plane_widget and the cursor in the
        # side view
        for axis_name, axis_number in self._axis_names.iteritems():
            ipw3d = getattr(self, 'ipw_3d_%s' % axis_name)
            ipw3d.ipw.slice_position = self.position[axis_number]

            # Go from the 3D position, to the 2D coordinates in the
            # side view
            position2d = list(self.position)
            position2d.pop(axis_number)
            if axis_name is 'y':
                position2d = position2d[::-1]
            # Move the cursor
            # For the following to work, you need Mayavi 3.4.0, if you
            # have a less recent version, use 'x=[position2d[0]]'
            self.cursors[axis_name].mlab_source.set(
                                                x=position2d[0],
                                                y=position2d[1],
                                                z=0)

        self.disable_render = False

    @on_trait_change('disable_render')
    def _render_enable(self):
        for scene in (self.scene3d, self.scene_x, self.scene_y,
                                                  self.scene_z):
            scene.scene.disable_render = self.disable_render


    #---------------------------------------------------------------------------
    # The layout of the dialog created
    #---------------------------------------------------------------------------
    view = View(HGroup(
                  Group(
                       Item('scene_y',
                            editor=SceneEditor(scene_class=Scene),
                            height=250, width=300),
                       Item('scene_z',
                            editor=SceneEditor(scene_class=Scene),
                            height=250, width=300),
                       show_labels=False,
                  ),
                  Group(
                       Item('scene_x',
                            editor=SceneEditor(scene_class=Scene),
                            height=250, width=300),
                       Item('scene3d',
                            editor=SceneEditor(scene_class=Scene),
                            height=250, width=300),
                       show_labels=False,
                  ),
                ),
                resizable=True,
                title='Volume Slicer',
                )


################################################################################
if __name__ == '__main__':
    #dcm_path = "/home/sansomk/caseFiles/mri/images/E431791260_Merge/0.4/102"
    #dcm_path = "/home/sansomk/caseFiles/mri/images/E431791260_FlowVol_01/mag"
    #dcm_path = "/home/ksansom/caseFiles/mri/images/0.4/102/"
    dcm_path = "/Users/sansomk/caseFiles/mri/E431791260_merge/0.4/102"
    dcm_files = []
    slice_location = []
    acq_N = []
    trig_count = 0
    for dirname, subdirlist, filelist in os.walk(dcm_path):
      for filen in filelist:
        filePath = os.path.join(dirname, filen)
        try:          
          #print(filePath)
          f = dicom.read_file(filePath, stop_before_pixels=True)
        except Exception as e: 
          print(str(e))
          print("error: {0}".format(filen))
        
        if (hasattr(f, "TriggerTime")):
          if (f.TriggerTime < 1.0 and filePath not in dcm_files):
            dcm_files.append(filePath)
            slice_location.append(f.SliceLocation)
            acq_N.append(f.AcquisitionNumber)
        else:
          trig_count += 1
          dcm_files.append(filePath)
          slice_location.append(f.SliceLocation)
          acq_N.append(f.AcquisitionNumber)
    if (trig_count > 0):
        print("No Trigger Time found in {0} images out of {1} images".format(trig_count, len(dcm_files)))
        #else:
        #  dcm_files.append(filePath)
    path_loc = zip(dcm_files, slice_location)
    path_loc.sort(key=lambda x: x[1])
    dcm_files, slice_location = zip(*path_loc)
    #print(slice_location, acq_N)
    # get reference image
    #print(len(dcm_files))
    ref_image = dicom.read_file(dcm_files[0])
    # load dimensions based on the number of rows columns and slices
    const_pixel_dims = (int(ref_image.Rows), int(ref_image.Columns), len(dcm_files))
    
    #check it whether image has been interpolated
    if (hasattr(ref_image, 'SpacingBetweenSlices')):
        if(ref_image.SpacingBetweenSlices < ref_image.SliceThickness):
            z_spacing = float(ref_image.SpacingBetweenSlices)
        else:
            z_spacing = float(ref_image.SliceThickness)
    else:
        z_spacing = float(ref_image.SliceThickness)

    # load spacing values in mm
    const_pixel_spacing = (float(ref_image.PixelSpacing[0]), 
                           float(ref_image.PixelSpacing[1]), z_spacing)
    #x = np.arange(0.0, (const_pixel_dims[0]+1)*const_pixel_spacing[0],
    #              const_pixel_spacing[0])
    #y = np.arange(0.0, (const_pixel_dims[1]+1)*const_pixel_spacing[1],
    #              const_pixel_spacing[1])
    #z = np.arange(0.0, (const_pixel_dims[2]+1)*const_pixel_spacing[2],
    #              const_pixel_spacing[2])
    # the array is sized based on 'const_pixel_dims
    
    array_dicom = np.zeros(const_pixel_dims, dtype=ref_image.pixel_array.dtype)
    print(array_dicom.shape)
    #loop through all the DICOM FILES
    for filenamedcm in dcm_files:
      #read the file
      ds = dicom.read_file(filenamedcm)
      #store the raw image data
      array_dicom[:, :, dcm_files.index(filenamedcm)] = ds.pixel_array
    print(array_dicom.shape)
    testindx = np.where(array_dicom !=0)
    minx = np.min(testindx[0])
    miny = np.min(testindx[1])
    minz = np.min(testindx[2])
    maxx = np.max(testindx[0])
    maxy = np.max(testindx[1])
    maxz = np.max(testindx[2])
    array_dicom.dump("merge")
    
    # Create some data
    #x, y, z = np.ogrid[-5:5:100j, -5:5:100j, -5:5:100j]
    #data = np.sin(3*x)/x + 0.05*z**2 + np.cos(3*y)
    print(array_dicom[minx:maxx, miny:maxy,:].shape)
    m = VolumeSlicer(data=array_dicom[minx:maxx, miny:maxy, :])
    m.configure_traits()
    array_dicom[minx:maxx, miny:maxy,:].dump("mergefiltered")