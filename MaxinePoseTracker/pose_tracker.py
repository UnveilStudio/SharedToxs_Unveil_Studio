import os

import carb
import numpy as np
import omni.maxine.sdk
from omni import ui
from pythonosc import udp_client

from .._omni_maxine_pose_tracker import acquire_interface, release_interface
from .widgets import PoseTrackerEngineWidget

NVAR_SDK_MODELS_ROOT = omni.maxine.sdk.get_models_root()


class PoseTracker:
    def __init__(self, ext_handle):
        self._tracker = acquire_interface()
        self._ext_handle = ext_handle

        self.draw_3D_points = ui.SimpleBoolModel(True)
        self.draw_skeleton_overlay = ui.SimpleBoolModel(True)
        self._widget = PoseTrackerEngineWidget(self)

        self._engine_started = False
        self._model_path = os.path.join(NVAR_SDK_MODELS_ROOT, "models")
        self._input_image_width = 0
        self._input_image_height = 0
        self._temporal_stability = True
        self._num_key_points = None
        self._ref_pose = None

        self._points_2d = None
        self._points_3d = None
        self._joint_rotations = None
        self._bounding_box = None
        self._confidence = None
        self.jump = False
        self._init_pose_tracker()

    def shutdown(self):
        self._ext_handle = None
        self._widget.shutdown()
        self._widget = None
        self._destroy_pose_tracker()
        if self._tracker:
            release_interface(self._tracker)
            self._tracker = None

    def start_pose_tracking(self):
        self.jump = True
        if self._widget and self._widget.start_button:
            self._widget.start_button.checked = True
        if not self._engine_started:
            self._engine_started = True
        self._ext_handle.update()

    def stop_pose_tracking(self):
        if self._widget and self._widget.start_button:
            self._widget.start_button.checked = False
        if self._engine_started:
            self._engine_started = False
        self._ext_handle.update()

    def run(self, frame):
        if not self.engine_started():
            carb.log_warn("PoseTracker: Engine has not been started")
            return False

        if frame is None:
            carb.log_warn("PoseTracker: Input frame is None")
            return False

        if not self._tracker.is_initialized():

            carb.log_warn("PoseTracker: Tracker is not initialized. Initializing...")
            self._input_image_height, self._input_image_width = frame.shape[0], frame.shape[1]
            self._init_pose_tracker()

        elif self._input_image_height != frame.shape[0] or self._input_image_width != frame.shape[1]:

            carb.log_info(
                "PoseTracker: Input frame shape has been changed ({} x {}). Resetting the tracker...".format(
                    frame.shape[1], frame.shape[0]
                )
            )
            self._input_image_height, self._input_image_width = frame.shape[0], frame.shape[1]
            self._reset_pose_tracker()

        self._tracker.run(frame)
        #it get the np arry from the tracker i guess the dll is called somehow
        self._points_2d = np.array(self._tracker.get_points_2d()).reshape((-1, 2))
        self._points_3d = np.array(self._tracker.get_points_3d()).reshape((-1, 3))
        self._joint_rotations = np.array(self._tracker.get_joint_rotations()).reshape((-1, 4))
        self._bounding_box = np.array(self._tracker.get_bounding_box()).reshape((-1, 4))
        self._confidence = np.array(self._tracker.get_keypoint_confidence()).reshape((-1, 1))
        
        #the in order to send value with udp i need to traslate pixels in something nice for the next stage
        
        list2d = self._points_2d.tolist()
        list3d = self._points_3d.tolist()
        listRot = self._joint_rotations.tolist()
        listBox = self._bounding_box.tolist()
        listConf = self._confidence.tolist() 

        

        #create server to send messages in UDP
        client = udp_client.SimpleUDPClient("192.168.178.98",  3335)
        #the client send
        client.send_message("/2dPos",list2d)
        client.send_message("/3dPos",list3d)
        client.send_message("/Rot",listRot)
        client.send_message("/Box",listBox)
        client.send_message("/Conf",listConf)

        return True

    def engine_started(self):
        return self._engine_started

    def get_ref_pose(self):
        return self._ref_pose

    def _init_pose_tracker(self):
        carb.log_info("PoseTracker: Frame shape: {} x {}".format(self._input_image_width, self._input_image_height))
        self._tracker.init(
            self._input_image_width, self._input_image_height, self._model_path, self._temporal_stability, 0.0
        )
        self._num_key_points = self._tracker.get_num_points()
        self._ref_pose = np.array(self._tracker.get_reference_pose()).reshape((-1, 3))

    def _destroy_pose_tracker(self):
        if self._tracker.is_initialized():
            self._tracker.destroy()

    def _reset_pose_tracker(self):
        self._destroy_pose_tracker()
        self._init_pose_tracker()

    def should_draw_skeleton_overlay(self):
        return self.draw_skeleton_overlay.as_bool

    def should_draw_3D_points(self):
        return self.draw_3D_points.as_bool

    def get_confidence_threshold(self):
        return self._widget.confidence_threshold
