#!/usr/bin/env python3
"""
VR Desktop Display with Controller Interaction
- Move overlay with grip (single controller)
- Resize with pinch (both grips)
- Pointer ray with trigger click / thumbstick right-click
"""

import sys
import time
import signal
import os
import mmap
import ctypes
import math

import numpy as np
import openvr

from OpenGL.GL import *
import glfw

from pynput.mouse import Button, Controller as MouseController

SHM_FILE = '/dev/shm/vr_desktop_frame'
DEBUG = True  # Show raycast lines always


class VRDesktopViewer:
    def __init__(self):
        self.running = True
        self.shm = None
        self.shm_fd = None
        self.window = None

        # Overlay state
        self.overlay_handle = None
        self.overlay_width = 2.5  # meters
        self.overlay_pos = [0.0, 0.0, -1.5]  # x, y, z from seated origin
        self.overlay_aspect = 16/9

        # Texture
        self.texture_id = None
        self.texture = None
        self.current_width = 0
        self.current_height = 0

        # Controller state
        self.left_id = None
        self.right_id = None
        self.left_grip = False
        self.right_grip = False
        self.left_trigger = False
        self.right_trigger = False
        self.left_thumbstick = False
        self.right_thumbstick = False

        # Interaction state
        self.dragging = False
        self.resizing = False
        self.drag_local_offset = np.array([0, 0, 0])
        self.drag_controller_rot = None  # Store initial controller rotation
        self.initial_overlay_rot = [0, 0, 0]  # pitch, yaw, roll
        self.overlay_rot = [0, 0, 0]  # Current rotation
        self.initial_pinch_dist = 0
        self.initial_width = 0

        # Pointer overlays
        self.left_pointer_handle = None
        self.right_pointer_handle = None

        # Mouse controller
        self.mouse = MouseController()
        self.left_trigger_was_pressed = False
        self.right_trigger_was_pressed = False
        self.left_thumb_was_pressed = False
        self.right_thumb_was_pressed = False
        self.last_scroll_time = 0
        self.right_b_was_pressed = False
        self.overlay_visible = True

        # Stats
        self.frame_count = 0
        self.start_time = None
        self.last_status = 0
        self.last_frame_num = 0
        self.last_debug_uv = None

    def setup_signal_handlers(self):
        # Only set up signal handlers if running in main thread
        import threading
        if threading.current_thread() is threading.main_thread():
            def handler(signum, frame):
                print("\nStopping...")
                self.running = False
            signal.signal(signal.SIGINT, handler)
            signal.signal(signal.SIGTERM, handler)

    def wait_for_capture(self):
        print("Waiting for capture to start...")
        while self.running and not os.path.exists(SHM_FILE):
            time.sleep(0.1)
        if not self.running:
            return False

        self.shm_fd = os.open(SHM_FILE, os.O_RDONLY)
        file_size = os.fstat(self.shm_fd).st_size
        while self.running and file_size < 16:
            time.sleep(0.1)
            file_size = os.fstat(self.shm_fd).st_size

        self.shm = mmap.mmap(self.shm_fd, file_size, access=mmap.ACCESS_READ)
        return True

    def init_glfw(self):
        if not glfw.init():
            print("Failed to initialize GLFW")
            return False

        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.window = glfw.create_window(1, 1, "VR Desktop", None, None)
        if not self.window:
            print("Failed to create GLFW window")
            glfw.terminate()
            return False

        glfw.make_context_current(self.window)
        return True

    def init_openvr(self):
        print("Initializing OpenVR...")
        openvr.init(openvr.VRApplication_Overlay)
        self.vrsys = openvr.VRSystem()
        self.vroverlay = openvr.IVROverlay()

        # Main desktop overlay
        self.overlay_handle = self.vroverlay.createOverlay('vr_desktop', 'VR Desktop')
        self.vroverlay.setOverlayWidthInMeters(self.overlay_handle, self.overlay_width)
        self.vroverlay.setOverlayAlpha(self.overlay_handle, 1.0)
        self.update_overlay_transform()
        self.vroverlay.showOverlay(self.overlay_handle)

        # Pointer overlays (thin lines) - two per controller for double-sided
        self.left_pointer_handle = self.vroverlay.createOverlay('vr_desktop_ptr_l', 'Pointer Left')
        self.right_pointer_handle = self.vroverlay.createOverlay('vr_desktop_ptr_r', 'Pointer Right')
        self.left_pointer_back = self.vroverlay.createOverlay('vr_desktop_ptr_lb', 'Pointer Left Back')
        self.right_pointer_back = self.vroverlay.createOverlay('vr_desktop_ptr_rb', 'Pointer Right Back')

        # Cursor overlays (circles showing hit point)
        self.left_cursor_handle = self.vroverlay.createOverlay('vr_desktop_cur_l', 'Cursor Left')
        self.right_cursor_handle = self.vroverlay.createOverlay('vr_desktop_cur_r', 'Cursor Right')

        # Setup pointer overlays later when we have controller IDs
        self.pointer_tex_id = None
        self.cursor_tex_id = None

        # Find controllers
        self.find_controllers()

        # Create texture
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        self.texture = openvr.Texture_t()
        self.texture.eType = openvr.TextureType_OpenGL
        self.texture.eColorSpace = openvr.ColorSpace_Gamma

        print("VR overlay ready")
        return True

    def setup_pointer_overlays(self):
        """Setup pointer overlays attached to controllers"""
        if self.pointer_tex_id is not None:
            return  # Already setup

        if self.left_id is None or self.right_id is None:
            return

        # Ensure OpenGL context is current
        glfw.make_context_current(self.window)

        # Clear any existing OpenGL errors
        while glGetError() != GL_NO_ERROR:
            pass

        # Create texture for pointer - a simple colored rectangle
        # Width x Height - will be stretched along the ray
        width = 4
        height = 256
        pixels = np.zeros((height, width, 4), dtype=np.uint8)
        pixels[:, :, 0] = 100  # R
        pixels[:, :, 1] = 180  # G
        pixels[:, :, 2] = 255  # B
        pixels[:, :, 3] = 200  # A

        self.pointer_tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.pointer_tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels)

        tex = openvr.Texture_t()
        tex.handle = ctypes.cast(int(self.pointer_tex_id), ctypes.c_void_p)
        tex.eType = openvr.TextureType_OpenGL
        tex.eColorSpace = openvr.ColorSpace_Gamma

        # Setup front and back overlays for each controller (double-sided effect)
        pointer_pairs = [
            (self.left_pointer_handle, self.left_pointer_back, self.left_id),
            (self.right_pointer_handle, self.right_pointer_back, self.right_id)
        ]

        for ptr_front, ptr_back, ctrl_id in pointer_pairs:
            for ptr in [ptr_front, ptr_back]:
                self.vroverlay.setOverlayTexture(ptr, tex)
                self.vroverlay.setOverlayWidthInMeters(ptr, 0.005)

            # Front face transform
            transform_front = openvr.HmdMatrix34_t()
            transform_front[0][0] = 0.0
            transform_front[0][1] = 0.0
            transform_front[0][2] = 1.0
            transform_front[0][3] = 0.0

            transform_front[1][0] = 1.0
            transform_front[1][1] = 0.0
            transform_front[1][2] = 0.0
            transform_front[1][3] = 0.0

            transform_front[2][0] = 0.0
            transform_front[2][1] = 1.0
            transform_front[2][2] = 0.0
            transform_front[2][3] = 0.0

            # Back face transform (rotated 180 deg around Y)
            transform_back = openvr.HmdMatrix34_t()
            transform_back[0][0] = 0.0
            transform_back[0][1] = 0.0
            transform_back[0][2] = -1.0
            transform_back[0][3] = 0.0

            transform_back[1][0] = 1.0
            transform_back[1][1] = 0.0
            transform_back[1][2] = 0.0
            transform_back[1][3] = 0.0

            transform_back[2][0] = 0.0
            transform_back[2][1] = -1.0
            transform_back[2][2] = 0.0
            transform_back[2][3] = 0.0

            self.vroverlay.setOverlayTransformTrackedDeviceRelative(
                ptr_front, ctrl_id, transform_front
            )
            self.vroverlay.setOverlayTransformTrackedDeviceRelative(
                ptr_back, ctrl_id, transform_back
            )
            self.vroverlay.showOverlay(ptr_front)
            self.vroverlay.showOverlay(ptr_back)

        # Clear any OpenGL errors from OpenVR calls
        while glGetError() != GL_NO_ERROR:
            pass

        # Create cursor texture (circle)
        cursor_size = 32
        cursor_pixels = np.zeros((cursor_size, cursor_size, 4), dtype=np.uint8)
        center = cursor_size // 2
        radius = center - 2
        for y in range(cursor_size):
            for x in range(cursor_size):
                dist = math.sqrt((x - center) ** 2 + (y - center) ** 2)
                if dist <= radius:
                    # Filled circle with soft edge
                    alpha = 255 if dist < radius - 1 else int(255 * (radius - dist))
                    cursor_pixels[y, x] = [255, 255, 255, alpha]

        self.cursor_tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.cursor_tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, cursor_size, cursor_size, 0, GL_RGBA, GL_UNSIGNED_BYTE, cursor_pixels)

        cursor_tex = openvr.Texture_t()
        cursor_tex.handle = ctypes.cast(int(self.cursor_tex_id), ctypes.c_void_p)
        cursor_tex.eType = openvr.TextureType_OpenGL
        cursor_tex.eColorSpace = openvr.ColorSpace_Gamma

        for cursor, color in [(self.left_cursor_handle, (0.3, 0.7, 1.0)),
                               (self.right_cursor_handle, (1.0, 0.5, 0.3))]:
            self.vroverlay.setOverlayTexture(cursor, cursor_tex)
            self.vroverlay.setOverlayWidthInMeters(cursor, 0.03)  # 3cm cursor
            self.vroverlay.setOverlayAlpha(cursor, 1.0)
            self.vroverlay.setOverlayColor(cursor, *color)
            self.vroverlay.setOverlaySortOrder(cursor, 100)  # Render on top
            self.vroverlay.hideOverlay(cursor)  # Start hidden

        print("Pointer rays initialized")

    def update_overlay_transform(self):
        # Build rotation matrix from euler angles (pitch, yaw, roll)
        pitch, yaw, roll = self.overlay_rot

        # Rotation matrices
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        cr, sr = math.cos(roll), math.sin(roll)

        # Combined rotation: Yaw * Pitch * Roll
        r00 = cy * cr + sy * sp * sr
        r01 = -cy * sr + sy * sp * cr
        r02 = sy * cp
        r10 = cp * sr
        r11 = cp * cr
        r12 = -sp
        r20 = -sy * cr + cy * sp * sr
        r21 = sy * sr + cy * sp * cr
        r22 = cy * cp

        transform = openvr.HmdMatrix34_t()
        transform[0][0] = r00
        transform[0][1] = r01
        transform[0][2] = r02
        transform[0][3] = self.overlay_pos[0]
        transform[1][0] = r10
        transform[1][1] = r11
        transform[1][2] = r12
        transform[1][3] = self.overlay_pos[1]
        transform[2][0] = r20
        transform[2][1] = r21
        transform[2][2] = r22
        transform[2][3] = self.overlay_pos[2]

        self.vroverlay.setOverlayTransformAbsolute(
            self.overlay_handle,
            openvr.TrackingUniverseSeated,
            transform
        )
        self.vroverlay.setOverlayWidthInMeters(self.overlay_handle, self.overlay_width)

    def find_controllers(self):
        for i in range(openvr.k_unMaxTrackedDeviceCount):
            device_class = self.vrsys.getTrackedDeviceClass(i)
            if device_class == openvr.TrackedDeviceClass_Controller:
                role = self.vrsys.getControllerRoleForTrackedDeviceIndex(i)
                if role == openvr.TrackedControllerRole_LeftHand:
                    self.left_id = i
                elif role == openvr.TrackedControllerRole_RightHand:
                    self.right_id = i

    def get_controller_pose(self, controller_id):
        """Get controller position, forward direction, and rotation matrix"""
        if controller_id is None:
            return None, None, None

        poses = self.vrsys.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseSeated, 0,
            openvr.k_unMaxTrackedDeviceCount
        )
        pose = poses[controller_id]
        if not pose.bPoseIsValid:
            return None, None, None

        m = pose.mDeviceToAbsoluteTracking
        pos = np.array([m[0][3], m[1][3], m[2][3]])
        # Forward is -Z in controller space
        forward = np.array([-m[0][2], -m[1][2], -m[2][2]])
        # Extract rotation matrix (3x3)
        rot = np.array([
            [m[0][0], m[0][1], m[0][2]],
            [m[1][0], m[1][1], m[1][2]],
            [m[2][0], m[2][1], m[2][2]]
        ])
        return pos, forward, rot

    def matrix_to_euler(self, rot):
        """Convert rotation matrix to euler angles (pitch, yaw, roll)"""
        # Extract euler angles from rotation matrix
        if abs(rot[1][2]) < 0.99999:
            pitch = math.asin(-rot[1][2])
            yaw = math.atan2(rot[0][2], rot[2][2])
            roll = math.atan2(rot[1][0], rot[1][1])
        else:
            # Gimbal lock
            pitch = math.copysign(math.pi / 2, -rot[1][2])
            yaw = math.atan2(-rot[2][0], rot[0][0])
            roll = 0
        return [pitch, yaw, roll]

    def get_controller_state(self, controller_id):
        """Get grip, trigger, thumbstick state and axes"""
        if controller_id is None:
            return False, False, False, 0.0, 0.0, False

        result, state = self.vrsys.getControllerState(controller_id)
        if not result:
            return False, False, False, 0.0, 0.0, False

        grip = bool(state.ulButtonPressed & (1 << openvr.k_EButton_Grip))
        thumbstick = bool(state.ulButtonPressed & (1 << openvr.k_EButton_SteamVR_Touchpad))
        button_b = bool(state.ulButtonPressed & (1 << openvr.k_EButton_ApplicationMenu))

        # Use analog trigger value with threshold for responsive clicks
        # rAxis[1] is typically the trigger axis
        trigger_value = state.rAxis[1].x
        trigger = trigger_value > 0.75  # 75% threshold

        # Thumbstick axis for scrolling (rAxis[0] is typically thumbstick)
        thumb_x = state.rAxis[0].x
        thumb_y = state.rAxis[0].y

        return grip, trigger, thumbstick, thumb_x, thumb_y, button_b

    def ray_plane_intersection(self, ray_origin, ray_dir, plane_pos, plane_normal):
        """Find where ray intersects plane, returns t parameter or None"""
        denom = np.dot(plane_normal, ray_dir)
        if abs(denom) < 1e-6:
            return None

        t = np.dot(plane_pos - ray_origin, plane_normal) / denom
        if t < 0:
            return None
        return t

    def get_overlay_rotation_matrix(self):
        """Get the overlay's rotation matrix"""
        pitch, yaw, roll = self.overlay_rot
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        cr, sr = math.cos(roll), math.sin(roll)

        return np.array([
            [cy * cr + sy * sp * sr, -cy * sr + sy * sp * cr, sy * cp],
            [cp * sr, cp * cr, -sp],
            [-sy * cr + cy * sp * sr, sy * sr + cy * sp * cr, cy * cp]
        ])

    def point_in_overlay(self, point):
        """Check if 3D point is within overlay bounds, return normalized UV if inside"""
        # Get offset from overlay center in world space
        world_rel = point - np.array(self.overlay_pos)

        # Transform to overlay's local space (inverse rotation = transpose)
        rot = self.get_overlay_rotation_matrix()
        local_rel = rot.T @ world_rel

        half_width = self.overlay_width / 2
        half_height = half_width / self.overlay_aspect

        # Check bounds in local X/Y (local Z is depth)
        if abs(local_rel[0]) <= half_width and abs(local_rel[1]) <= half_height:
            # Calculate normalized UV (0-1 range, origin top-left)
            u = (local_rel[0] + half_width) / self.overlay_width
            v = 1.0 - (local_rel[1] + half_height) / (half_height * 2)
            return (u, v)
        return None

    def get_screen_hit(self, pos, forward):
        """Get screen UV coordinates where ray hits overlay, or None"""
        if pos is None or forward is None:
            return None

        # TODO: Use overlay rotation for proper plane normal
        plane_normal = np.array([0, 0, 1])
        plane_pos = np.array(self.overlay_pos)

        hit_t = self.ray_plane_intersection(pos, forward, plane_pos, plane_normal)
        if hit_t is None:
            return None

        hit_point = pos + forward * hit_t
        return self.point_in_overlay(hit_point)

    def update_pointer_visibility(self, handle_front, handle_back, visible):
        """Show or hide pointer overlay (both sides)"""
        if visible:
            self.vroverlay.showOverlay(handle_front)
            self.vroverlay.showOverlay(handle_back)
        else:
            self.vroverlay.hideOverlay(handle_front)
            self.vroverlay.hideOverlay(handle_back)

    def update_cursor(self, cursor_handle, uv):
        """Update cursor position based on UV coordinates, or hide if None"""
        if uv is None:
            self.vroverlay.hideOverlay(cursor_handle)
            return

        # Convert UV to local overlay coordinates
        # UV (0,0) is top-left, (1,1) is bottom-right
        half_width = self.overlay_width / 2
        half_height = half_width / self.overlay_aspect

        # Local position on overlay surface (overlay-space coords)
        local_x = (uv[0] - 0.5) * self.overlay_width
        local_y = (0.5 - uv[1]) * (half_height * 2)
        local_z = 0.01  # Slightly in front (in overlay's local Z)

        # Get overlay rotation matrix
        pitch, yaw, roll = self.overlay_rot
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        cr, sr = math.cos(roll), math.sin(roll)

        r00 = cy * cr + sy * sp * sr
        r01 = -cy * sr + sy * sp * cr
        r02 = sy * cp
        r10 = cp * sr
        r11 = cp * cr
        r12 = -sp
        r20 = -sy * cr + cy * sp * sr
        r21 = sy * sr + cy * sp * cr
        r22 = cy * cp

        # Transform local position to world position
        world_x = self.overlay_pos[0] + r00 * local_x + r01 * local_y + r02 * local_z
        world_y = self.overlay_pos[1] + r10 * local_x + r11 * local_y + r12 * local_z
        world_z = self.overlay_pos[2] + r20 * local_x + r21 * local_y + r22 * local_z

        # Cursor uses same rotation as overlay so it faces the user
        transform = openvr.HmdMatrix34_t()
        transform[0][0] = r00
        transform[0][1] = r01
        transform[0][2] = r02
        transform[0][3] = world_x
        transform[1][0] = r10
        transform[1][1] = r11
        transform[1][2] = r12
        transform[1][3] = world_y
        transform[2][0] = r20
        transform[2][1] = r21
        transform[2][2] = r22
        transform[2][3] = world_z

        self.vroverlay.setOverlayTransformAbsolute(
            cursor_handle,
            openvr.TrackingUniverseSeated,
            transform
        )
        self.vroverlay.showOverlay(cursor_handle)

    def handle_controllers(self):
        """Process controller input for interaction"""
        if self.left_id is None or self.right_id is None:
            self.find_controllers()

        # Setup pointer overlays once we have controller IDs
        self.setup_pointer_overlays()

        # Get states
        left_pos, left_fwd, left_rot = self.get_controller_pose(self.left_id)
        right_pos, right_fwd, right_rot = self.get_controller_pose(self.right_id)

        left_grip, left_trigger, left_thumb, left_thumb_x, left_thumb_y, left_b = self.get_controller_state(self.left_id)
        right_grip, right_trigger, right_thumb, right_thumb_x, right_thumb_y, right_b = self.get_controller_state(self.right_id)

        # Toggle overlay visibility with right controller B button
        if right_b and not self.right_b_was_pressed:
            self.overlay_visible = not self.overlay_visible
            if self.overlay_visible:
                self.vroverlay.showOverlay(self.overlay_handle)
            else:
                self.vroverlay.hideOverlay(self.overlay_handle)
                self.vroverlay.hideOverlay(self.left_cursor_handle)
                self.vroverlay.hideOverlay(self.right_cursor_handle)
            print(f"Overlay {'visible' if self.overlay_visible else 'hidden'}")
        self.right_b_was_pressed = right_b

        # Skip all interaction if overlay is hidden
        if not self.overlay_visible:
            self.update_pointer_visibility(self.left_pointer_handle, self.left_pointer_back, False)
            self.update_pointer_visibility(self.right_pointer_handle, self.right_pointer_back, False)
            return

        # Plane normal - rotate [0,0,1] by overlay rotation
        rot = self.get_overlay_rotation_matrix()
        plane_normal = rot @ np.array([0, 0, 1])
        plane_pos = np.array(self.overlay_pos)

        # Calculate ray hits and screen positions
        left_uv = None
        right_uv = None

        if left_pos is not None and left_fwd is not None:
            hit_t = self.ray_plane_intersection(left_pos, left_fwd, plane_pos, plane_normal)
            if hit_t:
                hit_point = left_pos + left_fwd * hit_t
                left_uv = self.point_in_overlay(hit_point)

        if right_pos is not None and right_fwd is not None:
            hit_t = self.ray_plane_intersection(right_pos, right_fwd, plane_pos, plane_normal)
            if hit_t:
                hit_point = right_pos + right_fwd * hit_t
                right_uv = self.point_in_overlay(hit_point)

        # Debug output for screen position (only when UV changes significantly)
        if DEBUG:
            current_uv = left_uv or right_uv
            if current_uv:
                # Only print when UV changes by more than 1%
                should_print = self.last_debug_uv is None
                if not should_print and self.last_debug_uv:
                    du = abs(current_uv[0] - self.last_debug_uv[0])
                    dv = abs(current_uv[1] - self.last_debug_uv[1])
                    should_print = du > 0.01 or dv > 0.01

                if should_print:
                    self.last_debug_uv = current_uv
                    hand = 'L' if left_uv else 'R'
                    pixel_x = int(current_uv[0] * self.current_width) if self.current_width else 0
                    pixel_y = int(current_uv[1] * self.current_height) if self.current_height else 0
                    print(f"  {hand}: UV({current_uv[0]:.2f}, {current_uv[1]:.2f}) px({pixel_x}, {pixel_y})")

        # Update pointer visibility
        left_visible = DEBUG or left_uv is not None
        right_visible = DEBUG or right_uv is not None

        self.update_pointer_visibility(self.left_pointer_handle, self.left_pointer_back, left_visible)
        self.update_pointer_visibility(self.right_pointer_handle, self.right_pointer_back, right_visible)

        # Update cursor positions using UV coordinates
        self.update_cursor(self.left_cursor_handle, left_uv)
        self.update_cursor(self.right_cursor_handle, right_uv)

        # Handle grip interactions - only when pointing at overlay
        left_pointing = left_uv is not None
        right_pointing = right_uv is not None

        both_grips = left_grip and right_grip and left_pointing and right_pointing
        left_grab = left_grip and left_pointing and not right_grip
        right_grab = right_grip and right_pointing and not left_grip

        if both_grips and left_pos is not None and right_pos is not None:
            # Pinch resize
            dist = np.linalg.norm(left_pos - right_pos)
            if not self.resizing:
                self.resizing = True
                self.dragging = False
                self.initial_pinch_dist = dist
                self.initial_width = self.overlay_width
            else:
                scale = dist / self.initial_pinch_dist
                self.overlay_width = max(0.5, min(10.0, self.initial_width * scale))
                # Also move to midpoint
                mid = (left_pos + right_pos) / 2
                self.overlay_pos[0] = mid[0]
                self.overlay_pos[1] = mid[1]
                self.update_overlay_transform()

        elif left_grab or right_grab:
            # Single grip drag with rotation - overlay orbits around controller
            grip_pos = right_pos if right_grab else left_pos
            grip_rot = right_rot if right_grab else left_rot

            if grip_pos is not None and grip_rot is not None:
                if not self.dragging:
                    self.dragging = True
                    self.resizing = False
                    # Store offset in controller's LOCAL space
                    overlay_pos = np.array(self.overlay_pos)
                    # Transform world offset to controller local space
                    world_offset = overlay_pos - grip_pos
                    # Inverse of rotation matrix = transpose for orthonormal
                    self.drag_local_offset = grip_rot.T @ world_offset
                    self.drag_controller_rot = grip_rot.copy()
                    self.initial_overlay_rot = list(self.overlay_rot)
                else:
                    # Transform local offset back to world space using current controller rotation
                    world_offset = grip_rot @ self.drag_local_offset
                    self.overlay_pos[0] = grip_pos[0] + world_offset[0]
                    self.overlay_pos[1] = grip_pos[1] + world_offset[1]
                    self.overlay_pos[2] = grip_pos[2] + world_offset[2]

                    # Calculate rotation delta
                    initial_euler = self.matrix_to_euler(self.drag_controller_rot)
                    current_euler = self.matrix_to_euler(grip_rot)

                    # Apply rotation delta to overlay
                    self.overlay_rot[0] = self.initial_overlay_rot[0] + (current_euler[0] - initial_euler[0])
                    self.overlay_rot[1] = self.initial_overlay_rot[1] + (current_euler[1] - initial_euler[1])
                    self.overlay_rot[2] = self.initial_overlay_rot[2] + (current_euler[2] - initial_euler[2])

                    self.update_overlay_transform()
        elif not left_grip and not right_grip:
            self.dragging = False
            self.resizing = False
            self.drag_controller_rot = None

        # Mouse click handling - use the controller that's clicking
        if self.current_width > 0 and self.current_height > 0:
            # Left controller trigger
            if left_trigger and left_uv:
                pixel_x = int(left_uv[0] * self.current_width)
                pixel_y = int(left_uv[1] * self.current_height)
                self.mouse.position = (pixel_x, pixel_y)
                if not self.left_trigger_was_pressed:
                    self.mouse.press(Button.left)
            elif not left_trigger and self.left_trigger_was_pressed:
                self.mouse.release(Button.left)
            self.left_trigger_was_pressed = left_trigger and left_uv is not None

            # Right controller trigger
            if right_trigger and right_uv:
                pixel_x = int(right_uv[0] * self.current_width)
                pixel_y = int(right_uv[1] * self.current_height)
                self.mouse.position = (pixel_x, pixel_y)
                if not self.right_trigger_was_pressed:
                    self.mouse.press(Button.left)
            elif not right_trigger and self.right_trigger_was_pressed:
                self.mouse.release(Button.left)
            self.right_trigger_was_pressed = right_trigger and right_uv is not None

            # Left controller thumbstick click = right click
            if left_thumb and left_uv:
                pixel_x = int(left_uv[0] * self.current_width)
                pixel_y = int(left_uv[1] * self.current_height)
                self.mouse.position = (pixel_x, pixel_y)
                if not self.left_thumb_was_pressed:
                    self.mouse.press(Button.right)
            elif not left_thumb and self.left_thumb_was_pressed:
                self.mouse.release(Button.right)
            self.left_thumb_was_pressed = left_thumb and left_uv is not None

            # Right controller thumbstick click = right click
            if right_thumb and right_uv:
                pixel_x = int(right_uv[0] * self.current_width)
                pixel_y = int(right_uv[1] * self.current_height)
                self.mouse.position = (pixel_x, pixel_y)
                if not self.right_thumb_was_pressed:
                    self.mouse.press(Button.right)
            elif not right_thumb and self.right_thumb_was_pressed:
                self.mouse.release(Button.right)
            self.right_thumb_was_pressed = right_thumb and right_uv is not None

            # Scrolling - use the controller that's scrolling
            left_scrolling = abs(left_thumb_y) > 0.3 or abs(left_thumb_x) > 0.3
            right_scrolling = abs(right_thumb_y) > 0.3 or abs(right_thumb_x) > 0.3

            scroll_uv = None
            thumb_y = 0
            thumb_x = 0
            if left_scrolling and left_uv:
                scroll_uv = left_uv
                thumb_y = left_thumb_y
                thumb_x = left_thumb_x
            elif right_scrolling and right_uv:
                scroll_uv = right_uv
                thumb_y = right_thumb_y
                thumb_x = right_thumb_x

            if scroll_uv:
                pixel_x = int(scroll_uv[0] * self.current_width)
                pixel_y = int(scroll_uv[1] * self.current_height)
                self.mouse.position = (pixel_x, pixel_y)

                deadzone = 0.3
                now = time.time()
                max_val = max(abs(thumb_y), abs(thumb_x))
                normalized = (max_val - deadzone) / (1.0 - deadzone)
                interval = 0.2 - (normalized * 0.18)

                if now - self.last_scroll_time >= interval:
                    self.last_scroll_time = now
                    if abs(thumb_y) > deadzone:
                        self.mouse.scroll(0, 1 if thumb_y > 0 else -1)
                    if abs(thumb_x) > deadzone:
                        self.mouse.scroll(1 if thumb_x > 0 else -1, 0)

    def update_frame(self):
        """Read frame from shared memory and update texture"""
        try:
            self.shm.seek(0)
            header = self.shm.read(16)
            width = int.from_bytes(header[0:4], 'little')
            height = int.from_bytes(header[4:8], 'little')
            frame_num = int.from_bytes(header[8:16], 'little')

            if frame_num == self.last_frame_num or width == 0 or height == 0:
                return

            # Remap if size changed
            expected_size = 16 + (width * height * 4)
            if width != self.current_width or height != self.current_height:
                self.current_width = width
                self.current_height = height
                self.overlay_aspect = width / height
                self.shm.close()
                file_size = os.fstat(self.shm_fd).st_size
                if file_size >= expected_size:
                    self.shm = mmap.mmap(self.shm_fd, file_size, access=mmap.ACCESS_READ)
                    print(f"Resolution: {width}x{height}")
                else:
                    return

            # Read and flip
            self.shm.seek(16)
            data = self.shm.read(width * height * 4)
            arr = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)
            arr = np.flipud(arr).copy()

            # Upload texture
            glBindTexture(GL_TEXTURE_2D, self.texture_id)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, arr)

            self.texture.handle = ctypes.cast(int(self.texture_id), ctypes.c_void_p)
            self.vroverlay.setOverlayTexture(self.overlay_handle, self.texture)

            self.last_frame_num = frame_num
            self.frame_count += 1

        except Exception as e:
            import traceback
            traceback.print_exc()

    def run(self):
        print("VR Desktop Viewer")
        print("=" * 50)
        print()
        print("Run 'python test_wayland_capture.py' in another terminal first!")
        print()

        self.setup_signal_handlers()

        if not self.wait_for_capture():
            return 1

        if not self.init_glfw():
            return 1

        if not self.init_openvr():
            return 1

        self.start_time = time.time()
        self.last_status = time.time()

        try:
            while self.running:
                glfw.poll_events()
                self.handle_controllers()
                self.update_frame()

                now = time.time()
                if now - self.last_status >= 1.0:
                    elapsed = now - self.start_time
                    fps = self.frame_count / elapsed if elapsed > 0 else 0
                    print(f"VR updates: {self.frame_count}, {fps:.1f} FPS | Size: {self.overlay_width:.2f}m")
                    self.last_status = now

                time.sleep(1/120)

        finally:
            self.cleanup()

        print("Done")
        return 0

    def cleanup(self):
        if self.window:
            glfw.make_context_current(self.window)
        if self.texture_id:
            glDeleteTextures(1, [self.texture_id])
        if self.pointer_tex_id:
            glDeleteTextures(1, [self.pointer_tex_id])
        if self.cursor_tex_id:
            glDeleteTextures(1, [self.cursor_tex_id])
        if self.shm:
            self.shm.close()
        if self.shm_fd:
            os.close(self.shm_fd)
        if self.overlay_handle:
            self.vroverlay.destroyOverlay(self.overlay_handle)
        for ptr in [self.left_pointer_handle, self.left_pointer_back,
                    self.right_pointer_handle, self.right_pointer_back,
                    self.left_cursor_handle, self.right_cursor_handle]:
            if ptr:
                try:
                    self.vroverlay.destroyOverlay(ptr)
                except:
                    pass
        openvr.shutdown()
        if self.window:
            glfw.destroy_window(self.window)
        glfw.terminate()


if __name__ == '__main__':
    viewer = VRDesktopViewer()
    sys.exit(viewer.run())
