import cv2
import numpy as np
import torch
from PyQt6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QSizePolicy,
    QGraphicsPathItem, QGraphicsEllipseItem
)
from PyQt6.QtGui import QPixmap, QPainter, QImage, QPainterPath, QPen, QColor, QBrush
from PyQt6.QtCore import Qt, QBuffer, QIODevice, QRectF
from providers.sam_model_provider import SAMModelProvider

class CustomGraphicsView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.undo_callback = None  # This will be set by MainWindow
        self._undo_state_saved_current_action = False  # Ensure one save per action

        # Create scene and set view properties
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        # Scene items
        self.background_pixmap_item = None    # Background image item
        self.selected_pixmap_item = None      # Overlay for the selected region
        self.detection_overlay_item = None    # Detection overlay item
        self.quick_selection_overlay = None   # Quick selection ellipse
        self.clone_stamp_overlay = None       # Clone stamp overlay ellipse
        self.clone_source_overlay = None      # Clone source overlay ellipse

        # Transient state items
        self.selection_feedback_items = []    # For drawing feedback outlines (transient)
        self.dragging = False

        # Mode and selection points
        self.mode = "transform"  # Modes: "transform", "object selection", "quick selection", "clone stamp"
        self.positive_points = []  # For SAM prompt-based selection
        self.negative_points = []  # For SAM prompt-based selection

        # For Quick Selection
        self.quick_select_brush_size = 5

        # For Clone Stamp
        self.clone_stamp_brush_size = 5
        self.clone_source_point = None
        self.clone_offset = None

        # For selection mask and images
        self.selection_mask = None      # Binary mask from selection
        self.image_shape = None         # (height, width) of current image
        self.current_cv_image = None    # The current working image (BGRA or BGR)
        self.detection_cv_image = None  # Composite detection image
        self.display_cv_image = None    # RGBA image for display
        self.sam_cv_image = None        # Possibly enhanced image for SAM
        self.sam_cv_image_rgb = None    # RGB version for SAM

        # Other overlays
        self.checkerboard_pixmap = None

    # --- Helper methods ---
    def remove_item_safely(self, item):
        try:
            if item is not None and item.scene() is not None:
                self.scene.removeItem(item)
        except RuntimeError:
            # The item was already deleted – simply pass
            pass

    def clear_selection_feedback_items(self):
        """Safely remove all feedback items from the scene."""
        for item in list(self.selection_feedback_items):
            self.remove_item_safely(item)
        self.selection_feedback_items.clear()

    # --- State Saving ---
    def get_state(self):
        """Return a dictionary capturing the complete state of the view."""
        state = {}
        # Core state
        state["mode"] = self.mode
        state["dragging"] = self.dragging
        state["positive_points"] = list(self.positive_points)
        state["negative_points"] = list(self.negative_points)
        state["selection_mask"] = np.copy(self.selection_mask) if self.selection_mask is not None else None
        state["image_shape"] = self.image_shape

        # CV images (deep copies)
        state["current_cv_image"] = np.copy(self.current_cv_image) if self.current_cv_image is not None else None
        state["detection_cv_image"] = np.copy(self.detection_cv_image) if self.detection_cv_image is not None else None
        state["display_cv_image"] = np.copy(self.display_cv_image) if self.display_cv_image is not None else None
        state["sam_cv_image"] = np.copy(self.sam_cv_image) if self.sam_cv_image is not None else None
        state["sam_cv_image_rgb"] = np.copy(self.sam_cv_image_rgb) if self.sam_cv_image_rgb is not None else None

        # Brush & clone settings
        state["quick_select_brush_size"] = self.quick_select_brush_size
        state["clone_stamp_brush_size"] = self.clone_stamp_brush_size
        state["clone_source_point"] = self.clone_source_point
        state["clone_offset"] = self.clone_offset

        # Scene items: background, selected overlay, detection overlay
        if self.background_pixmap_item and self.background_pixmap_item.pixmap():
            state["background_pixmap"] = self.background_pixmap_item.pixmap().copy()
            state["background_position"] = self.background_pixmap_item.pos()
        else:
            state["background_pixmap"] = None
            state["background_position"] = None

        if self.selected_pixmap_item and self.selected_pixmap_item.pixmap():
            state["selected_pixmap"] = self.selected_pixmap_item.pixmap().copy()
            state["selected_position"] = self.selected_pixmap_item.pos()
        else:
            state["selected_pixmap"] = None
            state["selected_position"] = None

        if self.detection_overlay_item and self.detection_overlay_item.pixmap():
            state["detection_overlay_pixmap"] = self.detection_overlay_item.pixmap().copy()
            state["detection_overlay_position"] = self.detection_overlay_item.pos()
        else:
            state["detection_overlay_pixmap"] = None
            state["detection_overlay_position"] = None

        # Overlays geometry
        state["quick_selection_rect"] = self.quick_selection_overlay.rect() if self.quick_selection_overlay is not None else None
        state["clone_stamp_rect"] = self.clone_stamp_overlay.rect() if self.clone_stamp_overlay is not None else None
        state["clone_source_rect"] = self.clone_source_overlay.rect() if self.clone_source_overlay is not None else None

        # View transform
        state["view_transform"] = self.transform()

        # Transient feedback items are not saved.
        return state

    def set_state(self, state):
        """Restore the state from the given dictionary."""
        self.mode = state.get("mode", "transform")
        self.dragging = state.get("dragging", False)
        self.positive_points = state.get("positive_points", [])
        self.negative_points = state.get("negative_points", [])
        mask = state.get("selection_mask")
        if mask is not None:
            self.selection_mask = np.copy(mask)
        elif self.current_cv_image is not None:
            h, w = self.current_cv_image.shape[:2]
            self.selection_mask = np.zeros((h, w), dtype=np.uint8)
        else:
            self.selection_mask = None

        self.image_shape = state.get("image_shape")
        self.current_cv_image = np.copy(state.get("current_cv_image")) if state.get("current_cv_image") is not None else None
        self.detection_cv_image = np.copy(state.get("detection_cv_image")) if state.get("detection_cv_image") is not None else None
        self.display_cv_image = np.copy(state.get("display_cv_image")) if state.get("display_cv_image") is not None else None
        self.sam_cv_image = np.copy(state.get("sam_cv_image")) if state.get("sam_cv_image") is not None else None
        self.sam_cv_image_rgb = np.copy(state.get("sam_cv_image_rgb")) if state.get("sam_cv_image_rgb") is not None else None

        self.quick_select_brush_size = state.get("quick_select_brush_size", 5)
        self.clone_stamp_brush_size = state.get("clone_stamp_brush_size", 5)
        self.clone_source_point = state.get("clone_source_point")
        self.clone_offset = state.get("clone_offset")

        # Clear the scene completely to avoid using deleted objects.
        self.scene.clear()
        self.selected_pixmap_item = None
        self.background_pixmap_item = None
        self.detection_overlay_item = None
        self.quick_selection_overlay = None
        self.clone_stamp_overlay = None
        self.clone_source_overlay = None
        self.selection_feedback_items.clear()

        # Recreate background_pixmap_item
        bg_pixmap = state.get("background_pixmap")
        if bg_pixmap is not None:
            from PyQt6.QtWidgets import QGraphicsPixmapItem
            self.background_pixmap_item = QGraphicsPixmapItem(bg_pixmap)
            self.scene.addItem(self.background_pixmap_item)
            pos = state.get("background_position")
            if pos is not None:
                self.background_pixmap_item.setPos(pos)

        # Recreate selected_pixmap_item
        sel_pix = state.get("selected_pixmap")
        if sel_pix is not None:
            from PyQt6.QtWidgets import QGraphicsPixmapItem
            self.selected_pixmap_item = QGraphicsPixmapItem(sel_pix)
            self.selected_pixmap_item.setZValue(10)
            self.scene.addItem(self.selected_pixmap_item)
            pos = state.get("selected_position")
            if pos is not None:
                self.selected_pixmap_item.setPos(pos)

        # Recreate detection_overlay_item
        detect_pix = state.get("detection_overlay_pixmap")
        if detect_pix is not None:
            from PyQt6.QtWidgets import QGraphicsPixmapItem
            self.detection_overlay_item = QGraphicsPixmapItem(detect_pix)
            self.scene.addItem(self.detection_overlay_item)
            pos = state.get("detection_overlay_position")
            if pos is not None:
                self.detection_overlay_item.setPos(pos)

        # Recreate quick selection overlay (ellipse)
        qrect = state.get("quick_selection_rect")
        if qrect is not None:
            from PyQt6.QtWidgets import QGraphicsEllipseItem
            self.quick_selection_overlay = QGraphicsEllipseItem()
            self.quick_selection_overlay.setRect(qrect)
            self.scene.addItem(self.quick_selection_overlay)

        # Recreate clone stamp overlays
        cstamp_rect = state.get("clone_stamp_rect")
        if cstamp_rect is not None:
            from PyQt6.QtWidgets import QGraphicsEllipseItem
            self.clone_stamp_overlay = QGraphicsEllipseItem()
            self.clone_stamp_overlay.setRect(cstamp_rect)
            self.scene.addItem(self.clone_stamp_overlay)

        csource_rect = state.get("clone_source_rect")
        if csource_rect is not None:
            from PyQt6.QtWidgets import QGraphicsEllipseItem
            self.clone_source_overlay = QGraphicsEllipseItem()
            self.clone_source_overlay.setRect(csource_rect)
            self.scene.addItem(self.clone_source_overlay)

        # Restore view transformation
        view_transform = state.get("view_transform")
        if view_transform is not None:
            self.setTransform(view_transform)

        # Optionally, adjust the scene rect if background exists
        if self.background_pixmap_item is not None:
            self.setSceneRect(self.background_pixmap_item.boundingRect())
            self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

        # Transient feedback items are reset (will be re-created by update_display)
        self.selection_feedback_items = []

    # -------------------------------------------------------------------------
    # Basic image conversion and drawing methods
    # -------------------------------------------------------------------------

    def drawBackground(self, painter, rect):
        if self.background_pixmap_item:
            image_rect = self.background_pixmap_item.boundingRect()
        else:
            image_rect = rect
        tile_size = max(20, int(min(image_rect.width(), image_rect.height()) / 40))
        if self.checkerboard_pixmap is None or self.checkerboard_pixmap.width() != tile_size:
            self.checkerboard_pixmap = QPixmap(tile_size, tile_size)
            self.checkerboard_pixmap.fill(Qt.GlobalColor.white)
            tile_painter = QPainter(self.checkerboard_pixmap)
            tile_painter.fillRect(0, 0, tile_size // 2, tile_size // 2, Qt.GlobalColor.lightGray)
            tile_painter.fillRect(tile_size // 2, tile_size // 2, tile_size // 2, tile_size // 2, Qt.GlobalColor.lightGray)
            tile_painter.end()
        brush = QBrush(self.checkerboard_pixmap)
        painter.fillRect(image_rect, brush)

    def save(self, filepath=None):
        if filepath and self.background_pixmap_item:
            self.background_pixmap_item.pixmap().save(filepath, None, 100)


    # -------------------------------------------------------------------------
    # Load image
    # -------------------------------------------------------------------------
    def load_image(self, image_path):
        if self.selection_mask is not None and np.count_nonzero(self.selection_mask) > 0:
            self.apply_merge()
        self.clear_detection()
        self.scene.clear()
        self.selected_pixmap_item = None
        self.selection_feedback_items.clear()
        self.positive_points.clear()
        self.negative_points.clear()
        self.selection_mask = None
        if self.clone_stamp_overlay:
            self.remove_item_safely(self.clone_stamp_overlay)
            self.clone_stamp_overlay = None
        if self.clone_source_overlay:
            self.remove_item_safely(self.clone_source_overlay)
            self.clone_source_overlay = None

        self.image_path = image_path
        self.current_cv_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if self.current_cv_image is None:
            print(f"Error: Could not load image from {image_path}")
            return
        h, w = self.current_cv_image.shape[:2]
        self.selection_mask = np.zeros((h, w), dtype=np.uint8)
        self.detection_cv_image = self.current_cv_image.copy()

        if not image_path.lower().endswith('.png'):
            ret, buf = cv2.imencode('.png', self.current_cv_image)
            if ret:
                png_bytes = buf.tobytes()
                pixmap = QPixmap()
                pixmap.loadFromData(png_bytes, "PNG")
            else:
                print("Error: Could not encode image as PNG.")
                return
        else:
            pixmap = QPixmap(image_path)

        from PyQt6.QtWidgets import QGraphicsPixmapItem
        self.background_pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.background_pixmap_item)
        self.setSceneRect(self.background_pixmap_item.boundingRect())
        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def update_all_cv_image_conversions(self):
        if self.current_cv_image is None or len(self.current_cv_image.shape) < 3:
            return
        new_h, new_w = self.current_cv_image.shape[:2]
        self.image_shape = (new_h, new_w)
        if self.current_cv_image.shape[2] == 4:
            self.display_cv_image = cv2.cvtColor(self.current_cv_image, cv2.COLOR_BGRA2RGBA)
        else:
            self.display_cv_image = cv2.cvtColor(self.current_cv_image, cv2.COLOR_BGR2RGBA)
        self.sam_cv_image = self.apply_contrast_and_sharpen(self.current_cv_image)
        if self.current_cv_image.shape[2] == 4:
            self.sam_cv_image_rgb = cv2.cvtColor(self.sam_cv_image, cv2.COLOR_BGRA2RGB)
        else:
            self.sam_cv_image_rgb = cv2.cvtColor(self.sam_cv_image, cv2.COLOR_BGR2RGB)
        if self.selection_mask is not None:
            mask_h, mask_w = self.selection_mask.shape[:2]
            if (mask_h, mask_w) != (new_h, new_w):
                self.selection_mask = cv2.resize(self.selection_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    def update_display(self):
        """Update the display by re-drawing the background and selection overlays."""
        if self.current_cv_image is None or self.selection_mask is None:
            return
        if self.display_cv_image is None:
            self.update_all_cv_image_conversions()

        # Remove the old selected_pixmap_item safely
        self.remove_item_safely(self.selected_pixmap_item)
        self.selected_pixmap_item = None

        # Remove previous feedback items safely
        self.clear_selection_feedback_items()

        # Update the background pixmap to show "holes" for selected regions
        bg_rgba = self.display_cv_image.copy()
        if bg_rgba.shape[2] == 4:
            orig_alpha = bg_rgba[..., 3].copy()
        else:
            orig_alpha = np.full((bg_rgba.shape[0], bg_rgba.shape[1]), 255, dtype=np.uint8)
        bg_rgba[..., 3] = np.where(self.selection_mask == 255, 0, orig_alpha)
        h, w, ch = bg_rgba.shape
        bytes_per_line = ch * w
        bg_qimage = QImage(bg_rgba.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888)
        bg_pixmap = QPixmap.fromImage(bg_qimage)
        if self.background_pixmap_item:
            self.background_pixmap_item.setPixmap(bg_pixmap)

        # Create the selected overlay (only the selected region)
        sel_rgba = self.display_cv_image.copy()
        sel_rgba[self.selection_mask != 255] = [0, 0, 0, 0]
        sel_qimage = QImage(sel_rgba.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888)
        sel_pixmap = QPixmap.fromImage(sel_qimage)

        from PyQt6.QtWidgets import QGraphicsPixmapItem
        self.selected_pixmap_item = QGraphicsPixmapItem(sel_pixmap)
        self.selected_pixmap_item.setZValue(10)
        self.scene.addItem(self.selected_pixmap_item)

        # Draw selection outline using a white then black pen for contrast
        outline_path = self._get_outline_path(self.selection_mask)
        white_pen = QPen(QColor("white"), 2)
        item_white = QGraphicsPathItem(outline_path, self.selected_pixmap_item)
        item_white.setPen(white_pen)
        black_pen = QPen(QColor("black"), 1)
        item_black = QGraphicsPathItem(outline_path, self.selected_pixmap_item)
        item_black.setPen(black_pen)
        self.selection_feedback_items = [item_white, item_black]

        self.update_detection_composite()

    def _get_outline_path(self, binary_mask):
        kernel = np.ones((3, 3), np.uint8)
        outline_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_GRADIENT, kernel)
        contours, _ = cv2.findContours(outline_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        path = QPainterPath()
        for cnt in contours:
            if len(cnt) == 0:
                continue
            cnt = cnt.squeeze()
            if cnt.ndim < 2:
                continue
            path.moveTo(cnt[0][0], cnt[0][1])
            for pt in cnt[1:]:
                path.lineTo(pt[0], pt[1])
            path.closeSubpath()
        return path

    # -------------------------------------------------------------------------
    # Alpha compositing helper
    # -------------------------------------------------------------------------
    def alpha_composite(self, bg, fg):
        """Composite fg over bg using per-pixel alpha blending. Both should be RGBA numpy arrays."""
        bg = bg.astype(np.float32) / 255.0
        fg = fg.astype(np.float32) / 255.0
        alpha_fg = fg[..., 3:4]
        alpha_bg = bg[..., 3:4]
        out_alpha = alpha_fg + alpha_bg * (1 - alpha_fg)
        out_rgb = (fg[..., :3] * alpha_fg + bg[..., :3] * alpha_bg * (1 - alpha_fg)) / (out_alpha + 1e-6)
        out = np.concatenate([out_rgb, out_alpha], axis=2)
        out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
        return out

    def update_detection_composite(self):
        """Update detection_cv_image to include the selection overlay."""
        if self.display_cv_image is None or self.selection_mask is None:
            return
        bg_rgba = self.display_cv_image.copy()
        if bg_rgba.shape[2] == 4:
            orig_alpha = bg_rgba[..., 3].copy()
        else:
            orig_alpha = np.full((bg_rgba.shape[0], bg_rgba.shape[1]), 255, dtype=np.uint8)
        bg_rgba[..., 3] = np.where(self.selection_mask == 255, 0, orig_alpha)
        sel_rgba = self.display_cv_image.copy()
        sel_rgba[self.selection_mask != 255] = [0, 0, 0, 0]
        composite = self.alpha_composite(bg_rgba, sel_rgba)
        self.detection_cv_image = composite

    def clear_detection(self):
        if self.detection_overlay_item:
            self.remove_item_safely(self.detection_overlay_item)
            self.detection_overlay_item = None

    def set_mode(self, mode):
        self.mode = mode
        print(f"Mode set to: {mode}")
        if mode != "selection":
            self.positive_points.clear()
            self.negative_points.clear()
        if mode != "transform" and self.current_cv_image is not None:
            self.update_all_cv_image_conversions()
        if mode != "quick selection" and self.quick_selection_overlay:
            self.remove_item_safely(self.quick_selection_overlay)
            self.quick_selection_overlay = None
        if mode != "clone stamp":
            if self.clone_stamp_overlay:
                self.remove_item_safely(self.clone_stamp_overlay)
                self.clone_stamp_overlay = None
            if self.clone_source_overlay:
                self.remove_item_safely(self.clone_source_overlay)
                self.clone_source_overlay = None
            self.clone_offset = None

    # -------------------------------------------------------------------------
    # Mouse Events
    # -------------------------------------------------------------------------
    def _maybe_save_undo_state(self):
        if not self._undo_state_saved_current_action and self.undo_callback is not None:
            self.undo_callback()  # MainWindow will call save_undo_state()
            self._undo_state_saved_current_action = True

    def mousePressEvent(self, event):
        self._undo_state_saved_current_action = False
        pos = self.mapToScene(event.pos())
        if self.mode == "object selection":
            self._maybe_save_undo_state()
            if event.button() == Qt.MouseButton.LeftButton:
                self.positive_points.append([pos.x(), pos.y()])
                print(f"Added positive point: ({pos.x()}, {pos.y()})")
            elif event.button() == Qt.MouseButton.RightButton:
                self.negative_points.append([pos.x(), pos.y()])
                print(f"Added negative point: ({pos.x()}, {pos.y()})")
            self.object_selection()
        elif self.mode == "quick selection":
            self._maybe_save_undo_state()
            self._quick_select_at_position(pos, event.button())
            self._update_quick_selection_overlay(pos)
        elif self.mode == "clone stamp":
            self._maybe_save_undo_state()
            if event.button() == Qt.MouseButton.RightButton:
                offset = self.background_pixmap_item.pos()
                sample_x = int(pos.x() - offset.x())
                sample_y = int(pos.y() - offset.y())
                self.clone_source_point = (sample_x, sample_y)
                self.clone_offset = None
                self._update_clone_source_overlay(pos)
                print(f"Clone source set at: ({sample_x}, {sample_y})")
            elif event.button() == Qt.MouseButton.LeftButton:
                if self.clone_source_point is None:
                    print("Clone source not set. Right-click to set it.")
                    return
                if self.clone_offset is None:
                    offset = self.background_pixmap_item.pos()
                    dest_x = int(pos.x() - offset.x())
                    dest_y = int(pos.y() - offset.y())
                    self.clone_offset = (self.clone_source_point[0] - dest_x, self.clone_source_point[1] - dest_y)
                    print(f"Clone offset computed as: {self.clone_offset}")
                self._clone_stamp_at_position(pos)
                self._update_clone_stamp_overlay(pos)
        elif self.mode == "transform" and self.selected_pixmap_item:
            self._maybe_save_undo_state()
            self.dragging = True
            self.drag_start = pos
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        pos = self.mapToScene(event.pos())
        # Save undo state for all modes if not already done this action
        if self.mode in ["quick selection", "clone stamp", "object selection",
                         "transform"] and not self._undo_state_saved_current_action:
            self._maybe_save_undo_state()

        if self.mode == "quick selection" and event.buttons() in [Qt.MouseButton.LeftButton,
                                                                  Qt.MouseButton.RightButton]:
            if event.buttons() & Qt.MouseButton.LeftButton:
                self._quick_select_at_position(pos, Qt.MouseButton.LeftButton)
            elif event.buttons() & Qt.MouseButton.RightButton:
                self._quick_select_at_position(pos, Qt.MouseButton.RightButton)
            self._update_quick_selection_overlay(pos)
        elif self.mode == "clone stamp" and (event.buttons() & Qt.MouseButton.LeftButton):
            self._clone_stamp_at_position(pos)
            self._update_clone_stamp_overlay(pos)
        elif self.dragging and self.selected_pixmap_item:
            delta = pos - self.drag_start
            self.selected_pixmap_item.moveBy(delta.x(), delta.y())
            self.drag_start = pos
            # Real-time update: refresh the detection composite while dragging
            self.update_detection_composite()
        else:
            if self.mode == "quick selection":
                self._update_quick_selection_overlay(pos)
            if self.mode == "clone stamp":
                self._update_clone_stamp_overlay(pos)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
        self._undo_state_saved_current_action = False
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        zoomInFactor = 1.25
        zoomOutFactor = 1 / zoomInFactor
        factor = zoomInFactor if event.angleDelta().y() > 0 else zoomOutFactor
        self.scale(factor, factor)

    def apply_contrast_and_sharpen(self, image):
        contrast_image = cv2.convertScaleAbs(image, alpha=1.3, beta=0)
        blurred = cv2.GaussianBlur(contrast_image, (0, 0), sigmaX=3)
        sharpened = cv2.addWeighted(contrast_image, 1.5, blurred, -0.5, 0)
        return sharpened

    # -------------------------------------------------------------------------
    # Object Selection using SAM
    # -------------------------------------------------------------------------

    def object_selection(self):
        if self.current_cv_image is None:
            print("No image loaded")
            return
        if not self.positive_points and not self.negative_points:
            print("No selection points provided")
            return
        with torch.no_grad():
            predictor = SAMModelProvider.get_predictor()
            predictor.set_image(self.sam_cv_image_rgb)
            points = []
            labels = []
            if self.positive_points:
                points.extend(self.positive_points)
                labels.extend([1] * len(self.positive_points))
            if self.negative_points:
                points.extend(self.negative_points)
                labels.extend([0] * len(self.negative_points))
            points_array = np.array(points)
            labels_array = np.array(labels)
            masks, scores, logits = predictor.predict(
                point_coords=points_array,
                point_labels=labels_array,
                multimask_output=True
            )
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
        mask_uint8 = (mask.astype(np.uint8)) * 255
        kernel = np.ones((5, 5), np.uint8)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        self.selection_mask = cv2.bitwise_or(self.selection_mask, mask_uint8)
        bridge_kernel = np.ones((25, 25), np.uint8)
        self.selection_mask = cv2.morphologyEx(self.selection_mask, cv2.MORPH_CLOSE, bridge_kernel)
        print("Merged prompt-based selection into union mask with bridging.")
        self.positive_points.clear()
        self.negative_points.clear()
        self.update_display()

        # Save the new state after object selection is applied
        if self.undo_callback is not None:
            self.undo_callback()

    # -------------------------------------------------------------------------
    # Quick Selection and Clone Stamp Helpers
    # -------------------------------------------------------------------------
    def _quick_select_at_position(self, pos, button):
        if not self.background_pixmap_item or self.image_shape is None:
            return
        offset = self.background_pixmap_item.pos()
        x = int(pos.x() - offset.x())
        y = int(pos.y() - offset.y())
        if x < 0 or y < 0 or x >= self.image_shape[1] or y >= self.image_shape[0]:
            return
        brush_radius = self.quick_select_brush_size
        Y, X = np.ogrid[:self.image_shape[0], :self.image_shape[1]]
        dist = np.sqrt((X - x)**2 + (Y - y)**2)
        mask_area = dist <= brush_radius
        if button == Qt.MouseButton.LeftButton:
            self.selection_mask[mask_area] = 255
        elif button == Qt.MouseButton.RightButton:
            self.selection_mask[mask_area] = 0
        self.update_display()

    def _update_quick_selection_overlay(self, pos):
        radius = self.quick_select_brush_size
        rect = QRectF(pos.x() - radius, pos.y() - radius, 2*radius, 2*radius)
        if not self.quick_selection_overlay:
            from PyQt6.QtWidgets import QGraphicsEllipseItem
            self.quick_selection_overlay = QGraphicsEllipseItem()
            pen = QPen(QColor("red"))
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setWidth(2)
            self.quick_selection_overlay.setPen(pen)
            self.quick_selection_overlay.setZValue(30)
            self.scene.addItem(self.quick_selection_overlay)
        self.quick_selection_overlay.setRect(rect)

    # -------------------------------------------------------------------------
    # Merge selected region into background
    # -------------------------------------------------------------------------
    def apply_merge(self):
        if self.selected_pixmap_item is None and (self.selection_mask is None or np.count_nonzero(self.selection_mask) == 0):
            print("No active selection mask to merge.")
            self.clear_selection_feedback_items()
            return
        composite_image = self.background_pixmap_item.pixmap().toImage()
        painter = QPainter(composite_image)
        if self.selected_pixmap_item:
            selected_pixmap = self.selected_pixmap_item.pixmap()
            pos = self.selected_pixmap_item.pos()
            painter.drawPixmap(int(pos.x()), int(pos.y()), selected_pixmap)
        painter.end()
        merged_pixmap = QPixmap.fromImage(composite_image)
        self.background_pixmap_item.setPixmap(merged_pixmap)
        self.remove_item_safely(self.selected_pixmap_item)
        self.selected_pixmap_item = None
        self.clear_selection_feedback_items()
        self.selection_mask = np.zeros(self.image_shape, dtype=np.uint8)
        print("Merge applied: selection merged into current image.")
        buffer = QBuffer()
        buffer.open(QIODevice.OpenModeFlag.ReadWrite)
        merged_pixmap.save(buffer, "PNG")
        arr = np.frombuffer(buffer.data(), np.uint8)
        self.current_cv_image = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        buffer.close()
        self.update_all_cv_image_conversions()
        self.detection_cv_image = self.current_cv_image.copy()

    def _clone_stamp_at_position(self, pos):
        if not self.background_pixmap_item or self.image_shape is None or self.clone_offset is None:
            return
        offset = self.background_pixmap_item.pos()
        dest_x = int(pos.x() - offset.x())
        dest_y = int(pos.y() - offset.y())
        brush_radius = self.clone_stamp_brush_size
        src_x = dest_x + self.clone_offset[0]
        src_y = dest_y + self.clone_offset[1]
        img_h, img_w = self.image_shape
        half_patch = brush_radius
        src_x1 = max(src_x - half_patch, 0)
        src_y1 = max(src_y - half_patch, 0)
        src_x2 = min(src_x + half_patch, img_w)
        src_y2 = min(src_y + half_patch, img_h)
        patch = self.current_cv_image[src_y1:src_y2, src_x1:src_x2].copy()
        dest_x1 = max(dest_x - half_patch, 0)
        dest_y1 = max(dest_y - half_patch, 0)
        dest_x2 = min(dest_x + half_patch, img_w)
        dest_y2 = min(dest_y + half_patch, img_h)
        patch_h, patch_w = patch.shape[:2]
        dest_h = dest_y2 - dest_y1
        dest_w = dest_x2 - dest_x1
        h = min(patch_h, dest_h)
        w = min(patch_w, dest_w)
        if h > 0 and w > 0:
            self.current_cv_image[dest_y1:dest_y1 + h, dest_x1:dest_x1 + w] = patch[:h, :w]
            self.update_all_cv_image_conversions()
            h_img, w_img, ch = self.display_cv_image.shape
            bytes_per_line = ch * w_img
            qimage = QImage(self.display_cv_image.data, w_img, h_img, bytes_per_line, QImage.Format.Format_RGBA8888)
            pixmap = QPixmap.fromImage(qimage)
            self.background_pixmap_item.setPixmap(pixmap)

    def _update_clone_stamp_overlay(self, pos):
        radius = self.clone_stamp_brush_size
        rect = QRectF(pos.x() - radius, pos.y() - radius, 2*radius, 2*radius)
        if not self.clone_stamp_overlay:
            from PyQt6.QtWidgets import QGraphicsEllipseItem
            self.clone_stamp_overlay = QGraphicsEllipseItem()
            pen = QPen(QColor("green"))
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setWidth(2)
            self.clone_stamp_overlay.setPen(pen)
            self.clone_stamp_overlay.setZValue(30)
            self.scene.addItem(self.clone_stamp_overlay)
        self.clone_stamp_overlay.setRect(rect)

    def _update_clone_source_overlay(self, pos):
        radius = self.clone_stamp_brush_size
        rect = QRectF(pos.x() - radius, pos.y() - radius, 2*radius, 2*radius)
        if not self.clone_source_overlay:
            from PyQt6.QtWidgets import QGraphicsEllipseItem
            self.clone_source_overlay = QGraphicsEllipseItem()
            pen = QPen(QColor("blue"))
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setWidth(2)
            self.clone_source_overlay.setPen(pen)
            self.clone_source_overlay.setZValue(30)
            self.scene.addItem(self.clone_source_overlay)
        self.clone_source_overlay.setRect(rect)
