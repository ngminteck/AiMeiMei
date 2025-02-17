import sys
import math
from PyQt6.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QVBoxLayout, QWidget
from PyQt6.QtGui import QPixmap, QPen, QColor, QTransform, QPainter
from PyQt6.QtCore import Qt, QRectF, QPointF

class CustomGraphicsView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.main_pixmap_item = None
        self.original_pixmap = None
        self.start_pos = None
        self.current_rect = None
        self.selection_rect = None
        self.selected_pixmap_item = None
        self.dragging = False
        self.rotating = False
        self.last_angle = 0
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

    def load_image(self, image_path):
        self.original_pixmap = QPixmap(image_path)
        self.main_pixmap_item = QGraphicsPixmapItem(self.original_pixmap)
        self.scene.addItem(self.main_pixmap_item)
        self.setSceneRect(self.main_pixmap_item.boundingRect())

    def mousePressEvent(self, event):
        pos = self.mapToScene(int(event.position().x()), int(event.position().y()))
        if event.button() == Qt.MouseButton.LeftButton:
            if self.selected_pixmap_item:
                self.dragging = True
                self.drag_start = pos
            else:
                self.start_pos = pos
                self.current_rect = None
        elif event.button() == Qt.MouseButton.RightButton and self.selected_pixmap_item:
            self.rotating = True
            self.rotation_center = self.selected_pixmap_item.sceneBoundingRect().center()
            self.last_angle = self.angle_between_points(self.rotation_center, pos)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        pos = self.mapToScene(int(event.position().x()), int(event.position().y()))
        if self.dragging and self.selected_pixmap_item:
            delta = pos - self.drag_start
            self.selected_pixmap_item.moveBy(delta.x(), delta.y())
            self.drag_start = pos
        elif self.rotating and self.selected_pixmap_item:
            new_angle = self.angle_between_points(self.rotation_center, pos)
            delta_angle = new_angle - self.last_angle
            self.selected_pixmap_item.setRotation(self.selected_pixmap_item.rotation() + delta_angle)
            self.last_angle = new_angle
        elif self.start_pos and not self.selected_pixmap_item:
            self.current_rect = QRectF(self.start_pos, pos).normalized()
            if self.selection_rect:
                self.scene.removeItem(self.selection_rect)
            self.selection_rect = self.scene.addRect(self.current_rect, QPen(QColor(255, 0, 0), 2, Qt.PenStyle.DashLine))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
            if self.current_rect and not self.selected_pixmap_item:
                self.cut_and_create_selected_pixmap()
            self.start_pos = None
        elif event.button() == Qt.MouseButton.RightButton:
            self.rotating = False
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.reset_image()
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event):
        if self.selected_pixmap_item:
            scale_factor = 1.1 if event.angleDelta().y() > 0 else 0.9
            self.selected_pixmap_item.setScale(self.selected_pixmap_item.scale() * scale_factor)
        super().wheelEvent(event)

    def cut_and_create_selected_pixmap(self):
        if self.current_rect:
            rect = self.current_rect.toRect()
            selected_pixmap = self.original_pixmap.copy(rect)
            
            # Create the selected pixmap item
            if self.selected_pixmap_item:
                self.scene.removeItem(self.selected_pixmap_item)
            self.selected_pixmap_item = QGraphicsPixmapItem(selected_pixmap)
            self.selected_pixmap_item.setPos(self.current_rect.topLeft())
            self.scene.addItem(self.selected_pixmap_item)
            
            # Remove the selection rectangle
            if self.selection_rect:
                self.scene.removeItem(self.selection_rect)
                self.selection_rect = None

            # Create a new pixmap for the main image with a transparent hole
            new_main_pixmap = QPixmap(self.original_pixmap.size())
            new_main_pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(new_main_pixmap)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
            painter.drawPixmap(0, 0, self.original_pixmap)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.fillRect(rect, Qt.GlobalColor.transparent)
            painter.end()
            
            # Update the main pixmap item
            self.main_pixmap_item.setPixmap(new_main_pixmap)

    def reset_image(self):
        if self.original_pixmap:
            self.main_pixmap_item.setPixmap(self.original_pixmap)
            if self.selected_pixmap_item:
                self.scene.removeItem(self.selected_pixmap_item)
            if self.selection_rect:
                self.scene.removeItem(self.selection_rect)
            self.selected_pixmap_item = None
            self.selection_rect = None
            self.current_rect = None
            self.dragging = False
            self.rotating = False

    def angle_between_points(self, center, point):
        delta = point - center
        return math.degrees(math.atan2(delta.y(), delta.x()))

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.view = CustomGraphicsView()
        layout.addWidget(self.view)

        self.setLayout(layout)
        self.setWindowTitle("Image Selection Tool")
        self.resize(800, 600)

        # Load the image with the provided path
        self.view.load_image("C:/Users/Makarand.Patwardhan/OneDrive - AddSecure/Pictures/book.jpg")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
