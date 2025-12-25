import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QFileDialog, QWidget, QGridLayout, QSizePolicy, QGroupBox, QScrollArea,QSlider
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import Qt
from skimage.metrics import structural_similarity as ssim
from math import log10, sqrt

class SplashScreen:
    def __init__(self, image_path: str, display_time: int = 3000):
        self.image_path = image_path
        self.display_time = display_time

    def show(self):
        # Load the image
        image = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Image not found: {self.image_path}")

        # Resize image for a better display (optional)
        screen_width, screen_height = 300,300
        image = cv2.resize(image, (screen_width, screen_height), interpolation=cv2.INTER_AREA)

        cv2.namedWindow("Loading...", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Loading...", screen_width, screen_height)

        cv2.imshow("Loading...", image)
        cv2.waitKey(self.display_time)  # Display for the specified time
        cv2.destroyWindow("Loading...")


class ImageEnhancementApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Enhancement App")
        self.setWindowIcon(QIcon("mpicon.jpeg"))
        self.setGeometry(100, 100, 1200, 700)  # Enlarged window size

        # Initialize variables
        self.original_image = None
        self.enhanced_image = None
        self.file_path = None
        self.saved_image_path = None
        self.thumbnail_counter = 0
        self.thumbnail_size = (300, 300)
        self.setMouseTracking(True)
        self.annotation = None 
        
        # Setup UI
        self.initUI()

    def initUI(self):
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QGridLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Define fonts
        label_font = QFont("Arial", 12, QFont.Bold)
        button_font = QFont("Arial", 10, QFont.Bold)
        button_font1 = QFont("Arial",7,QFont.Bold)

        # Create the left sidebar layout for buttons
        sidebar_layout = QVBoxLayout()
        
        # Create a group box for the Image Operations section
        image_operations_groupbox = QGroupBox("Image Operations")
        image_operations_groupbox.setFont(label_font)
        image_operations_groupbox.setAlignment(Qt.AlignCenter)
        image_operations_layout = QVBoxLayout()
        image_operations_groupbox.setLayout(image_operations_layout)
        image_operations_groupbox.setStyleSheet("QGroupBox {"
                                         "border: 2px solid #2c3e50;"  # Thicker border (2px)
                                         "border-radius: 5px;"          # Optional: rounded corners
                                         "padding: 10px;"               # Space between the border and content
                                         "font-weight: bold;"           # Bold title text (optional)
                                         "}")

        # Image operation buttons inside the image_operations_groupbox
        self.exit_button = QPushButton("Exit")
        self.exit_button.setFont(button_font)
        self.exit_button.setStyleSheet("background-color: #e74c3c; padding: 10px;")
        self.exit_button.clicked.connect(self.close)
        image_operations_layout.addWidget(self.exit_button)

        self.open_image_button = QPushButton("Load Image")
        self.open_image_button.setFont(button_font)
        self.open_image_button.setStyleSheet("background-color: #1abc9c; padding: 10px;")
        self.open_image_button.clicked.connect(self.load_image)
        image_operations_layout.addWidget(self.open_image_button)

        self.reset_button = QPushButton("Reset")
        self.reset_button.setFont(button_font)
        self.reset_button.setStyleSheet("background-color: #1abc9c; padding: 10px;")
        self.reset_button.clicked.connect(self.reset_image)
        image_operations_layout.addWidget(self.reset_button)

        self.save_button = QPushButton("Save")
        self.save_button.setFont(button_font)
        self.save_button.setStyleSheet("background-color: #27ae60; padding: 10px;")
        self.save_button.clicked.connect(self.save_image)
        image_operations_layout.addWidget(self.save_button)

        self.save_as_button = QPushButton("Save As")
        self.save_as_button.setFont(button_font)
        self.save_as_button.setStyleSheet("background-color: #27ae60; padding: 10px;")
        self.save_as_button.clicked.connect(self.save_image_as)
        image_operations_layout.addWidget(self.save_as_button)
        
        self.thumbnail_button = QPushButton("Image Viewer")
        self.thumbnail_button.setFont(button_font)
        self.thumbnail_button.setStyleSheet("background-color: #27ae60; padding: 10px;")
        self.thumbnail_button.clicked.connect(self.create_imageviewer)
        image_operations_layout.addWidget(self.thumbnail_button)

        # Add the Image Operations group box to the sidebar layout
        sidebar_layout.addWidget(image_operations_groupbox)

        preprocessing_groupbox = QGroupBox("Preprocessing")
        preprocessing_groupbox.setFont(label_font)
        preprocessing_groupbox.setAlignment(Qt.AlignCenter)
        preprocessing_layout = QVBoxLayout()
        preprocessing_groupbox.setLayout(preprocessing_layout)

        # Thicken the border line of the group box
        preprocessing_groupbox.setStyleSheet("QGroupBox {"
                                            "border: 2px solid #2c3e50;"  # Thicker border (2px)
                                            "border-radius: 5px;"          # Optional: rounded corners
                                            "padding: 10px;"               # Space between the border and content
                                            "font-weight: bold;"           # Bold title text (optional)
                                            "}")

        # Preprocessing buttons
        self.noise_button = QPushButton("Noise Removal")
        self.noise_button.setFont(button_font1)
        self.noise_button.setStyleSheet("background-color: #f7b731; padding: 10px;")
        self.noise_button.clicked.connect(self.noise_removal)
        self.noise_slider = QSlider(Qt.Horizontal)
        self.noise_slider.setMinimum(1)
        self.noise_slider.setMaximum(15)
        self.noise_slider.setValue(0)
        self.noise_slider.setTickInterval(2)
        self.noise_slider.setTickPosition(QSlider.TicksBelow)
        self.noise_slider.setFixedHeight(10)
        preprocessing_layout.addWidget(self.noise_button)
        preprocessing_layout.addWidget(QLabel("Noise Reduction Level"))
        preprocessing_layout.addWidget(self.noise_slider)
        self.noise_slider.valueChanged.connect(self.update_noise_removal)

        self.sharp_button = QPushButton("Sharpness")
        self.sharp_button.setFont(button_font1)
        self.sharp_button.setStyleSheet("background-color: #f7b731; padding: 10px;")
        self.sharp_button.clicked.connect(self.apply_sharpening)
        self.sharp_slider = QSlider(Qt.Horizontal)
        self.sharp_slider.setMinimum(0)
        self.sharp_slider.setMaximum(100)
        self.sharp_slider.setValue(0)
        self.sharp_slider.setTickInterval(10)
        self.sharp_slider.setTickPosition(QSlider.TicksBelow)
        self.sharp_slider.setFixedHeight(10)
        preprocessing_layout.addWidget(self.sharp_button)
        preprocessing_layout.addWidget(QLabel("Sharpness Level"))
        preprocessing_layout.addWidget(self.sharp_slider)
        self.sharp_slider.valueChanged.connect(self.update_sharpness)

        self.he_button = QPushButton("Histogram Equalization")
        self.he_button.setFont(button_font1)
        self.he_button.setStyleSheet("background-color: #f7b731; padding: 10px;")
        self.he_button.clicked.connect(self.histogram_equalization)
        preprocessing_layout.addWidget(self.he_button)
        self.he_slider = QSlider(Qt.Horizontal)
        self.he_slider.setMinimum(0)
        self.he_slider.setMaximum(100)
        self.he_slider.setValue(0)
        self.he_slider.setTickInterval(10)
        self.he_slider.setTickPosition(QSlider.TicksBelow)
        self.he_slider.setFixedHeight(10)
        preprocessing_layout.addWidget(QLabel("Histogram Equalization Level"))
        preprocessing_layout.addWidget(self.he_slider)
        self.he_slider.valueChanged.connect(self.update_histogram_equalization)
        # Add the Preprocessing group box to the sidebar layout
        sidebar_layout.addWidget(preprocessing_groupbox)
        

        enhancement_groupbox = QGroupBox("Enhancement")
        enhancement_groupbox.setFont(label_font)
        enhancement_groupbox.setAlignment(Qt.AlignCenter)
        enhancement_layout = QVBoxLayout()
        enhancement_groupbox.setLayout(enhancement_layout)
        # Thicken the border line of the group box
        enhancement_groupbox.setStyleSheet("QGroupBox {"
                                            "border: 2px solid #2c3e50;"  # Thicker border (2px)
                                            "border-radius: 5px;"          # Optional: rounded corners
                                            "padding: 10px;"               # Space between the border and content
                                            "font-weight: bold;"           # Bold title text (optional)
                                            "}")

        # Add Enhancement buttons
        self.dcp_button = QPushButton("Dark Channel Prior(DCP)")
        self.dcp_button.setFont(button_font)
        self.dcp_button.setStyleSheet("background-color: #f39c12; padding: 10px;")
        self.dcp_button.clicked.connect(self.apply_dcp)
        enhancement_layout.addWidget(self.dcp_button)

        self.hdr_button = QPushButton("High Dynamic Range(HDR)")
        self.hdr_button.setFont(button_font)
        self.hdr_button.setStyleSheet("background-color: #f39c12; padding: 10px;")
        self.hdr_button.clicked.connect(self.apply_hdr)
        enhancement_layout.addWidget(self.hdr_button)

        self.combine_button = QPushButton("Combine")
        self.combine_button.setFont(button_font)
        self.combine_button.setStyleSheet("background-color: #f39c12; padding: 10px;")
        self.combine_button.clicked.connect(self.combine_methods)
        enhancement_layout.addWidget(self.combine_button)

        # Add the Enhancement group box to the sidebar layout
        sidebar_layout.addWidget(enhancement_groupbox)

        # Transformation and Reflect Group Box
        transform_groupbox = QGroupBox("Tools")
        transform_groupbox.setFont(label_font)
        transform_groupbox.setAlignment(Qt.AlignCenter)
        transform_layout = QVBoxLayout()
        transform_groupbox.setLayout(transform_layout)
        
        transform_groupbox.setStyleSheet("QGroupBox {"
                                            "border: 2px solid #2c3e50;"  # Thicker border (2px)
                                            "border-radius: 5px;"          # Optional: rounded corners
                                            "padding: 10px;"               # Space between the border and content
                                            "font-weight: bold;"           # Bold title text (optional)
                                            "}")

        # Add Transform and Reflect buttons
        self.transform_button = QPushButton("Transform Image")
        self.transform_button.setFont(button_font)
        self.transform_button.setStyleSheet("background-color: #3498db; padding: 10px;")
        self.transform_button.clicked.connect(self.apply_transformation)
        transform_layout.addWidget(self.transform_button)

        self.reflect_button = QPushButton("Reflect Image")
        self.reflect_button.setFont(button_font)
        self.reflect_button.setStyleSheet("background-color: #3498db; padding: 10px;")
        self.reflect_button.clicked.connect(self.reflect_image)
        transform_layout.addWidget(self.reflect_button)
        
        self.zoom_in_button = QPushButton("Expand Image")
        self.zoom_in_button.setFont(button_font)
        self.zoom_in_button.setStyleSheet("background-color: #3498db; padding: 10px;")
        self.zoom_in_button.clicked.connect(self.expand_image)
        transform_layout.addWidget(self.zoom_in_button)

        self.zoom_out_button = QPushButton("Compress Image")
        self.zoom_out_button.setFont(button_font)
        self.zoom_out_button.setStyleSheet("background-color: #3498db; padding: 10px;")
        self.zoom_out_button.clicked.connect(self.compress_image)
        transform_layout.addWidget(self.zoom_out_button)

        # Add the Transformation group box to the sidebar layout
        sidebar_layout.addWidget(transform_groupbox)
        
        # Main layout: Grid with sidebar (left) and content (right)
        sidebar_widget = QWidget()
        sidebar_widget.setLayout(sidebar_layout)
        sidebar_widget.setFixedWidth(250)
        main_layout.addWidget(sidebar_widget, 0, 0, 6, 1)  # Sidebar on the left, spanning 6 rows

        # Right side layout for the main content
        content_layout = QGridLayout()
        main_layout.addLayout(content_layout, 0, 1, 6, 2)  # Content on the right

        self.original_title_label = QLabel("Original Image")
        self.original_title_label.setAlignment(Qt.AlignCenter)
        self.original_title_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.original_title_label.setStyleSheet("background-color: lightgray; border: 1px solid #ccc; padding: 5px;")
        content_layout.addWidget(self.original_title_label, 0, 0)  # Title at (0,0)

        # Original image section
        self.original_label = QLabel("Empty, Please Load An Image")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setStyleSheet("background-color: lightgray; border: 1px solid #ccc;")
        self.original_label.setFont(label_font)
        self.original_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Allow resizing
        content_layout.addWidget(self.original_label, 1, 0)

        self.enhanced_title_label = QLabel("Enhanced Image")
        self.enhanced_title_label.setAlignment(Qt.AlignCenter)
        self.enhanced_title_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.enhanced_title_label.setStyleSheet("background-color: lightgray; border: 1px solid #ccc; padding: 5px;")
        content_layout.addWidget(self.enhanced_title_label, 0, 1)  # Title at (0,1)

        # Enhanced image section
        self.enhanced_label = QLabel("Empty")
        self.enhanced_label.setAlignment(Qt.AlignCenter)
        self.enhanced_label.setStyleSheet("background-color: lightgray; border: 1px solid #ccc;")
        self.enhanced_label.setFont(label_font)
        self.enhanced_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Allow resizing
        content_layout.addWidget(self.enhanced_label, 1, 1)

        # Histogram sections
        self.original_hist_label = QLabel("Original Histogram Chart")
        self.original_hist_label.setAlignment(Qt.AlignCenter)
        self.original_hist_label.setStyleSheet("background-color: white; border: 1px solid #ccc;")
        self.original_hist_label.setFont(label_font)
        self.original_hist_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Allow resizing
        content_layout.addWidget(self.original_hist_label, 2, 0)
        
        self.enhanced_hist_label = QLabel("Enhanced Histogram Chart")
        self.enhanced_hist_label.setAlignment(Qt.AlignCenter)
        self.enhanced_hist_label.setStyleSheet("background-color: white; border: 1px solid #ccc;")
        self.enhanced_hist_label.setFont(label_font)
        self.enhanced_hist_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Allow resizing
        content_layout.addWidget(self.enhanced_hist_label, 2, 1)
                
        # Metrics section
        self.ssim_label_1 = QLabel("SSIM: ")
        self.ssim_label_1.setStyleSheet("background-color: lightblue; border: 1px solid #ccc; padding: 5px;")
        self.ssim_label_1.setFont(label_font)
        content_layout.addWidget(self.ssim_label_1, 3, 0)

        self.psnr_label_1 = QLabel("PSNR: ")
        self.psnr_label_1.setStyleSheet("background-color: lightblue; border: 1px solid #ccc; padding: 5px;")
        self.psnr_label_1.setFont(label_font)
        content_layout.addWidget(self.psnr_label_1, 4, 0)

        self.ssim_label = QLabel("SSIM: ")
        self.ssim_label.setStyleSheet("background-color: lightblue; border: 1px solid #ccc; padding: 5px;")
        self.ssim_label.setFont(label_font)
        content_layout.addWidget(self.ssim_label, 3, 1)

        self.psnr_label = QLabel("PSNR: ")
        self.psnr_label.setStyleSheet("background-color: lightblue; border: 1px solid #ccc; padding: 5px;")
        self.psnr_label.setFont(label_font)
        content_layout.addWidget(self.psnr_label, 4, 1)

        self.original_properties_scroll_area = QScrollArea()
        self.original_properties_scroll_area.setWidgetResizable(True)
        self.original_properties_scroll_area.setFixedHeight(100)  # Set fixed height for the scroll area
        self.original_properties_scroll_area.setStyleSheet("border: none;")  # Optional: Remove outer border

        self.original_properties_label = QLabel("Original Image Properties:")
        self.original_properties_label.setAlignment(Qt.AlignTop)
        self.original_properties_label.setStyleSheet("background-color: lightgreen; border: 1px solid #ccc; padding: 5px;")
        self.original_properties_label.setFont(label_font)
        self.original_properties_label.setWordWrap(True)  # Ensure text wraps within the label
        self.original_properties_scroll_area.setWidget(self.original_properties_label)  # Add the label to the scroll area
        content_layout.addWidget(self.original_properties_scroll_area, 6, 0)

        # Scroll Area for Enhanced Properties Label
        self.enhanced_properties_scroll_area = QScrollArea()
        self.enhanced_properties_scroll_area.setWidgetResizable(True)
        self.enhanced_properties_scroll_area.setFixedHeight(100)  # Set fixed height for the scroll area
        self.enhanced_properties_scroll_area.setStyleSheet("border: none;")  # Optional: Remove outer border

        self.enhanced_properties_label = QLabel("Enhanced Image Properties:")
        self.enhanced_properties_label.setAlignment(Qt.AlignTop)
        self.enhanced_properties_label.setStyleSheet("background-color: lightgreen; border: 1px solid #ccc; padding: 5px;")
        self.enhanced_properties_label.setFont(label_font)
        self.enhanced_properties_label.setWordWrap(True)  # Ensure text wraps within the label
        self.enhanced_properties_scroll_area.setWidget(self.enhanced_properties_label)  # Add the label to the scroll area
        content_layout.addWidget(self.enhanced_properties_scroll_area, 6, 1)
        
    def create_imageviewer(self):
        if self.enhanced_image is not None:
            # Increment the counter for the new thumbnail
            self.thumbnail_counter += 1

            # Get original dimensions of the image
            original_height, original_width = self.enhanced_image.shape[:2]

            # Set the window name
            window_name = f"Image Viewer {self.thumbnail_counter}"

            # Create a resizable window
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            # Display the original image
            cv2.imshow(window_name, self.enhanced_image)

            # Adjust the window size to match the image dimensions
            cv2.resizeWindow(window_name, original_width, original_height)

            # Wait for user input to close the window
            key = cv2.waitKey(0)

            # Check if the window is still open before destroying it
            try:
                cv2.destroyWindow(window_name)
            except cv2.error as e:
                print(f"Error destroying window: {e}")
        
    def load_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)", options=options)
        if file_path:
            self.file_path = file_path
            self.original_image = cv2.imread(file_path)
            self.enhanced_image = self.original_image.copy()

            # Display the images
            self.display_image(self.original_image, self.original_label)
            self.display_image(self.enhanced_image, self.enhanced_label)

            self.show_histogram(self.original_image, self.original_hist_label)
            self.show_histogram(self.enhanced_image, self.enhanced_hist_label)

            # Calculate SSIM and PSNR
            ssim_value_1 = self.calculate_ssim(self.original_image, self.original_image)
            psnr_value_1 = self.calculate_psnr(self.original_image, self.original_image)

            # Update the SSIM and PSNR labels
            self.ssim_label_1.setText(f"SSIM: {ssim_value_1:.4f}")
            self.psnr_label_1.setText(f"PSNR: {psnr_value_1:.4f} dB")
            
            # Calculate SSIM and PSNR 
            ssim_value = self.calculate_ssim(self.original_image, self.enhanced_image)
            psnr_value = self.calculate_psnr(self.original_image, self.enhanced_image)

            # Update the SSIM and PSNR labels with new values
            self.ssim_label.setText(f"SSIM: {ssim_value:.4f}")
            self.psnr_label.setText(f"PSNR: {psnr_value:.4f} dB")

            # Display image properties with titles
            self.display_image_properties(self.original_image, self.original_properties_label, "Original Image Properties :")
            self.display_image_properties(self.enhanced_image, self.enhanced_properties_label, "Enhanced Image Properties :")

    def display_image_properties(self, image, label, title):
        # Calculate pixel intensity for each channel
        blue_channel = image[:, :, 0]
        green_channel = image[:, :, 1]
        red_channel = image[:, :, 2]

        # Calculate average pixel value for each channel
        avg_blue = np.mean(blue_channel)
        avg_green = np.mean(green_channel)
        avg_red = np.mean(red_channel)

        height, width, channels = image.shape
        file_size = sys.getsizeof(image)  # In bytes
        
        label.setText(
            f"{title}\n\n"
            f"Dimensions: {width}x{height}\n"
            f"Channels: {channels}\n"
            f"Size: {file_size / 1024:.2f} KB\n"
            f"Average Blue pixel value: {avg_blue:.2f}\n"
            f"Average Green pixel value: {avg_green:.2f}\n"
            f"Average Red pixel value: {avg_red:.2f}"
        )
        
    
    def expand_image(self):
        """Zoom in on the enhanced image."""
        if self.enhanced_image is not None:
            # Scale up the image by 1.2x
            height, width = self.enhanced_image.shape[:2]
            new_size = (int(width * 1.2), int(height * 1.2))
            zoomed_image = cv2.resize(self.enhanced_image, new_size, interpolation=cv2.INTER_LINEAR)

            # Display the zoomed-in image
            self.enhanced_image = zoomed_image  # Update the enhanced image
            self.display_image(self.enhanced_image, self.enhanced_label)
            self.display_image_properties(self.enhanced_image, self.enhanced_properties_label, "Enhanced Image Properties:")


    def compress_image(self):
        """Zoom out of the enhanced image."""
        if self.enhanced_image is not None:
            # Scale down the image by 0.8x
            height, width = self.enhanced_image.shape[:2]
            new_size = (int(width * 0.8), int(height * 0.8))
            zoomed_image = cv2.resize(self.enhanced_image, new_size, interpolation=cv2.INTER_LINEAR)

            # Display the zoomed-out image
            self.enhanced_image = zoomed_image  # Update the enhanced image
            self.display_image(self.enhanced_image, self.enhanced_label)
            self.display_image_properties(self.enhanced_image, self.enhanced_properties_label, "Enhanced Image Properties:")
         
    def apply_transformation(self):
        """Reflect left-right, up-down, and rotate right-up."""
        if self.enhanced_image is not None:
            # Apply transformations in sequence:
            # 1. Reflect left-right
            transformed_image = cv2.flip(self.enhanced_image, 1)  # 1 means flipping horizontally
            # 2. Reflect up-down
            transformed_image = cv2.flip(transformed_image, 0)  # 0 means flipping vertically
            # 3. Rotate 90 degrees right
            transformed_image = cv2.rotate(transformed_image, cv2.ROTATE_90_CLOCKWISE)

            # Display the transformed image
            self.enhanced_image = transformed_image  # Update the enhanced image
            self.display_image(self.enhanced_image, self.enhanced_label)
            self.display_image_properties(self.enhanced_image, self.enhanced_properties_label, "Enhanced Image Properties :")


    def reflect_image(self):
        """Reflect the image along both x-axis and y-axis."""
        if self.enhanced_image is not None:
            # Reflect the enhanced image along both axes
            reflected_image = cv2.flip(self.enhanced_image, 1)  # -1 means both horizontal and vertical flip

            # Display the reflected image
            self.enhanced_image = reflected_image  # Update the enhanced image
            self.display_image(self.enhanced_image, self.enhanced_label)
            self.display_image_properties(self.enhanced_image, self.enhanced_properties_label, "Enhanced Image Properties :")

            
    def display_image(self, image, label):
        # Convert the OpenCV image (BGR) to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create a Matplotlib figure and axis for displaying the image
        fig, ax = plt.subplots(figsize=(label.width() / 3, label.height() / 3), dpi=30)  # Use label size
        fig.patch.set_facecolor('lightgray')
        ax.axis('off')  # Hide the axis to just show the image

        # Display the image using imshow
        ax.imshow(rgb_image)

        # Create a FigureCanvas for embedding the figure into the label
        canvas = FigureCanvas(fig)

        # Ensure the label has a layout and clear any old widgets
        label.setLayout(QVBoxLayout())  # Ensure the label has a layout
        layout = label.layout()

        # Remove any existing widgets from the label layout
        for i in reversed(range(layout.count())):
            layout.itemAt(i).widget().deleteLater()

        layout.addWidget(canvas)

        # Resize the canvas to fit the label size while maintaining the aspect ratio
        canvas_width, canvas_height = label.width(), label.height()
        aspect_ratio = rgb_image.shape[1] / rgb_image.shape[0]

        if canvas_width / canvas_height > aspect_ratio:
            new_width = canvas_height * aspect_ratio
            new_height = canvas_height
        else:
            new_width = canvas_width
            new_height = canvas_width / aspect_ratio

        # Convert the dimensions to integers before resizing
        canvas.resize(int(new_width), int(new_height))
        canvas.draw()
        
    def save_image(self):
        if self.enhanced_image is not None:
            # If no new file was chosen, save to the original file if it exists
            if self.saved_image_path is not None:
                # Save the enhanced image to the last saved path
                cv2.imwrite(self.saved_image_path, self.enhanced_image)
            elif self.file_path is not None:
                # If no new save path was provided, save to the original file path
                cv2.imwrite(self.file_path, self.enhanced_image)

    def save_image_as(self):
        if self.enhanced_image is not None:
            # Open a file dialog to choose a location and filename
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Image As", "", "Image Files (*.png *.jpg *.bmp)", options=options)
            if file_path:
                # Save to the new file path and update saved image path
                cv2.imwrite(file_path, self.enhanced_image)
                self.saved_image_path = file_path  # Update the saved image path

    def reset_image(self):
        """Reset the enhanced image back to the original image."""
        if self.original_image is not None:
            self.enhanced_image = self.original_image.copy()
        
            self.display_image(self.enhanced_image, self.enhanced_label)
            self.show_histogram(self.enhanced_image, self.enhanced_hist_label)
            self.display_image_properties(self.enhanced_image, self.enhanced_properties_label, "Enhanced Image Properties :")

            # Calculate SSIM and PSNR 
            ssim_value = self.calculate_ssim(self.original_image, self.enhanced_image)
            psnr_value = self.calculate_psnr(self.original_image, self.enhanced_image)

            # Update the SSIM and PSNR labels with new values
            self.ssim_label.setText(f"SSIM: {ssim_value:.4f}")
            self.psnr_label.setText(f"PSNR: {psnr_value:.4f} dB")

    def update_noise_removal(self, kernel_size):
        if kernel_size % 2 == 0:  # Ensure kernel size is always odd for median blur
            kernel_size += 1

        if self.enhanced_image is not None:
            small_image = cv2.resize(self.enhanced_image, 
                                    (self.enhanced_image.shape[1] // 2, 
                                    self.enhanced_image.shape[0] // 2))

            denoised_image = cv2.medianBlur(small_image, kernel_size)

            denoised_image = cv2.resize(denoised_image, 
                                        (self.enhanced_image.shape[1], 
                                        self.enhanced_image.shape[0]))

            # Update and display
            self.enhanced_image = denoised_image
            self.display_image(self.enhanced_image, self.enhanced_label)
            self.show_histogram(self.enhanced_image, self.enhanced_hist_label)
            self.display_image_properties(self.enhanced_image, self.enhanced_properties_label, "Enhanced Image Properties :")

            # Calculate and update SSIM and PSNR
            ssim_value = self.calculate_ssim(self.original_image, self.enhanced_image)
            psnr_value = self.calculate_psnr(self.original_image, self.enhanced_image)
            self.ssim_label.setText(f"SSIM: {ssim_value:.4f}")
            self.psnr_label.setText(f"PSNR: {psnr_value:.4f} dB")

    def noise_removal(self):
        if self.enhanced_image is not None:
            # Resize the image to a smaller size for faster processing
            small_image = cv2.resize(self.enhanced_image, (self.enhanced_image.shape[1] // 2, self.enhanced_image.shape[0] // 2))

            # Apply noise removal (e.g., GaussianBlur or MedianBlur)
            denoised_image = cv2.medianBlur(small_image, 5)

            # Resize the denoised image back to the original size
            denoised_image = cv2.resize(denoised_image, (self.enhanced_image.shape[1], self.enhanced_image.shape[0]))

            # Update the enhanced image
            self.enhanced_image = denoised_image
            
            # Display the denoised image and histogram
            self.display_image(self.enhanced_image, self.enhanced_label)
            self.show_histogram(self.enhanced_image, self.enhanced_hist_label)
            self.display_image_properties(self.enhanced_image, self.enhanced_properties_label, "Enhanced Image Properties :")

            # Calculate SSIM and PSNR after noise removal
            ssim_value = self.calculate_ssim(self.original_image, self.enhanced_image)
            psnr_value = self.calculate_psnr(self.original_image, self.enhanced_image)

            # Update the SSIM and PSNR labels with new values
            self.ssim_label.setText(f"SSIM: {ssim_value:.4f}")
            self.psnr_label.setText(f"PSNR: {psnr_value:.4f} dB")

    def update_sharpness(self, sharpness_value):
        if self.enhanced_image is not None:
            # Define sharpening kernel
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

            # Apply sharpening filter
            sharp_image = cv2.filter2D(self.enhanced_image, -1, kernel)

            # Blend between original and sharp image based on slider value
            alpha = sharpness_value / 100  # Normalize slider value to 0.0 - 1.0
            blended_image = cv2.addWeighted(self.enhanced_image, 1 - alpha, sharp_image, alpha, 0)

            # Update and display
            self.enhanced_image = blended_image
            self.display_image(self.enhanced_image, self.enhanced_label)
            self.show_histogram(self.enhanced_image, self.enhanced_hist_label)
            self.display_image_properties(self.enhanced_image, self.enhanced_properties_label, "Enhanced Image Properties :")

            # Calculate and update SSIM and PSNR
            ssim_value = self.calculate_ssim(self.original_image, self.enhanced_image)
            psnr_value = self.calculate_psnr(self.original_image, self.enhanced_image)
            self.ssim_label.setText(f"SSIM: {ssim_value:.4f}")
            self.psnr_label.setText(f"PSNR: {psnr_value:.4f} dB")

    def apply_sharpening(self):
        if self.enhanced_image is not None:
            # Sharpen the image using a kernel
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            self.enhanced_image = cv2.filter2D(self.enhanced_image, -1, kernel)
    
           # Display the enhanced image and histogram
            self.display_image(self.enhanced_image, self.enhanced_label)
            self.show_histogram(self.enhanced_image, self.enhanced_hist_label)
            self.display_image_properties(self.enhanced_image, self.enhanced_properties_label, "Enhanced Image Properties :")

            # Calculate SSIM and PSNR after sharpening
            ssim_value = self.calculate_ssim(self.original_image, self.enhanced_image)
            psnr_value = self.calculate_psnr(self.original_image, self.enhanced_image)

            # Update the SSIM and PSNR labels with new values
            self.ssim_label.setText(f"SSIM: {ssim_value:.4f}")
            self.psnr_label.setText(f"PSNR: {psnr_value:.4f} dB")

    def update_histogram_equalization(self, equalization_value):
        if self.enhanced_image is not None:
            # Convert the image to YCrCb (to apply equalization only on luminance)
            ycrcb = cv2.cvtColor(self.enhanced_image, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)

            # Apply histogram equalization on the Y channel
            equalized_y = cv2.equalizeHist(y)

            # Blend between the original and equalized luminance
            alpha = equalization_value / 100  # Normalize slider value to 0.0 - 1.0
            blended_y = cv2.addWeighted(y, 1 - alpha, equalized_y, alpha, 0)

            # Merge channels back and convert to BGR
            ycrcb_equalized = cv2.merge((blended_y, cr, cb))
            equalized_image = cv2.cvtColor(ycrcb_equalized, cv2.COLOR_YCrCb2BGR)

            # Update and display
            self.enhanced_image = equalized_image
            self.display_image(self.enhanced_image, self.enhanced_label)
            self.show_histogram(self.enhanced_image, self.enhanced_hist_label)
            self.display_image_properties(self.enhanced_image, self.enhanced_properties_label, "Enhanced Image Properties :")

            # Calculate and update SSIM and PSNR
            ssim_value = self.calculate_ssim(self.original_image, self.enhanced_image)
            psnr_value = self.calculate_psnr(self.original_image, self.enhanced_image)
            self.ssim_label.setText(f"SSIM: {ssim_value:.4f}")
            self.psnr_label.setText(f"PSNR: {psnr_value:.4f} dB")

    def histogram_equalization(self):
        if self.enhanced_image is not None:
            # Perform histogram equalization
            img_yuv = cv2.cvtColor(self.enhanced_image, cv2.COLOR_BGR2YUV)
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
            self.enhanced_image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
            
            # Display the enhanced image and histogram
            self.display_image(self.enhanced_image, self.enhanced_label)
            self.show_histogram(self.enhanced_image, self.enhanced_hist_label)
            self.display_image_properties(self.enhanced_image, self.enhanced_properties_label, "Enhanced Image Properties :")

            # Calculate SSIM and PSNR after enhancement
            ssim_value = self.calculate_ssim(self.original_image, self.enhanced_image)
            psnr_value = self.calculate_psnr(self.original_image, self.enhanced_image)

            # Update the SSIM and PSNR labels with new values
            self.ssim_label.setText(f"SSIM: {ssim_value:.4f}")
            self.psnr_label.setText(f"PSNR: {psnr_value:.4f} dB")

    def show_histogram(self, image, label):
        if image is not None:
            # Convert image to RGB if it's in BGR
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Create a Matplotlib figure
            fig = plt.figure(figsize=(6, 5))
            fig.patch.set_facecolor('white')
            ax = fig.add_subplot(111)
            
            colors = ('r', 'g', 'b')
            hist_data = []
            
            # Calculate histogram for each channel
            for i, col in enumerate(colors):
                hist = cv2.calcHist([image_rgb], [i], None, [256], [0, 256])
                ax.plot(hist, color=col, label=f'Channel {col.upper()}')
                hist_data.append(hist)

            # Set the labels and title for the plot
            ax.set_xlim([0, 256])  # Ensure x-axis is 0-256
            ax.set_title("Histogram")
            ax.set_xlabel("Pixel Value")
            ax.set_ylabel("Frequency")
            ax.legend(loc="upper right")
            plt.tight_layout()

            plt.tight_layout(pad=4.0)
            # Create a FigureCanvas to embed the plot into PyQt5 label
            canvas = FigureCanvas(fig)
            
            # Clear any old widgets in the label layout
            label.setLayout(QVBoxLayout())  # Ensure label has a layout
            layout = label.layout()
            for i in reversed(range(layout.count())):
                layout.itemAt(i).widget().deleteLater()

            # Add the new canvas widget with the histogram plot
            layout.addWidget(canvas)

            # Resize the canvas to fit the label size while maintaining the aspect ratio
            canvas_width, canvas_height = label.width(), label.height()
            aspect_ratio = fig.get_figwidth() / fig.get_figheight()
            
            if canvas_width / canvas_height > aspect_ratio:
                new_width = canvas_height * aspect_ratio
                new_height = canvas_height
            else:
                new_width = canvas_width
                new_height = canvas_width / aspect_ratio
            
            # Resize the canvas
            canvas.resize(int(new_width), int(new_height))
            canvas.draw()

            # Function to update the mouse position
            def on_mouse_move(event):
                # Remove the previous annotation if it exists
                if self.annotation:
                    if self.annotation in ax.texts:
                        self.annotation.remove()

                if event.inaxes == ax:  # Only update if the mouse is within the axes
                    x = int(event.xdata)  # Get the x (pixel value) in range 0 to 255
                    if x < 0 or x > 255:
                        return  # Ignore values outside the range of 0 to 255
                        
                    # Get the y values (frequency) for all three channels at the x position
                    y_r = hist_data[0][x][0]  # Red channel
                    y_g = hist_data[1][x][0]  # Green channel
                    y_b = hist_data[2][x][0]  # Blue channel

                    # Create the annotation text with (pixel value) and frequency for each channel
                    annotation_text = f'x: {x}  \nR: {int(y_r)}  \nG: {int(y_g)}  \nB: {int(y_b)}'

                    # Create the new annotation at the current position
                    self.annotation = ax.annotate(annotation_text, xy=(x, max(y_r, y_g, y_b)),
                                                  xytext=(x + 5, max(y_r, y_g, y_b) + 5),
                                                  arrowprops=dict(facecolor='black', arrowstyle='->'),
                                                  fontsize=8, color='black')
                    canvas.draw()

            # Connect the mouse move event to the function
            fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
            
    def apply_enhancement(self):
        if self.enhanced_image is not None:
            # Example enhancement: histogram equalization
            img_yuv = cv2.cvtColor(self.enhanced_image, cv2.COLOR_BGR2YUV)
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
            self.enhanced_image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
           
            self.display_image(self.enhanced_image, self.enhanced_label)
            self.show_histogram(self.enhanced_image, self.enhanced_hist_label)
            self.display_image_properties(self.enhanced_image, self.enhanced_properties_label, "Enhanced Image Properties :")

            # Calculate SSIM and PSNR
            ssim_value = self.calculate_ssim(self.original_image, self.enhanced_image)
            psnr_value = self.calculate_psnr(self.original_image, self.enhanced_image)

            print(f"SSIM: {ssim_value:.4f}, PSNR: {psnr_value:.4f} dB")  # Debug log

            self.ssim_label.setText(f"SSIM: {ssim_value:.4f}")
            self.psnr_label.setText(f"PSNR: {psnr_value:.4f} dB")

    def calculate_ssim(self, img1, img2):
        """Calculate SSIM between two images using skimage's structural_similarity."""
        # Ensure both images are uint8 and within the valid range
        img1 = np.clip(img1, 0, 255).astype(np.uint8)
        img2 = np.clip(img2, 0, 255).astype(np.uint8)

        # Convert to grayscale
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Compute SSIM using skimage's structural_similarity function
        ssim_value, _ = ssim(img1_gray, img2_gray, full=True)

        return ssim_value

    def calculate_psnr(self, img1, img2):
        """Calculate PSNR between two images."""
        # Ensure both images are uint8 and within the valid range
        img1 = np.clip(img1, 0, 255).astype(np.uint8)
        img2 = np.clip(img2, 0, 255).astype(np.uint8)
        
        # Compute MSE (Mean Squared Error)
        mse = np.mean((img1 - img2) ** 2)
        
        # # If MSE is zero (no difference), return infinity (perfect match)
        if mse == 0:
            return float('100')
        
        # Calculate PSNR using the formula
        psnr_value = 20 * log10(255.0 / sqrt(mse))
        return psnr_value
    
    def calculate_refined_transmission(self, image_float):
        # Simple transmission calculation based on dark channel prior (DCP) idea.
        # This is a simplified approach, and can be refined further depending on the method you are using.

        # Calculate dark channel of the image (minimum of each pixel across the channels)
        dark_channel = np.min(image_float, axis=-1)

        # Estimate the transmission using the dark channel
        transmission = 1 - 0.95 * dark_channel  # Adjust the scaling factor (0.95) based on your method
        
        # Optionally apply a refinement step (e.g., using a guided filter, or bilateral filter)
        # For now, we can just normalize the transmission
        transmission = np.clip(transmission, 0.1, 1)  # Prevent transmission from going below a certain value
        
        return transmission

    def apply_dcp(self):
        if self.enhanced_image is not None:
            
            # Convert the image to float32 for processing
            image_float = self.enhanced_image.astype(np.float32) / 255.0

            # Calculate the atmospheric light A (maximum intensity in the image)
            # This is a common approach in dehazing algorithms like DCP
            A = np.max(image_float, axis=(0, 1))  # Shape becomes (3,) if RGB image
            
            # If A is a 1D array of shape (3,), you can expand it to match the shape (1024, 681, 3)
            A = np.expand_dims(A, axis=(0, 1))  # Shape becomes (1, 1, 3)

            # Calculate the transmission map (refined_transmission should also be calculated or defined here)
            refined_transmission = self.calculate_refined_transmission(image_float)  # This is just a placeholder

            # Expand refined_transmission to match the shape of image_float
            refined_transmission = np.expand_dims(refined_transmission, axis=-1)  # Shape becomes (1024, 681, 1)

            # Now the shapes of image_float, A, and refined_transmission match
            dehazed_image = (image_float - A) / refined_transmission + A

            # Convert back to uint8
            dehazed_image = np.clip(dehazed_image * 255.0, 0, 255).astype(np.uint8)

            # Display the enhanced image
            self.enhanced_image = dehazed_image
            
            self.display_image(self.enhanced_image, self.enhanced_label)
            self.show_histogram(self.enhanced_image, self.enhanced_hist_label)
            self.display_image_properties(self.enhanced_image, self.enhanced_properties_label, "Enhanced Image Properties :")

            # Calculate SSIM and PSNR after dehazing
            ssim_value = self.calculate_ssim(self.original_image, self.enhanced_image)
            psnr_value = self.calculate_psnr(self.original_image, self.enhanced_image)

            # Update the SSIM and PSNR labels with new values
            self.ssim_label.setText(f"SSIM: {ssim_value:.4f}")
            self.psnr_label.setText(f"PSNR: {psnr_value:.4f} dB")

    def apply_hdr(self):
        if self.enhanced_image is not None:
            # Convert to float32 for HDR processing
            hdr_image = self.enhanced_image.astype(np.float32) / 255.0

            # Apply Reinhard tone mapping to the HDR image
            tonemap = cv2.createTonemapReinhard(2.2, 0, 0, 0)
            hdr_image = tonemap.process(hdr_image)

            # Convert back to uint8
            hdr_image = np.clip(hdr_image * 255, 0, 255).astype(np.uint8)

            # Display the enhanced image
            self.enhanced_image = hdr_image
            
            self.display_image(self.enhanced_image, self.enhanced_label)
            self.show_histogram(self.enhanced_image, self.enhanced_hist_label)
            self.display_image_properties(self.enhanced_image, self.enhanced_properties_label, "Enhanced Image Properties :")

            # Calculate SSIM and PSNR after HDR enhancement
            ssim_value = self.calculate_ssim(self.original_image, self.enhanced_image)
            psnr_value = self.calculate_psnr(self.original_image, self.enhanced_image)

            # Update the SSIM and PSNR labels with new values
            self.ssim_label.setText(f"SSIM: {ssim_value:.4f}")
            self.psnr_label.setText(f"PSNR: {psnr_value:.4f} dB")
        
    def combine_methods(self):
        if self.enhanced_image is not None:
            # Convert the original image to float32 for processing
            image_float = self.enhanced_image.astype(np.float32) / 255.0
            
            # Step 1: Apply DCP (Dehazing)
            # Calculate atmospheric light (A)
            A = np.max(image_float, axis=(0, 1))  # (3,) shape if RGB image
            A = np.expand_dims(A, axis=(0, 1))  # (1, 1, 3) shape for broadcasting
            
            # Calculate the refined transmission map
            refined_transmission = self.calculate_refined_transmission(image_float)  # Placeholder function
            
            refined_transmission = np.expand_dims(refined_transmission, axis=-1)  # (1024, 681, 1)
            
            # Apply dehazing formula
            dehazed_image = (image_float - A) / refined_transmission + A
            
            # Step 2: Apply HDR (High Dynamic Range)
            # Apply Reinhard tone mapping to enhance the dynamic range
            tonemap = cv2.createTonemapReinhard(2.2, 0, 0, 0)
            hdr_image = tonemap.process(dehazed_image)  # Process the dehazed image

            # Convert back to uint8
            final_image = np.clip(hdr_image * 255, 0, 255).astype(np.uint8)

            # Display the combined enhanced image
            self.enhanced_image = final_image
            
            self.display_image(self.enhanced_image, self.enhanced_label)
            self.show_histogram(self.enhanced_image, self.enhanced_hist_label)
            self.display_image_properties(self.enhanced_image, self.enhanced_properties_label, "Enhanced Image Properties :")

            # Calculate SSIM and PSNR after combining methods
            ssim_value = self.calculate_ssim(self.original_image, self.enhanced_image)
            psnr_value = self.calculate_psnr(self.original_image, self.enhanced_image)

            # Update the SSIM and PSNR labels with new values
            self.ssim_label.setText(f"SSIM: {ssim_value:.4f}")
            self.psnr_label.setText(f"PSNR: {psnr_value:.4f} dB")



if __name__ == "__main__":
    app = QApplication(sys.argv)
    splash = SplashScreen("mpicon.jpeg", display_time=3000)
    splash.show()
    main_window = ImageEnhancementApp()
    main_window.show()
    sys.exit(app.exec_())