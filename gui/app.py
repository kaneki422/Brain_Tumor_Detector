import sys, os
import numpy as np
import torch
import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PIL import Image, ImageQt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocessing import preprocess_image
from feature_extraction import get_feature_vector
from traditional_ml import predict_ml
from cnn_model import BrainTumorCNN

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BrainTumorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Brain Tumor Detection System")
        self.setMinimumSize(1000, 700)
        self.image_path = None
        self.cnn_model = self._load_cnn()
        self._build_ui()

    def _load_cnn(self):
        model = BrainTumorCNN().to(DEVICE)
        try:
            model.load_state_dict(torch.load('models/cnn.pth', map_location=DEVICE))
            model.eval()
        except FileNotFoundError:
            QMessageBox.warning(self, "Warning", "CNN model not found. Train first.")
        return model

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # ── Left Panel ──────────────────────────────────────────
        left = QVBoxLayout()

        title = QLabel("🧠 Brain Tumor Detection")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        left.addWidget(title)

        self.img_label = QLabel("No image loaded")
        self.img_label.setFixedSize(380, 380)
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setStyleSheet(
            "border: 2px dashed #aaa; background: #f5f5f5;")
        left.addWidget(self.img_label)

        btn_load = QPushButton("📂  Load MRI Image")
        btn_load.setFixedHeight(40)
        btn_load.clicked.connect(self.load_image)
        left.addWidget(btn_load)

        # Model selector
        model_group = QGroupBox("Select ML Model")
        mg_layout = QVBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(
            ['SVM', 'KNN', 'Naive_Bayes', 'Random_Forest', 'CNN'])
        mg_layout.addWidget(self.model_combo)
        model_group.setLayout(mg_layout)
        left.addWidget(model_group)

        btn_predict = QPushButton("🔍  Detect Tumor")
        btn_predict.setFixedHeight(45)
        btn_predict.setStyleSheet(
            "background:#2196F3; color:white; font-size:14px; border-radius:5px;")
        btn_predict.clicked.connect(self.predict)
        left.addWidget(btn_predict)

        btn_compare = QPushButton("📊  Compare All Models")
        btn_compare.setFixedHeight(40)
        btn_compare.clicked.connect(self.compare_all)
        left.addWidget(btn_compare)

        left.addStretch()
        main_layout.addLayout(left, 40)

        # ── Right Panel ─────────────────────────────────────────
        right = QVBoxLayout()

        result_group = QGroupBox("Detection Result")
        rg_layout = QVBoxLayout()
        self.result_label = QLabel("—")
        self.result_label.setFont(QFont("Arial", 24, QFont.Bold))
        self.result_label.setAlignment(Qt.AlignCenter)
        self.confidence_label = QLabel("Confidence: —")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        rg_layout.addWidget(self.result_label)
        rg_layout.addWidget(self.confidence_label)
        result_group.setLayout(rg_layout)
        right.addWidget(result_group)

        features_group = QGroupBox("Extracted Features")
        fg_layout = QVBoxLayout()
        self.features_text = QTextEdit()
        self.features_text.setReadOnly(True)
        self.features_text.setFont(QFont("Courier", 10))
        fg_layout.addWidget(self.features_text)
        features_group.setLayout(fg_layout)
        right.addWidget(features_group)

        compare_group = QGroupBox("Model Comparison")
        cg_layout = QVBoxLayout()
        self.compare_table = QTableWidget(5, 3)
        self.compare_table.setHorizontalHeaderLabels(
            ['Model', 'Prediction', 'Confidence'])
        self.compare_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        cg_layout.addWidget(self.compare_table)
        compare_group.setLayout(cg_layout)
        right.addWidget(compare_group)

        main_layout.addLayout(right, 60)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select MRI Image", "",
            "Images (*.jpg *.jpeg *.png *.bmp)")
        if path:
            self.image_path = path
            pixmap = QPixmap(path).scaled(
                380, 380, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.img_label.setPixmap(pixmap)
            self.img_label.setText("")

    def _run_inference(self, model_name):
        norm, raw = preprocess_image(self.image_path)
        features  = get_feature_vector(norm)

        if model_name == 'CNN':
            img_t = torch.FloatTensor(norm).unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out  = self.cnn_model(img_t)
                prob = torch.softmax(out, dim=1)[0]
                pred = out.argmax(1).item()
                conf = prob[pred].item()
        else:
            pred, conf = predict_ml(features, model_name)

        label = "🔴 TUMOR DETECTED" if pred == 1 else "🟢 NO TUMOR"
        return label, conf, features

    def predict(self):
        if not self.image_path:
            QMessageBox.warning(self, "No Image", "Please load an MRI image first.")
            return
        model_name = self.model_combo.currentText()
        label, conf, features = self._run_inference(model_name)

        self.result_label.setText(label)
        color = "#e53935" if "TUMOR" in label else "#43a047"
        self.result_label.setStyleSheet(f"color:{color};")
        self.confidence_label.setText(f"Confidence: {conf*100:.1f}%")

        feat_str = "\n".join(
            [f"  {k}: {v:.6f}" for k, v in zip(
                ['Energy','Contrast','Correlation','Homogeneity',
                 'Dissimilarity','Mean','Std','Entropy','Skewness','Kurtosis'],
                features)])
        self.features_text.setText(feat_str)

    def compare_all(self):
        if not self.image_path:
            QMessageBox.warning(self, "No Image", "Please load an MRI image first.")
            return
        models = ['SVM', 'KNN', 'Naive_Bayes', 'Random_Forest', 'CNN']
        self.compare_table.setRowCount(len(models))
        for i, m in enumerate(models):
            try:
                label, conf, _ = self._run_inference(m)
                self.compare_table.setItem(i, 0, QTableWidgetItem(m))
                self.compare_table.setItem(i, 1, QTableWidgetItem(label))
                self.compare_table.setItem(i, 2,
                    QTableWidgetItem(f"{conf*100:.1f}%"))
            except Exception as e:
                self.compare_table.setItem(i, 0, QTableWidgetItem(m))
                self.compare_table.setItem(i, 1, QTableWidgetItem("Error"))
                self.compare_table.setItem(i, 2, QTableWidgetItem(str(e)))

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = BrainTumorApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()