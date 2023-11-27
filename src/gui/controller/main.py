from pathlib import Path
import sys

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QRadioButton, QButtonGroup,
    QGraphicsScene, QMessageBox
)
from PyQt5.QtCore import Qt, QObject, pyqtSignal, QThread
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
from non_cnn.model import get_image
from gui.views.mainWindow import Ui_MainGUI
from gui.models import MODELS
from PIL import Image
import cv2


class VideoWorker(QThread):
    changeFrame = pyqtSignal(int)  # curFrame
    changePlay = pyqtSignal(bool)  # isPlaying
    togglePlay = pyqtSignal()
    initialized = pyqtSignal(int)  # totalFrame
    progress = pyqtSignal(int, Image.Image)  # curFrame, prediction
    playToggled = pyqtSignal(bool)  # isPlaying

    changeModel = pyqtSignal(dict)  # model

    def __init__(self, media, model):
        super().__init__()
        self.model = model
        self.media = cv2.VideoCapture(media)
        self.isPlaying = False
        self.curFrame = 0
        self.totalFrame = int(self.media.get(cv2.CAP_PROP_FRAME_COUNT))
        self.initialized.emit(self.totalFrame)
        self.changeFrame.connect(self._seek)
        self.changePlay.connect(self._changePlay)
        self.togglePlay.connect(self._togglePlay)
        self.changeModel.connect(self._changeModel)
        self.playToggled.emit(self.isPlaying)
        # process first frame
        self._process()

    def _changeModel(self, model):
        self.model = model

    def _togglePlay(self):
        print('toggle play')
        self.isPlaying = not self.isPlaying
        self.playToggled.emit(self.isPlaying)

    def _changePlay(self, isPlaying):
        print('change play', isPlaying)
        self.isPlaying = isPlaying
        self.playToggled.emit(self.isPlaying)

    def _seek(self, frame_num):
        if frame_num > -1:
            self.media.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            self.curFrame = frame_num
        # process current frame
        self._process()

    def _process(self):
        if self.media.isOpened():
            ret, frame = self.media.read()
            if not ret:
                return False
            m = self.model
            predictions = m['predict'](m['model'], frame)
            self.progress.emit(self.curFrame, predictions)
            self.curFrame += 1
            return True
        return False

    def run(self):
        while True:
            if not self.isPlaying:
                continue
            cur_play = self._process()
            if self.isPlaying != cur_play:
                self.isPlaying = cur_play
                self.playToggled.emit(self.isPlaying)


class Window(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainGUI()
        self.ui.setupUi(self)
        self.connectSignalsSlots()
        self.setupModel()
        self.setup()

        self.media = None
        self.is_video = False
        self.media_thread: QThread = None
        self.media_worker: VideoWorker

    def setup(self):
        self.ui.mediaCntWidget.hide()
        self.ui.predictWidget.hide()

    def setupModel(self):
        self.current_model = list(MODELS.keys())[0]
        self.models = MODELS.copy()
        for k, v in self.models.items():
            if 'from' in v:
                v['model'] = self.models[v['from']]['model']
            else:
                v['model'] = v['loader'](
                    Path(__file__).parent /
                    f"../../{v['type']}/models/{v['path']}",
                )
        btn_group = QButtonGroup(self.ui.modelWidget)
        mdl_layout = self.ui.modelWidget.layout()

        def make_change_mdl(k):
            return lambda: self.changeModel(k)
        for k in self.models.keys():
            btn = QRadioButton(k, self.ui.modelWidget)
            change_mdl = make_change_mdl(k)
            btn.clicked.connect(change_mdl)
            btn_group.addButton(btn)
            mdl_layout.addWidget(btn)
            self.models[k]['button'] = btn
        btn_group.buttons()[0].setChecked(True)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)

        scene = self.ui.graphicsView.scene()
        if scene:
            self.ui.graphicsView.fitInView(
                scene.sceneRect(),
                Qt.AspectRatioMode.KeepAspectRatio,
            )

    def resetPrediction(self):
        if not self.media:
            return
        if self.is_video:
            self.media_worker.changePlay.emit(False)
            self.media_worker.changeFrame.emit(0)
        else:
            self.setImage(self.media)
            self.ui.predictLabel.setText('Click Predict to get prediction')

    def changeModel(self, model_name):
        if self.is_video and self.notSupportVideo(model_name):
            self.models[self.current_model]['button'].setChecked(True)
            return
        if self.is_video and self.media:
            self.media_worker.changeModel.emit(
                self.models[model_name]
            )
        self.current_model = model_name
        print('change model', model_name)
        self.resetPrediction()

    def connectSignalsSlots(self):
        self.ui.browseFileBtn.clicked.connect(self.browseFile)
        self.ui.predictBtn.clicked.connect(self.predict)
        self.ui.playBtn.clicked.connect(self.playPause)
        self.ui.stopBtn.clicked.connect(self.stop)

    def notSupportVideo(self, mdl=None):
        if mdl is None:
            mdl = self.current_model
        if not self.models[mdl]['video']:
            QMessageBox.critical(
                self,
                'Invalid Model',
                f'Selected model {mdl} does not support video',
                QMessageBox.Ok,
            )
            return True
        return False

    def browseFile(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select Image/Video")
        if file_path:
            if file_path.endswith(('.mp4', '.avi')):
                if self.notSupportVideo():
                    return
                self.ui.mediaCntWidget.show()
                self.ui.predictWidget.hide()
                self.is_video = True
                self.initVideo(file_path)
            elif file_path.endswith(('.png', '.jpg', '.jpeg')):
                self.ui.mediaCntWidget.hide()
                self.ui.predictWidget.show()
                self.is_video = False
            else:
                raise ValueError(f'Invalid file type: {file_path}')
            self.media = file_path
            self.resetPrediction()
            self.ui.filePathTxt.setText(file_path)

    def initVideo(self, file_path):
        if self.media_thread:
            self.media_thread.quit()
        self.media_thread = QThread()
        self.media_worker = VideoWorker(
            file_path,
            self.models[self.current_model],
        )
        self.media_worker.moveToThread(self.media_thread)
        self.media_thread.started.connect(self.media_worker.run)
        self.media_thread.finished.connect(self.media_worker.deleteLater)
        self.media_worker.initialized.connect(
            lambda total_frame: self.ui.seekSlider.setRange(0, total_frame)
        )
        self.media_worker.progress.connect(
            lambda cur_frame, prediction: self.setMediaControl(
                curFrame=cur_frame,
                media=prediction,
            )
        )
        self.media_worker.playToggled.connect(
            lambda is_playing: self.setMediaControl(isPlaying=is_playing)
        )
        self.ui.seekSlider.sliderMoved.connect(
            lambda frame_num: self.media_worker.changeFrame.emit(
                frame_num
            )
        )
        self.media_thread.start()

    def setImage(self, file):
        scene = QGraphicsScene(self.ui.graphicsView)
        if isinstance(file, str):
            pixmap = QPixmap(file)
        elif isinstance(file, QPixmap):
            pixmap = file
        elif isinstance(file, Path):
            pixmap = QPixmap(str(file))
        elif isinstance(file, bytes):
            pixmap = QPixmap()
            pixmap.loadFromData(file)
        elif isinstance(file, QImage):
            pixmap = QPixmap(file)
        elif isinstance(file, np.ndarray):
            h, w, ch = file.shape
            bytes_per_line = ch * w
            pixmap = QPixmap(QImage(
                file.data,
                w,
                h,
                bytes_per_line,
                QImage.Format_RGB888,
            ))
        elif isinstance(file, Image.Image):
            pixmap = QPixmap(QImage(
                file.tobytes(),
                file.size[0],
                file.size[1],
                QImage.Format_RGB888,
            ))
        else:
            raise ValueError(f'Invalid file type: {type(file)}')
        scene.addPixmap(pixmap)
        self.ui.graphicsView.setScene(scene)
        self.ui.graphicsView.fitInView(scene.sceneRect(),
                                        Qt.AspectRatioMode.KeepAspectRatio)

    def setMediaControl(self, curFrame=None, isPlaying=None, media=None):
        if media:
            print("get media frame:", curFrame)
            self.setImage(media)
        if curFrame:
            total_frame = self.media_worker.totalFrame
            self.ui.seekSlider.setValue(curFrame)
            self.ui.timeLbl.setText(f'{curFrame}/{total_frame}')
        if isPlaying is not None:
            if isPlaying:
                self.ui.playBtn.setText('Pause')
            else:
                self.ui.playBtn.setText('Play')

    def playPause(self):
        self.media_worker.togglePlay.emit()

    def stop(self):
        self.resetPrediction()

    def predict(self):
        file_path = self.ui.filePathTxt.text()
        if not file_path:
            return
        m = self.models[self.current_model]
        im_loader = get_image if 'img_loader' not in m else m['img_loader']
        predictions = m['predict'](m['model'], im_loader(file_path))
        if not m['multilabel']:
            # Predictions must be dict probabilities for each class
            predictions = ', '.join([
                f'{k} ({v*100:.2f}%)' for k, v in predictions.items()
            ])
            self.ui.predictLabel.setText(
                f'Predictions: {predictions}'
            )
        else:
            # Predictions must be the annotated image
            self.setImage(predictions)
            self.ui.predictLabel.setText(
                'Done predicting. Check the image.'
            )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec())
