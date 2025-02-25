import cv2
import threading
import queue

class VideoRecorder:
    def __init__(self, filename="recorded_video.avi", fps=10, frame_size=None):
        """
        Initializes the video recorder.
        If frame_size is None, the recorder will use the size of the first received frame.
        """
        self.filename = filename
        self.fps = fps
        self.frame_size = frame_size  # (width, height)
        self.frame_queue = queue.Queue()
        self.out = None
        self.recording = True
        self.thread = threading.Thread(target=self.record_loop)
        self.thread.daemon = True
        self.thread.start()
    
    def record_loop(self):
        while self.recording or not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get(timeout=1)
                if self.out is None:
                    if self.frame_size is None:
                        self.frame_size = (frame.shape[1], frame.shape[0])
                    fourcc = cv2.VideoWriter_fourcc(*"XVID")
                    self.out = cv2.VideoWriter(self.filename, fourcc, self.fps, self.frame_size)
                if (frame.shape[1], frame.shape[0]) != self.frame_size:
                    frame = cv2.resize(frame, self.frame_size)
                self.out.write(frame)
            except queue.Empty:
                continue
        if self.out is not None:
            self.out.release()
    
    def record_frame(self, frame):
        self.frame_queue.put(frame)
    
    def stop(self):
        self.recording = False
        self.thread.join()
        print(f"Recording stopped. Video saved to {self.filename}")
