"""
一个基于OpenCV的简易渲染查看器
https://github.com/zuoxingdong/dm2gym
"""

import uuid
import cv2

class OpenCVImageViewer:
    """
    一个基于OpenCV highgui 的简易 dm_control 图像查看器 
    此类旨在作为`gym.envs.classic_control.rendering.SimpleImageViewer`的直接替代品。
    """
    def __init__(self, *, escape_to_exit=False):
        """
        Construct the viewing window
        """
        self._escape_to_exit = escape_to_exit
        self._window_name = str(uuid.uuid4())
        cv2.namedWindow(self._window_name, cv2.WINDOW_AUTOSIZE)
        self._isopen = True
        
    def __del__(self):
        """
        Close the window
        """
        cv2.destroyWindow(self._window_name)
        self._isopen = False
        
    def imshow(self, image):
        """
        show an image
        """
        cv2.imshow(self._window_name, image[:, :, [2, 1, 0]]) # OpenCV default mode: B G R
        # 具体来说，cv2.waitKey(1) 会等待 1 毫秒以检测键盘事件，并返回按下的键的 ASCII 码。如果返回的值是 27（即 ESC 键）
        if cv2.waitKey(1) in [27] and self._escape_to_exit:
            exit()
        
    @property # 可作为属性访问
    def isopen(self):
        """
        Is the window open?
        """
        return self._isopen
    
    def close(self):
        pass
        