
import wx
import numpy as np
import cv2
from PIL import Image
from bresenham import bresenham
import matplotlib
import matplotlib.pyplot as plt
import mplcursors
from wx.core import Cursor
matplotlib.use('WXAgg')

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
from matplotlib.figure import Figure


class GraphFrame(wx.Frame):

    def __init__(self, title: str, coord_array: np.ndarray, parent=None):
        # visualization of lineprofile
        wx.Frame.__init__(self, parent=parent, title=title)
        self.coord_array = coord_array
        l = 4 if self.coord_array.shape[0] == 3 else self.coord_array.shape[0]
        self.figure, self.axes = plt.subplots(l, sharex=True, sharey=False)
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        self.SetSizer(self.sizer)
        self.Fit()
        #print(self.coord_array)
        self.InitUI()

    def InitUI(self):
        # check for dimensions
        if self.coord_array.shape[0] == 3:
            self.axes[0].plot(range(self.coord_array.shape[1]), self.coord_array[0, :], color='red')
            self.axes[1].plot(range(self.coord_array.shape[1]), self.coord_array[1, :], color='green')
            self.axes[2].plot(range(self.coord_array.shape[1]), self.coord_array[2, :], color='blue')
            self.axes[3].plot(range(self.coord_array.shape[1]), [x/3 for x in self.coord_array.sum(axis=0)], color='black')
        if self.coord_array.shape[0] > 3:
            for c, v in enumerate(self.coord_array):
                colors = ['red', 'green', 'blue']
                self.axes[c].plot(range(self.coord_array.shape[1]), v, color=colors[c%3])
        mplcursors.cursor(multiple=True)
        self.Show(True)

class LineProfile(wx.Frame):

    def __init__(self, parent, title):
        super(LineProfile, self).__init__(parent, title=title, size=(1100, 1100))
        self.image = None
        self.mouse_pos = (0, 0)
        self.transform = (1, 1)
        self.imageBitmap = None
        self.isDrag = False
        self.dc = None
        self.Centre()
        self.Move(100, 100)
        self.Maximize(True)
        self.InitUI()

    def InitUI(self):
        # init ui
        mb = wx.MenuBar()
        fm = wx.Menu()
        open = wx.MenuItem(fm, wx.ID_OPEN, '&Open')
        fm.Append(open)
        self.Bind(wx.EVT_MENU, self.OnOpen, open)

        mb.Append(fm, '&File')
        self.SetMenuBar(mb)

    def OnOpen(self, event):
        # 'choose file' routine
        with wx.FileDialog(self, "Open PNG File", wildcard="PNG (*.png)|*.png",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:

            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return     # the user changed their mind

            # Proceed loading the file chosen by the user
            newfile = fileDialog.GetPath()
            try:
                self.image = cv2.cvtColor(cv2.imread(newfile), cv2.COLOR_BGR2RGB)
                #self.image[:,:,0] = ((self.image[:,:,0] - self.image[:,:,0].min()) / self.image[:,:,0].max()
                scaled = cv2.resize(self.image, (900, 900))
                self.transform = (self.image.shape[0] / 900, self.image.shape[1] / 900)
                width, height = 510, 90
                self.imageBitmap = wx.StaticBitmap(self, -1, wx.Bitmap.FromBuffer(900, 900, scaled), (width, height))
                self.imageBitmap.Bind(wx.EVT_LEFT_DOWN, self.Image_OnMouse1)
                self.imageBitmap.Bind(wx.EVT_LEFT_UP, self.Image_OnMouse2)
                self.dc = wx.ClientDC(self.imageBitmap)
            except IOError:
                wx.LogError("Cannot open file '%s'." % newfile)

    def Image_OnMouse1(self, event):
        # mouse pos of line start
        ctrl_pos = event.GetPosition()
        print("ctrl_pos: " + str(ctrl_pos.x) + ", " + str(ctrl_pos.y))
        self.isDrag = True
        self.mouse_pos = ctrl_pos
        self.dc.DrawBitmap(self.imageBitmap.GetBitmap(), 0, 0)

    def Image_OnMouse2(self, event):
        # mouse pos of line end
        if self.isDrag == True:
            ctrl_pos = event.GetPosition()
            print("ctrl_pos: " + str(ctrl_pos.x) + ", " + str(ctrl_pos.y))
            self.isDrag = False

            # line calculation and processing
            realPoints = [int(self.mouse_pos.x * self.transform[1]),
                          int(self.mouse_pos.y * self.transform[0]),
                          int(ctrl_pos.x * self.transform[1]),
                          int(ctrl_pos.y * self.transform[0])]

            self.dc.SetPen(wx.Pen(wx.Colour(255, 0, 255), 1))
            self.dc.DrawLine(ctrl_pos[0], ctrl_pos[1], self.mouse_pos[0], self.mouse_pos[1])
            print(realPoints)

            points = list(bresenham(realPoints[0], realPoints[1], realPoints[2], realPoints[3]))
            coord_array = np.zeros((self.image.shape[2], len(points)))
            normalized = np.array([(x - x.min()) / x.max() if x.max() > 0 else 1 for x in self.image])
            for i, (x, y) in enumerate(points):
                for t, c in enumerate(normalized[y, x, :]):
                    coord_array[t, i] = c

            #print(coord_array)
            frame = GraphFrame('Graph', coord_array)

def main():

    app = wx.App()
    frame = LineProfile(None, title='Line Profile')
    frame.Show(True)

    app.MainLoop()


if __name__ == '__main__':
    main()
