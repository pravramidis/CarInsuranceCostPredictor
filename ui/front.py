import kivy
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.core.window import Window
from kivymd.app import MDApp





#Define different screens
class LoginScreen(Screen):
    def __init__(self, **kwargs):
        super(LoginScreen, self).__init__(**kwargs)
        #Window.size = (800, 600)  # Width = 800, Height = 600
        #Window.resizable = False
        
        
        fl = FloatLayout()
        bg = Image(source='login_background.png', allow_stretch=True, keep_ratio=False)
        fl.add_widget(bg)
        self.add_widget(fl)

class MainScreen(Screen):
    pass

class ScreenManager(ScreenManager):
    pass



class SafeWheels(MDApp):
    def build(self):
        self.theme_cls.theme_style = "Light"
        self.theme_cls.primary_palette = "Blue"
        return Builder.load_file('front.kv')


if __name__ == '__main__':
    SafeWheels().run()