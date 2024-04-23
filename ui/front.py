import kivy
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.core.window import Window
from kivymd.app import MDApp
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.pickers import MDDockedDatePicker
from kivy.metrics import dp




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
    def __init__(self, **kwargs):
        super(MainScreen, self).__init__(**kwargs)
                
        
        fl = FloatLayout()
        bg = Image(source='main_background.png', allow_stretch=True, keep_ratio=False)
        fl.add_widget(bg)
        self.add_widget(fl)

        self.menu = None
    
    def open_area_menu(self):
        if self.menu:
            self.menu.open()

    def on_enter(self):
        # Assuming the TextField id is 'area_field' in your .kv file
        area_field = self.ids.area_field
        
        menu_items = [
            {"text": "Rural", "on_release": lambda x="Rural": self.set_item(x)},
            {"text": "Urban", "on_release": lambda x="Urban": self.set_item(x)}
        ]
        
        self.menu = MDDropdownMenu(
            caller=area_field,  # Caller is now the TextField
            items=menu_items,
            width_mult=4,
            position="bottom"
        )

    def set_item(self, text):
        # This method sets the selected item's text to the TextField
        self.ids.area_field.text = text
        self.menu.dismiss()
    
    def on_cancel(self, *args):
        if self.date_dialog:
            self.date_dialog.dismiss()

    def on_ok(self, *args):
        date_list = self.date_dialog.get_date()
        if date_list:  # Check if the list is not empty
            selected_date = date_list[0]  # Extract the first item (the date object)
            formatted_date = selected_date.strftime('%m/%d/%Y')
            print(formatted_date)
        
            self.ids.dob.text = formatted_date
            self.ids.licence_issue_date_field.text = formatted_date

            
            
            
        print(self.date_dialog.get_date())
        self.on_cancel()

    def show_date_picker(self, widget, focus):
        if not focus:
            return
        
        
        date_dialog = MDDockedDatePicker()
        self.date_dialog = date_dialog
        self.date_dialog.bind(
            on_ok=self.on_ok,
            on_cancel=self.on_cancel,
        )
        # Set the position based on the widget's properties
        date_dialog.pos = [
            widget.center_x - date_dialog.width / 2,
            widget.y - (date_dialog.height - dp(142)),
        ]


        date_dialog.open()


class ScreenManager(ScreenManager):
    pass



class SafeWheels(MDApp):
    def build(self):
        self.theme_cls.theme_style = "Light"
        self.theme_cls.primary_palette = "Blue"
        return Builder.load_file('front.kv')


if __name__ == '__main__':
    SafeWheels().run()