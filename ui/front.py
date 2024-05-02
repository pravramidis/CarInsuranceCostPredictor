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

class DatePickerMixin:
    date_dialog = None

    def show_date_picker(self, widget, focus):
        if not focus:
            return
        
        self.trigger_widget = widget

        if not self.date_dialog:
            self.date_dialog = MDDockedDatePicker()
            self.date_dialog.bind(on_ok=self.on_ok, on_cancel=self.on_cancel)
        
        # Positioning the dialog based on the widget's properties
        self.date_dialog.pos = [
            widget.center_x - self.date_dialog.width / 2,
            widget.y - (self.date_dialog.height - dp(142))
        ]
        self.date_dialog.open()

    def on_ok(self, instance):
        value = instance.get_date()[0]
        if value:
            formatted_date = value.strftime('%m/%d/%Y')
            if hasattr(self, 'trigger_widget') and self.trigger_widget:
                self.trigger_widget.text = formatted_date
            self.date_dialog.dismiss()

    def on_cancel(self,instance):
        if self.date_dialog:
            self.date_dialog.dismiss()

class DropdownMenuMixin:
    def create_menu(self, textfield, menu_items):
        """Create and return a dropdown menu for the specified text field and menu items."""
        menu = MDDropdownMenu(
            caller=textfield,
            items=menu_items,
            width_mult=4,
            position='bottom'
        )
        return menu

    def open_menu(self, menu):
        """Open the specified menu."""
        if menu:
            menu.open()

    def set_item(self, text, menu, text_field):
        """Set the selected item's text to the TextField and dismiss the menu."""
        text_field.text = text
        menu.dismiss()

#Define different screens
class LoginScreen(Screen):
    def __init__(self, **kwargs):
        super(LoginScreen, self).__init__(**kwargs)
        
        fl = FloatLayout()
        bg = Image(source='login_background.png', allow_stretch=True, keep_ratio=False)
        fl.add_widget(bg)
        self.add_widget(fl)

class MainScreen(Screen, DatePickerMixin, DropdownMenuMixin):
    def __init__(self, **kwargs):
        super(MainScreen, self).__init__(**kwargs)
                
        
        fl = FloatLayout()
        bg = Image(source='main_background.png', allow_stretch=True, keep_ratio=False)
        fl.add_widget(bg)
        self.add_widget(fl)

        self.menu = None
   
    
    def on_leave(self):
        # Make sure to clean up by dismissing the menu if it's open
        if self.menu:
            self.menu.dismiss()
    
    def on_dropdown_focus(self, instance, value):
        area_field = self.ids.area_field
            
        menu_items = [
            {"text": "Rural", "on_release": lambda x="Rural": self.set_item(x, self.menu, area_field)},
            {"text": "Urban", "on_release": lambda x="Urban": self.set_item(x, self.menu, area_field)}
        ]
        
        self.menu = self.create_menu(area_field, menu_items)
        self.open_menu(self.menu) 
            

    def on_text_field_focus(self, instance, value):
        if value:  # if the text field is focused
            self.show_date_picker(instance,value)
    

class VehicleScreen(Screen, DropdownMenuMixin):
    def __init__(self, **kwargs):
        super(VehicleScreen, self).__init__(**kwargs)
                
        
        fl = FloatLayout()
        bg = Image(source='main_background.png', allow_stretch=True, keep_ratio=False)
        fl.add_widget(bg)
        self.add_widget(fl)

    def on_dropdown_focus(self, instance, value):
        if instance == self.ids.type_field:
            menu_items = [
            {"text": "Passenger Car", "on_release": lambda x="Passenger Car": self.set_item(x,self.menu, instance)},
            {"text": "Motorbike", "on_release": lambda x="Motorbike": self.set_item(x,self.menu, instance)},
            {"text": "Van", "on_release": lambda x="Van": self.set_item(x,self.menu, instance)},
            {"text": "Agricultural Vehicle", "on_release": lambda x="Agricultural Vehicle": self.set_item(x,self.menu, instance)}
            ]
        elif instance == self.ids.fuel_field:   
            menu_items = [
            {"text": "Petrol", "on_release": lambda x="Petrol": self.set_item(x,self.menu, instance)},
            {"text": "Diesel", "on_release": lambda x="Diesel": self.set_item(x,self.menu, instance)}
            ]
        self.menu = self.create_menu(instance, menu_items)
        self.open_menu(self.menu) 
    

class InsuranceScreen(Screen, DatePickerMixin, DropdownMenuMixin):
    def __init__(self, **kwargs):
        super(InsuranceScreen, self).__init__(**kwargs)
                
        
        fl = FloatLayout()
        bg = Image(source='main_background.png', allow_stretch=True, keep_ratio=False)
        fl.add_widget(bg)
        self.add_widget(fl)

    def on_text_field_focus(self, instance, value):
        if value:  # if the text field is focused
            self.show_date_picker(instance,value)
    
    def on_dropdown_focus(self, instance, value):
        if instance == self.ids.driver:
            menu_items = [
            {"text": "Yes", "on_release": lambda x="Yes": self.set_item(x,self.menu, instance)},
            {"text": "No", "on_release": lambda x="No": self.set_item(x,self.menu, instance)}
            ]
        elif instance == self.ids.channel:   
            menu_items = [
            {"text": "Agent", "on_release": lambda x="Agent": self.set_item(x,self.menu, instance)},
            {"text": "Insurance Brokers", "on_release": lambda x="Insurance Brokers": self.set_item(x,self.menu, instance)}
            ]
        elif instance == self.ids.payment:
            menu_items = [
            {"text": "Half-yearly", "on_release": lambda x="Half-yearly": self.set_item(x,self.menu, instance)},
            {"text": "Annual", "on_release": lambda x="Annual": self.set_item(x,self.menu, instance)}
            ]
        self.menu = self.create_menu(instance, menu_items)
        self.open_menu(self.menu) 



class PriceScreen(Screen):
    def __init__(self, **kwargs):
        super(PriceScreen, self).__init__(**kwargs)
                
        
        fl = FloatLayout()
        bg = Image(source='main_background.png', allow_stretch=True, keep_ratio=False)
        fl.add_widget(bg)
        self.add_widget(fl)

class ScreenManager(ScreenManager):
    pass



class SafeWheels(MDApp):
    def build(self):
        self.theme_cls.theme_style = "Light"
        self.theme_cls.primary_palette = "Blue"
        return Builder.load_file('front.kv')


if __name__ == '__main__':
    SafeWheels().run()